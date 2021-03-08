# pylint:disable=too-many-lines
"""
This module contains SAP HANA APL regression algorithm.

The following classes are available:

    * :class:`AutoRegressor`
"""
from collections import OrderedDict
import logging
import pandas as pd
from hdbcli import dbapi
from hana_ml.dataframe import (
    DataFrame,
    quotename)
from hana_ml.ml_base import execute_logged
from hana_ml.ml_exceptions import FitIncompleteError
from hana_ml.algorithms.apl.robust_regression_base import RobustRegressionBase

logger = logging.getLogger(__name__) #pylint: disable=invalid-name


class AutoRegressor(RobustRegressionBase):
    """
    This module provides the SAP HANA APL regression algorithm.

    Parameters
    ----------
    conn_context :  ConnectionContext, optional
        The connection object to an SAP HANA database.
        This parameter is not needed anymore.
        It will be set automatically when a dataset is used in fit() or predict().
    variable_auto_selection : bool optional
        When set to True, variable auto-selection is activated.
        Variable auto-selection enables to maintain the performance of a model while keeping
        the lowest number of variables
    polynomial_degree : int optional
        The polynomial degree of the model. Default is 1.
    variable_storages: dict optional
        Specifies the variable data types (string, integer, number).
        For example, {'VAR1': 'string', 'VAR2': 'number'}.
        See notes below for more details.
    variable_value_types: dict optional
        Specifies the variable value type (continuous, nominal, ordinal).
        For example, {'VAR1': 'continuous', 'VAR2': 'nominal'}.
        See notes below for more details.
    variable_missing_strings: dict, optional
        Specifies the variable values that will be taken as missing.
        For example, {'VAR1': '???'} means anytime the variable value equals to '???',
        it will be taken as missing.
    extra_applyout_settings: dict optional
        Defines other outputs the model should generate in addition to the predicted values.
        For example: {'APL/ApplyReasonCode':'3;Mean;Below;False'}
        will add reason codes in the output when the model is applied.
        These reason codes provide explanation about the prediction.
        See *OPERATION_CONFIG parameters* in *APPLY_MODEL function*, `SAP HANA APL Reference Guide
        <https://help.sap.com/viewer/p/apl>`_.
    other_params: dict optional
        Corresponds to advanced settings.
        The dictionary contains {<parameter_name>: <parameter_value>}.
        The possible parameters are:
           - 'correlations_lower_bound'
           - 'correlations_max_kept'
           - 'cutting_strategy'
           - 'exclude_low_predictive_confidence'
           - 'risk_fitting'
           - 'risk_fitting_min_cumulated_frequency'
           - 'risk_fitting_nb_pdo'
           - 'risk_fitting_use_weights'
           - 'risk_gdo'
           - 'risk_mode'
           - 'risk_pdo'
           - 'risk_score'
           - 'score_bins_count'
           - 'variable_auto_selection'
           - 'variable_selection_best_iteration'
           - 'variable_selection_min_nb_of_final_variables'
           - 'variable_selection_max_nb_of_final_variables'
           - 'variable_selection_mode'
           - 'variable_selection_nb_variables_removed_by_step'
           - 'variable_selection_percentage_of_contribution_kept_by_step'
           - 'variable_selection_quality_bar'
           - 'variable_selection_quality_criteria'
        See *Common APL Aliases for Model Training* in `SAP HANA APL Reference Guide
        <https://help.sap.com/viewer/p/apl>`_.
    other_train_apl_aliases: dict, optional
        Users can provide APL aliases as advanced settings to the model.
        Unlike 'other_params' described above, users are free to input any possible value.
        There is no control in python.

    Examples
    --------
    >>> from hana_ml.algorithms.apl.regression import AutoRegressor
    >>> from hana_ml.dataframe import ConnectionContext, DataFrame

    Connecting to SAP HANA Database

    >>> CONN = ConnectionContext('HDB_HOST', HDB_PORT, 'HDB_USER', 'HDB_PASS')
    >>> # -- Creates Hana DataFrame
    >>> hana_df = DataFrame(CONN, 'select * from APL_SAMPLES.CENSUS')

    Creating and fitting the model

    >>> model = AutoRegressor(variable_auto_selection=True)
    >>> model.fit(hana_df, label='age',
    ...      features=['workclass', 'fnlwgt', 'education', 'education-num', 'marital-status'],
    ...      key='id')

    Making a prediction

    >>> applyout_df = model.predict(hana_df)
    >>> print(applyout_df.head(5).collect())
              id  TRUE_LABEL  PREDICTED
    0         30          49         42
    1         63          48         42
    2         66          36         42
    3        110          42         42
    4        335          53         42

    Debriefing

    >>> model.get_performance_metrics()
    OrderedDict([('L1', 8.59885654599923), ('L2', 11.012352163260505)...

    >>> model.get_feature_importances()
    OrderedDict([('marital-status', 0.7916100739306074), ('education-num', 0.13524836400650087)

    Saving the model in the schema named 'MODEL_STORAGE'
    Please see model_storage class for further features of model storage

    >>> model_storage = ModelStorage(connection_context=CONN, schema='MODEL_STORAGE')
    >>> model.name = 'My regression model name'
    >>> model_storage.save_model(model=model, if_exists='replace')

    Reloading the model and making another prediction

    >>> model2 = AutoRegressor(conn_context=CONN)
    >>> model2.load_model(schema_name='MySchema', table_name='MyTable')

    >>> applyout2 = model2.predict(hana_df)
    >>> applyout2.head(5).collect()
              id  TRUE_LABEL  PREDICTED
    0         30          49         42
    1         63          48         42
    2         66          36         42
    3        110          42         42
    4        335          53         42

    Notes
    -----
    It is highly recommended to use a dataset with a key provided in the fit() method.
    If not, once the model is trained, it will not be possible anymore to use the predict() method
    with a key, because the model will not expect it.

    """

    def __init__(self,
                 conn_context=None,
                 variable_auto_selection=True,
                 polynomial_degree=None,
                 variable_storages=None,
                 variable_value_types=None,
                 variable_missing_strings=None,
                 extra_applyout_settings=None,
                 ** other_params): #pylint:disable=too-many-arguments
        super(AutoRegressor, self).__init__(
            conn_context,
            variable_auto_selection,
            polynomial_degree,
            variable_storages,
            variable_value_types,
            variable_missing_strings,
            extra_applyout_settings,
            ** other_params)

        # For classification, the target variable must be nominal
        self._force_target_var_type = 'continuous'

    # pylint: disable=too-many-arguments
    def fit(self, data,
            key=None,
            features=None,
            label=None,
            weight=None):
        """
        Fits the model.

        Parameters
        ----------
        data : DataFrame
            The training dataset
        key : str, optional
            The name of the ID column.
            If `key` is not provided,
            it is assumed that the input has no ID column.
        features : list of str, optional
            Names of the feature columns.
            If `features` is not provided, default will be to all the non-ID and non-label columns.
        label : str, optional
            The name of the label column. Default is the last column.
        weight : str, optional
            The name of the weight variable.
            A weight variable allows one to assign a relative weight to each of the observations.

        Returns
        -------
        self : object

        Notes
        -----
        It is highly recommended to use a dataset with a key provided in the fit() method.
        If not, once the model is trained, it will not be possible anymore to use the predict()
        method with a dataset with a key, because the model will not expect it.
        """
        if label is None:
            label = data.columns[-1]
        return self._fit(data=data,
                         key=key,
                         features=features,
                         label=label,
                         weight=weight)

    def _rewrite_applyout_df(self, data, applyout_df):
        """
        Rewrites the applyout dataframe so it outputs standardized column names.
        """
        # Find the name of the label column in applyout table (rr_XXXX)
        label_name = self._get_target_varname_from_applyout_table(
            applyout_df=applyout_df,
            target_varname=None
            )
        mapping = OrderedDict()
        mapping['?'] = '@FIRST_COL'  # whatever first column - key column
        mapping[label_name] = 'TRUE_LABEL'
        mapping['rr_' + label_name] = 'PREDICTED'

        sql = 'SELECT '
        i = 0
        # ---- maps columns
        applyout_table_columns = applyout_df.columns
        mapped_cols = []  # cols that are mapped
        for old_col in mapping.keys():
            new_col = mapping[old_col]
            if new_col == '@FIRST_COL':
                # key column
                new_col = applyout_table_columns[0]
                old_col = new_col
                mapped_cols.append(new_col)
            # TRUE_LABEL has to be ignored if it is not given as input
            if new_col == 'TRUE_LABEL':
                if old_col not in data.columns:
                    continue
            if old_col not in applyout_table_columns:
                raise Exception(
                    'Cannot find column {old_col} from the output'.format(
                        old_col=old_col))
            if i > 0:
                sql = sql + ', '
            sql = (sql + '{old_col} {new_col}'.format(
                old_col=quotename(old_col),
                new_col=quotename(new_col)))
            mapped_cols.append(old_col)
            i = i + 1
        # add extra columns for which we can't map (reason_code for example)
        for i, out_col in enumerate(applyout_df.columns):
            if out_col not in mapped_cols \
                    and out_col != label_name:  # label might be ignored it was not in apply-in
                sql = sql + ', '
                sql = (sql + '{ex_col} '.format(
                    ex_col=quotename(out_col)))

        sql = sql + ' FROM ' + self.applyout_table_.name
        applyout_df_new = DataFrame(connection_context=self.conn_context,
                                    select_statement=sql)
        # logger.info('DataFrame for predict ouput: ' + sql)
        return applyout_df_new

    def predict(self, data):
        """
        Makes prediction with a fitted model.
        It is possible to add special outputs, such as reason codes, by specifying
        extra_applyout_setting parameter in the model.
        This parameter is explained above in the model class section.

        Parameters
        ----------
        data : hana_ml DataFrame
            The dataset used for prediction

        Returns
        -------
        Prediction output: a hana_ml DataFrame.
        The dataframe contains the following columns:
        - KEY : the key column if it was provided in the dataset
        - TRUE_LABEL : the true value if it was provided in the dataset
        - PREDICTED : the predicted value
        """
        # APPLY_CONFIG
        apply_config_data_df = pd.DataFrame()
        if self.extra_applyout_settings is not None:
            # add extra options given by user
            for k in self.extra_applyout_settings:
                apply_config_data_df = apply_config_data_df.append(
                    [[k, self.extra_applyout_settings[k], None]])
        applyout_df = self._predict(data=data,
                                    apply_config_data_df=apply_config_data_df)
        return self._rewrite_applyout_df(data=data,
                                         applyout_df=applyout_df)

    def score(self, data):
        """
        Returns the coefficient of determination R^2 of the prediction.

        Parameters
        ----------
        data : hana_ml DataFrame
            The dataset used for prediction.
            It must contain the true value so that the score could be computed.

        Returns
        -------
            mean average accuracy: float

        """

        applyout_df = self.predict(data)

        # Find the name of the label column in applyout table
        true_y_col = 'TRUE_LABEL'
        pred_y_col = 'PREDICTED'

        # Check if the label column is given in input dataset (true label)
        # If it is not present, score can't be calculated
        if true_y_col not in applyout_df.columns:
            raise FitIncompleteError("Cannot find true label column in dataset")
        try:
            with self.conn_context.connection.cursor() as cur:
                # Count TP + TN
                sql = (
                    'SELECT 1- (SUM(POWER((applyout.{true_y_col} - applyout.{pred_y_col}), 2)))/'
                    + '(SUM(POWER((applyout.{true_y_col} - gdt.av), 2)))'
                    + ' FROM ({applyout_df}) applyout, '
                    + '  (select avg({true_y_col}) as av from ({applyout_df}) ) as gdt'
                    )
                sql = sql.format(applyout_df=applyout_df.select_statement,
                                 true_y_col=true_y_col,
                                 pred_y_col=pred_y_col)
                execute_logged(cur, sql)
                ret = cur.fetchone()
                return float(ret[0])
        except dbapi.Error as db_er:
            logger.error(
                "Failed to calculate the score, the error message: %s",
                db_er,
                exc_info=True)
            raise
