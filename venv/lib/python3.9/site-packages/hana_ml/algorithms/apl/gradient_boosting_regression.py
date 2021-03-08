# pylint:disable=too-many-lines
"""
This module provides the SAP HANA APL gradient boosting regression algorithm.

The following classes are available:

    * :class:`GradientBoostingRegressor`
"""
import logging
import pandas as pd
from hdbcli import dbapi
from hana_ml.dataframe import DataFrame
from hana_ml.dataframe import quotename
from hana_ml.ml_exceptions import FitIncompleteError
from hana_ml.algorithms.apl.gradient_boosting_base import GradientBoostingBase
from hana_ml.ml_base import execute_logged


logger = logging.getLogger(__name__) #pylint: disable=invalid-name


class GradientBoostingRegressor(GradientBoostingBase):
    """
    SAP HANA APL Gradient Boosting Regression algorithm.

    Parameters
    ----------
    conn_context :  ConnectionContext, optional
        The connection object to an SAP HANA database.
        This parameter is not needed anymore.
        It will be set automatically when a dataset is used in fit() or predict().
    early_stopping_patience: int, optional
        If the performance does not improve after **early_stopping_patience iterations**,
        the model training will stop before reaching **max_iterations**.
        Please refer to APL documentation for default value.
    eval_metric: str, optional
        The name of the metric used to evaluate the model performance on validation dataset along
        the boosting iterations.
        The possible values are 'MAE' and 'RMSE'.
        Please refer to APL documentation for default value.
    learning_rate: float, optional
        The weight parameter controlling the model regularization to avoid overfitting risk.
        A small value improves the model generalization to unseen dataset at the expense of the
        computational cost.
        Please refer to APL documentation for default value.
    max_depth: int, optional
        The maximum depth of the decision trees added as a base learner to the model at each
        boosting iteration.
        Please refer to APL documentation for default value.
    max_iterations: int, optional
        The maximum number of boosting iterations to fit the model.
        Please refer to APL documentation for default value.
    number_of_jobs: int, optional
        The number of threads allocated to the model training and apply parallelization.
        Please refer to APL documentation for default value.
    variable_storages: dict, optional
        Specifies the variable data types (string, integer, number).
        For example, {'VAR1': 'string', 'VAR2': 'number'}.
        See notes below for more details.
    variable_value_types: dict, optional
        Specifies the variable value types (continuous, nominal, ordinal).
        For example, {'VAR1': 'continuous', 'VAR2': 'nominal'}.
        See notes below for more details.
    variable_missing_strings: dict, optional
        Specifies the variable values that will be taken as missing.
        For example, {'VAR1': '???'} means anytime the variable value equals to '???',
        it will be taken as missing.
    extra_applyout_settings: dict, optional
        Determines the output of the predict() method.
        The possible values are:
        - By default (None value): the default output.
            - <KEY>: the key column if provided in the dataset
            - TRUE_LABEL: the actual value if provided
            - PREDICTED: the predicted value
        - {'APL/ApplyExtraMode': 'Individual Contributions'}: the feature importance for every
        sample
            - <KEY>: the key column if provided
            - TRUE_LABEL: the actual value if provided
            - PREDICTED: the predicted value
            - gb_contrib_<VAR1>: the contribution of the VAR1 variable to the score
            ...
            - gb_contrib_<VARN>: the contribution of the VARN variable to the score
            - gb_contrib_constant_bias: the constant bias contribution
    other_params: dict optional
        Corresponds to advanced settings.
        The dictionary contains {<parameter_name>: <parameter_value>}.
        The possible parameters are:
            - 'correlations_lower_bound'
            - 'correlations_max_kept'
            - 'cutting_strategy'
            - 'interactions'
            - 'interactions_max_kept'
            - 'variable_selection_max_nb_of_final_variables'
        See *Common APL Aliases for Model Training* in the `SAP HANA APL Reference Guide
        <https://help.sap.com/viewer/p/apl>`_.
    other_train_apl_aliases: dict, optional
        Users can provide APL aliases as advanced settings to the model.
        Users are free to input any possible value.
        Pleae see *Common APL Aliases for Model Training* in the `SAP HANA APL Reference Guide
        <https://help.sap.com/viewer/p/apl>`_.

    Attributes
    ----------
    label: str
      The target column name. This attribute is set when the fit() method is called.
      Users don't need to set it explicitly, except if the model is loaded from a table.
      In this case, this attribute must be set before calling predict().
    model_: hana_ml DataFrame
        The trained model content
    summary_: APLArtifactTable
        The reference to the "SUMMARY" table generated by the model training.
        This table contains the content of the model training summary.
    indicators_: APLArtifactTable
        The reference to the "INDICATORS" table generated by the model training.
        This table contains various metrics related to the model and model variables.
    fit_operation_logs_: APLArtifactTable
        The reference to the "OPERATION_LOG" table generated by the model training
    var_desc_: APLArtifactTable
        The reference to the "VARIABLE_DESCRIPTION" table that was built during the model training
    applyout_: hana_ml DataFrame
        The predictions generated the last time the model was applied
    predict_operation_logs_: APLArtifactTable
        The reference to the "OPERATION_LOG" table when a prediction was made

    Examples
    --------
    >>> from hana_ml.algorithms.apl.gradient_boosting_regression import GradientBoostingRegressor
    >>> from hana_ml.dataframe import ConnectionContext, DataFrame

    Connecting to SAP HANA

    >>> CONN = ConnectionContext('HDB_HOST', HDB_PORT, 'HDB_USER', 'HDB_PASS')
    >>> # -- Creates hana_ml DataFrame
    >>> hana_df = DataFrame(CONN,
    ...                     'SELECT "id", "class", "capital-gain", '
    ...                     '"native-country", "age" from APL_SAMPLES.CENSUS')

    Creating and fitting the model

    >>> model = GradientBoostingRegressor()
    >>> model.fit(hana_df, label='age', key='id')

    Getting variable interactions

    >>> model.set_params(other_train_apl_aliases={
    ...     'APL/Interactions': 'true',
    ...     'APL/InteractionsMaxKept': '3'
    ... })
    >>> model.fit(data=self._df_train, key=self._key, label=self._label)
    >>> # Checks interaction info in INDICATORS table
    >>> output = model.get_indicators().filter("KEY LIKE 'Interaction%'").collect()

    Debriefing

    >>> # Global performance metrics of the model
    >>> model.get_performance_metrics()
    {'L1': 7.31774, 'MeanAbsoluteError': 7.31774, 'L2': 9.42497, 'RootMeanSquareError': 9.42497, ...

    >>> model.get_feature_importances()
    {'Gain': OrderedDict([('class', 0.8728259801864624), ('capital-gain', 0.10493823140859604), ...

    Making predictions

    >>> # Default output
    >>> applyout_df = model.predict(hana_df)
    >>> applyout_df.collect().head(3) # returns the output as a pandas DataFrame
              id  TRUE_LABEL  PREDICTED
    39184  21772          27         25
    16537   7331          33         43
    7908   35226          65         42
    >>> # Individual Contributions
    >>> model.set_params(extra_applyout_settings={'APL/ApplyExtraMode': 'Individual Contributions'})
    >>> applyout_df = model.predict(hana_df)
    >>> applyout_df.collect().head(3) # returns the output as a pandas DataFrame
         id  TRUE_LABEL  gb_contrib_workclass  gb_contrib_fnlwgt  gb_contrib_education  ...
    0  6241          21             -1.330736          -0.385088              0.373539  ...
    1  6248          18             -0.784536          -2.191791             -1.788672  ...
    2  6253          26             -0.773891           0.358133             -0.185864  ...

    Saving the model in the schema named 'MODEL_STORAGE'

    >>> from hana_ml.model_storage import ModelStorage
    >>> model_storage = ModelStorage(connection_context=CONN, schema='MODEL_STORAGE')
    >>> model.name = 'My model name'
    >>> model_storage.save_model(model=model, if_exists='replace')

    Reloading the model for new predictions
    >>> model2 = model_storage.load_model(name='My model name')
    >>> out2 = model2.predict(data=hana_df)

    Please see model_storage class for further features of model storage

    Notes
    -----
        It is highly recommended to specify a key column in the training dataset.
        If not, once the model is trained, it won't be possible anymore to have a key defined in
        any input dataset. The key is particularly useful to join the predictions output to
        the input dataset.

        By default, if not provided, SAP HANA APL guesses the variable description by reading
        the first 100 rows. But, the results may be incorrect.
        The user can overwrite the guessed description by explicitly setting the variable_storages,
        variable_value_types and variable_missing_strings parameters. For example:
        ::
            model.set_params(
                    variable_storages = {
                        'ID': 'integer',
                        'sepal length (cm)': 'number'
                        })
            model.set_params(
                    variable_value_types = {
                        'sepal length (cm)': 'continuous'
                        })
            model.set_params(
                    variable_missing_strings = {
                        'sepal length (cm)': '-1'
                        })
    """
    APL_ALIAS_KEYS = {
        'model_type': 'APL/ModelType',
        '_algorithm_name': 'APL/AlgorithmName',
        # Specific to AGB
        'early_stopping_patience': 'APL/EarlyStoppingPatience',
        'eval_metric': 'APL/EvalMetric',
        'learning_rate': 'APL/LearningRate',
        'max_depth': 'APL/MaxDepth',
        'max_iterations': 'APL/MaxIterations',
        'number_of_jobs': 'APL/NumberOfJobs',
        # Inheritates from RobustRegression (FPA78-3936)
        'correlations_lower_bound': 'APL/CorrelationsLowerBound',
        'correlations_max_kept': 'APL/CorrelationsMaxKept',
        'cutting_strategy': 'APL/CuttingStrategy',
        'interactions': 'APL/Interactions',
        'interactions_max_kept': 'APL/InteractionsMaxKept',
        'variable_selection_max_nb_of_final_variables':
            'APL/VariableSelectionMaxNbOfFinalVariables',
    }
    def __init__(self,
                 conn_context=None,
                 early_stopping_patience=None,
                 eval_metric=None,  #'RMSE',
                 learning_rate=None,
                 max_depth=None,
                 max_iterations=None,
                 number_of_jobs=None,
                 variable_storages=None,
                 variable_value_types=None,
                 variable_missing_strings=None,
                 extra_applyout_settings=None,
                 ** other_params): #pylint: disable=too-many-arguments
        super(GradientBoostingRegressor, self).__init__(
            conn_context=conn_context,
            early_stopping_patience=early_stopping_patience,
            eval_metric=eval_metric,
            learning_rate=learning_rate,
            max_depth=max_depth,
            max_iterations=max_iterations,
            number_of_jobs=number_of_jobs,
            variable_storages=variable_storages,
            variable_value_types=variable_value_types,
            variable_missing_strings=variable_missing_strings,
            extra_applyout_settings=extra_applyout_settings,
            ** other_params)
        self._model_type = 'regression'
        self._force_target_var_type = 'continuous'

    def set_params(self, **parameters):
        """
        Sets attributes of the current model.

        Parameters
        ----------
        parameters: dict
            The attribute names and values
        """
        if 'eval_metric' in parameters:
            val = parameters.pop('eval_metric')
            valid_vals = ['RMSE', 'MAE']
            if val not in valid_vals:
                raise ValueError("Invalid eval_metric. The value must be among " + str(valid_vals))
            self.eval_metric = self._arg('eval_metric', val, str)
        return super(GradientBoostingRegressor, self).set_params(**parameters)

    def get_performance_metrics(self):
        """
        Returns the performance metrics of the last trained model.

        Returns
        -------
        A dictionary with metric name as key and metric value as value.
        """
        if not hasattr(self, 'indicators_'):
            raise FitIncompleteError(
                "The indicators table was not found. Please fit the model.")
        df_ind = self.indicators_
        cond = "VARIABLE like 'gb_%' and DETAIL IS NULL"
        df_ind = df_ind.filter(cond)   # hana DataFrame
        df_ind = df_ind.collect()  # to pd.DataFrame
        # Create the OrderedDict to be returned
        # converts df_ind.VALUE str to float if possible
        ret = {}
        for _, row in df_ind.iterrows():
            key = row.loc['KEY']
            old_v = row.loc['VALUE']
            try:
                new_v = float(old_v)
            except ValueError:
                new_v = old_v
            ret[key] = new_v
        ret.update({'perf_per_iteration': self.get_evalmetrics()})
        ret.update({'BestIteration': self.get_best_iteration()})
        return ret

    def predict(self, data):
        """
        Generates predictions with the fitted model.
        It is possible to add special outputs, such as variable individual contributions,
        through the 'extra_applyout_settings' parameter in the model.
        This parameter is described with examples in the class section.

        Parameters
        ----------
        data: hana_ml DataFrame
            The input dataset used for prediction

        Returns
        -------
        Prediction output: hana_ml DataFrame

        """
        if not self.model_:
            raise FitIncompleteError("Model not initialized. Perform a fit first.")
        # APPLY_CONFIG
        apply_config_data_df = None
        if self.extra_applyout_settings:
            extra_mode = self.extra_applyout_settings.get('APL/ApplyExtraMode', None)
            if extra_mode:
                if extra_mode == 'Individual Contributions':
                    apply_config_data_df = self._get_indiv_contrib_applyconf()
                # Free advanced settings. Can be used for reason code
                elif extra_mode == 'Advanced Apply Settings':
                    cfg_vals = [('APL/ApplyExtraMode', 'Advanced Apply Settings', None),]
                    for alias, param_val in self.extra_applyout_settings.items():
                        if alias != 'APL/ApplyExtraMode':
                            cfg_vals.append((alias, param_val, None))
                    apply_config_data_df = pd.DataFrame(cfg_vals)
                else:
                    # User provides an ExtraMode explicitly
                    apply_config_data_df = pd.DataFrame([
                        ('APL/ApplyExtraMode',
                         extra_mode, None)
                    ])
        if apply_config_data_df is None:
            # Default
            apply_config_data_df = pd.DataFrame([])
        applyout_df = self._predict(data=data,
                                    apply_config_data_df=apply_config_data_df)
        return self._rewrite_applyout_df(data=data, applyout_df=applyout_df)

    def score(self, data):
        """
        Computes the R^2 (Coefficient of determination) indicator on the predictions of the
        provided dataset.

        Parameters
        ----------
        data: hana_ml DataFrame
            The dataset used for prediction.
            It must contain the actual target values so that the score could be computed.

        Returns
        -------
            R2 indicator: float
        """

        applyout_df = self.predict(data)

        # Find the name of the label column in applyout table
        true_y_col = 'TRUE_LABEL'
        pred_y_col = 'PREDICTED'

        # Check if the label column is given in input dataset (true label)
        # If it is not present, score can't be calculated
        if true_y_col not in applyout_df.columns:
            raise ValueError("Cannot find true label column in the output of predict()")
        if pred_y_col not in applyout_df.columns:
            raise ValueError('Cannot find PREDICTED column in the output of predict().'
                             ' Please check the extra_applyout_settings')
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

    def _rewrite_applyout_df(self, data, applyout_df):
        """
        Rewrites the applyout dataframe so it outputs standardized column names.
        """

        # Determines the mapping old columns to new columns
        # Stores the mapping into different list of tuples [(old_column, new_columns)]
        cols_map = []      # starting columns: ID, TRUE_LABEL
        for i, old_col in enumerate(applyout_df.columns):
            if i == 0:
                # key column
                new_col = old_col
                cols_map.append((old_col, new_col))
            elif old_col == self.label:
                new_col = 'TRUE_LABEL'
                cols_map.append((old_col, new_col))
            elif old_col == 'gb_score_{LABEL}'.format(LABEL=self.label):
                new_col = 'PREDICTED'
                cols_map.append((old_col, new_col))
            else:
                cols_map.append((old_col, old_col))

        # Writes the select SQL by renaming the columns
        sql = ''
        # Starting columns
        for old_col, new_col in cols_map:
            # If the target var is not given in applyin, do not output it
            if old_col == self.label and old_col not in data.columns:
                continue
            if sql:
                sql = sql + ', '
            sql = (sql + '{old_col} {new_col}'.format(
                old_col=quotename(old_col),
                new_col=quotename(new_col)))
        sql = 'SELECT ' + sql + ' FROM ' + self.applyout_table_.name
        applyout_df_new = DataFrame(connection_context=self.conn_context,
                                    select_statement=sql)
        logger.info('DataFrame for predict ouput: %s', sql)
        return applyout_df_new

    def _get_indiv_contrib_applyconf(self): #pylint: disable=no-self-use
        """
        Gets the apply configuration for 'Individual Contributions' output.
        Returns
        -------
        A pandas dataframe for operation_config table
        """
        return pd.DataFrame([
            ('APL/ApplyExtraMode', 'Advanced Apply Settings', None),
            ('APL/ApplyPredictedValue', 'true', None),
            ('APL/ApplyContribution', 'all', None),
        ])
