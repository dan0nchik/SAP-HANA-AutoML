# pylint:disable=too-many-lines
"""
This module provides the SAP HANA APL gradient boosting classification algorithm.

The following classes are available:

    * :class:`GradientBoostingClassifier`
    * :class:`GradientBoostingBinaryClassifier`
"""

import logging
import pandas as pd
from hdbcli import dbapi
from hana_ml.dataframe import DataFrame
from hana_ml.ml_base import execute_logged
from hana_ml.dataframe import quotename
from hana_ml.ml_exceptions import FitIncompleteError
from hana_ml.algorithms.apl.gradient_boosting_base import GradientBoostingBase

logger = logging.getLogger(__name__) #pylint: disable=invalid-name


class _GradientBoostingClassifierBase(GradientBoostingBase):
    """
    Abstract class for SAP HANA APL Gradient Boosting Classifier algorithm
    """

    def __init__(self,
                 conn_context=None,
                 early_stopping_patience=10,
                 eval_metric='MultiClassLogLoss',
                 learning_rate=0.05,
                 max_depth=4,
                 max_iterations=1000,
                 number_of_jobs=None,
                 variable_storages=None,
                 variable_value_types=None,
                 variable_missing_strings=None,
                 extra_applyout_settings=None,
                 ** other_params): #pylint: disable=too-many-arguments
        super(_GradientBoostingClassifierBase, self).__init__(
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

        # For classification, the target variable must be nominal
        self._force_target_var_type = 'nominal'

    def set_params(self, **parameters):
        """
        Sets attributes of the current model.

        Parameters
        ----------
        parameters: dict
            The names and values of the attributes to change
        """
        if 'extra_applyout_settings' in parameters:
            param = self._arg(
                'extra_applyout_settings',
                parameters.pop('extra_applyout_settings'),
                dict)
            self.extra_applyout_settings = param
        return super(_GradientBoostingClassifierBase, self).set_params(**parameters)

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
        cond = "VARIABLE like 'gb_%' and (DETAIL is null or to_varchar(DETAIL)='?')"
        df_ind = df_ind.filter(cond)   # hana DataFrame
        df_ind = df_ind.collect()  # to pd.DataFrame
        # Create the dictionary to be returned
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
        Makes predictions with the fitted model.
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
        The default output is (if the model 'extra_applyout_settings' parameter is unset):
            - ID: the key column
            - TRUE_LABEL: the true label if it is given in the input dataset
            - PREDICTED: the predicted label
            - PROBABILITY: the probability of the predicted label

        In multinomial classification, users can request the probabilities of all classes by
        setting the parameter **'extra_applyout_settings'** to
        **{'APL/ApplyExtraMode': 'AllProbabilities'}**.
        The output will be:
            - ID: the key column
            - TRUE_LABEL: the true label if it is given in the input dataset
            - PREDICTED: the predicted label
            - PROBA_<class_1>: the probability of the class <class_1>
            ...
            - PROBA_<class_n>: the probability of the class <class_n>

        To get the individual contributions of each variable for each individual sample,
        the 'extra_applyout_settings' parameter must be set to
        **{'APL/ApplyExtraMode': 'Individual Contributions'}**.
        The output will contain the following columns:
            - ID: key column,
            - TRUE_LABEL: the actual label
            - PREDICTED: the predicted label
            - gb_contrib_<VAR1>: the contribution of the variable <VAR1> to the score
            ...
            - gb_contrib_<VARN>: the contribution of the variable <VARN> to the score
            - gb_contrib_constant_bias: the constant bias contribution to the score

        Users can also set APL/ApplyExtraMode with other values, for instance:
        **'extra_applyout_settings'** = **{'APL/ApplyExtraMode': 'BestProbabilityAndDecision'}**.
        New SAP Hana APL settings may be provided over time, so please check the SAP HANA APL
        documentation to know which settings are available:
        See Function Reference > Predictive Model Services > *APPLY_MODEL* > OPERATION_CONFIG
        Parameters in the `SAP HANA APL Reference Guide <https://help.sap.com/viewer/p/apl>`_.

        """
        if not self.model_:
            raise FitIncompleteError("Model not initialized. Perform a fit first.")
        if not self.label:
            raise ValueError('Label is unknown. Please specify the label parameter of the model.')
        # APPLY_CONFIG
        apply_config_data_df = None
        param_pattern = ('Protocols/Default/Transforms/GradientBoosting/Parameters/'
                         'ApplySettings/Supervised/{LABEL}/{OPTION}')
        if self.extra_applyout_settings:
            if self.extra_applyout_settings.get('APL/ApplyExtraMode', False):
                extra_mode = self.extra_applyout_settings.get('APL/ApplyExtraMode')
                if extra_mode == 'AllProbabilities':
                    # decision and probabilities of all classes
                    # There is no true APL Alias yet for this feature
                    param_decision = param_pattern.format(LABEL=self.label, OPTION='Decision')
                    param_proba = param_pattern.format(LABEL=self.label, OPTION='Probability')
                    param_score = param_pattern.format(LABEL=self.label, OPTION='Score')
                    apply_config_data_df = pd.DataFrame([
                        ('APL/ApplyExtraMode', 'Advanced Apply Settings', None),
                        (param_decision, 'true', None),
                        (param_proba, 'true', None),
                        (param_score, 'false', None)
                    ])
                elif extra_mode == 'Individual Contributions':
                    apply_config_data_df = self._get_indiv_contrib_applyconf()
                # Free advanced settings. Can be used for reason code
                elif extra_mode == 'Advanced Apply Settings':
                    cfg_vals = [('APL/ApplyExtraMode', 'Advanced Apply Settings', None)]
                    for alias, param_val in self.extra_applyout_settings.items():
                        if alias != 'APL/ApplyExtraMode':
                            cfg_vals.append((alias, param_val, None))
                    apply_config_data_df = pd.DataFrame(cfg_vals)
                if apply_config_data_df is None:
                    # any other APL/ApplyExtraMode
                    apply_config_data_df = pd.DataFrame([
                        ('APL/ApplyExtraMode', extra_mode, None)])
        if apply_config_data_df is None:
            # Default config: decision + best_proba
            # ExtraMode = 'BestProbabilityAndDecision'
            apply_config_data_df = pd.DataFrame([
                ('APL/ApplyExtraMode', 'BestProbabilityAndDecision', None),
                ])
        applyout_df = self._predict(data=data,
                                    apply_config_data_df=apply_config_data_df)
        return self._rewrite_applyout_df(data=data, applyout_df=applyout_df)

    def _get_indiv_contrib_applyconf(self):
        """
        Gets the apply configuration for 'Individual Contributions' output.
        Returns
        -------
        A pandas dataframe for operation_config table
        """
        raise NotImplementedError()  # to be implemented by subclass

    def score(self, data):
        """
        Returns the mean accuracy on the provided test dataset.

        Parameters
        ----------
        data: hana_ml DataFrame
            The test dataset used to compute the score.
            The labels must be provided in the dataset.

        Returns
        -------
            mean average accuracy: float
        """

        applyout_df = self.predict(data)

        # Find the name of the label column in applyout table
        true_label_col = 'TRUE_LABEL'
        pred_label_col = 'PREDICTED'

        # Check if the label column is given in input dataset (true label)
        # If it is not present, score can't be calculated
        if true_label_col not in applyout_df.columns:
            raise ValueError("Cannot find true label column in dataset")
        if pred_label_col not in applyout_df.columns:
            raise ValueError('Cannot find the PREDICTED column in the output of predict().'
                             ' Please check the "extra_applyout_settings" parameter.')
        try:
            with self.conn_context.connection.cursor() as cur:
                # Count TP + TN
                sql = ('SELECT COUNT(*) FROM ({APPLYOUT_DF}) '
                       + 'WHERE "{true_label_col}"="{pred_label_col}"')
                sql = sql.format(APPLYOUT_DF=applyout_df.select_statement,
                                 true_label_col=true_label_col,
                                 pred_label_col=pred_label_col)
                execute_logged(cur, sql)
                ret = cur.fetchone()
                return ret[0]/float(data.count())  # (TP + TN)/ Total
        except dbapi.Error as db_er:
            logger.error(
                "Failed to calculate the score, the error message: %s",
                db_er,
                exc_info=True)
            raise

    def _rewrite_applyout_df(self, data, applyout_df): #pylint:disable=too-many-branches
        """
        Rewrites the applyout dataframe so it outputs standardized column names.
        """

        # Determines the mapping old columns to new columns
        # Stores the mapping into different list of tuples [(old_column, new_columns)]
        cols_map = []
        # indices of the starting columns. That will be used to well order the output columns
        i_id = None
        i_true_label = None
        i_predicted = None
        for i, old_col in enumerate(applyout_df.columns):
            if i == 0:
                # key column
                new_col = old_col
                cols_map.append((old_col, new_col))
                i_id = i
            elif old_col == self.label:
                new_col = 'TRUE_LABEL'
                cols_map.append((old_col, new_col))
                i_true_label = i
            elif old_col == 'gb_decision_{LABEL}'.format(LABEL=self.label):
                new_col = 'PREDICTED'
                cols_map.append((old_col, new_col))
                i_predicted = i
            else:
                # Proba of the predicted
                found = self._get_new_column_name(
                    old_col_re=r'gb_best_proba_{LABEL}'.format(LABEL=self.label),
                    old_col=old_col,
                    new_col_re=r'PROBABILITY')
                if found:
                    new_col = found
                    cols_map.append((old_col, new_col))
                else:
                    # If vector of proba
                    found = self._get_new_column_name(
                        old_col_re=r'gb_proba_{LABEL}_(.+)'.format(LABEL=self.label),
                        old_col=old_col,
                        new_col_re=r'PROBA_\1')
                    if found:
                        new_col = found
                        cols_map.append((old_col, new_col))
                    else:
                        new_col = old_col
                        cols_map.append((old_col, new_col))
        # Indices of the columns to be displayed as first columns, in a certain order
        first_cols_ix = [i for i in [i_id, i_true_label, i_predicted] if i is not None]
        # Writes the select SQL by renaming the columns
        sql = ''
        # Starting columns
        for i in first_cols_ix:
            old_col, new_col = cols_map[i]
            # If the target var is not given in applyin, do not output it
            if old_col == self.label and old_col not in data.columns:
                continue
            if sql:
                sql = sql + ', '
            sql = (sql + '{old_col} {new_col}'.format(
                old_col=quotename(old_col),
                new_col=quotename(new_col)))
        # Remaining columns
        for i, (old_col, new_col) in enumerate(cols_map):
            if i not in first_cols_ix:
                sql = sql + ', '
                sql = (sql + '{old_col} {new_col}'.format(
                    old_col=quotename(old_col),
                    new_col=quotename(new_col)))
        sql = 'SELECT ' +  sql + ' FROM ' + self.applyout_table_.name
        applyout_df_new = DataFrame(connection_context=self.conn_context,
                                    select_statement=sql)
        logger.info('DataFrame for predict ouput: %s', sql)
        return applyout_df_new


class GradientBoostingClassifier(_GradientBoostingClassifierBase):
    """
    SAP HANA APL Gradient Boosting Multiclass Classifier algorithm.

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
        The possible values are 'MultiClassClassificationError' and 'MultiClassLogLoss'.
        Please refer to APL documentation for default value..
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
        The default value is 1000.
    number_of_jobs: int, optional
        The number of threads allocated to the model training and apply parallelization.
        Please refer to APL documentation for default value.
    variable_storages: dict, optional
        Specifies the variable data types (string, integer, number).
        For example, {'VAR1': 'string', 'VAR2': 'number'}.
        See notes below for more details.
    variable_value_types: dict, optional
        Specifies the variable value type (continuous, nominal, ordinal).
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
            - <KEY>: the key column if it provided in the dataset
            - TRUE_LABEL: the class label if provided in the dataset
            - PREDICTED: the predicted label
            - PROBABILITY: the probability of the prediction(confidence)
        - {'APL/ApplyExtraMode': 'AllProbabilities'}: the probabilities for each class.
            - <KEY>: the key column if provided in the dataset
            - TRUE_LABEL: the class label if given in the dataset
            - PREDICTED: the predicted label
            - PROBA_<label_value1>: the probability for the class <label_value1>
            ...
            - PROBA_<label_valueN>: the probability for the class <label_valueN>
        - {'APL/ApplyExtraMode': 'Individual Contributions'}: the feature importance for every
        sample
            - <KEY>: the key column if provided in the dataset
            - TRUE_LABEL: the class label when if provided in the dataset
            - PREDICTED: the predicted label
            - gb_contrib_<VAR1>: the contribution of the variable VAR1 to the score
            ...
            - gb_contrib_<VARN>: the contribution of the variable VARN to the score
            - gb_contrib_constant_bias: the constant bias contribution to the score
    other_params: dict optional
        Corresponds to advanced settings.
        The dictionary contains {<parameter_name>: <parameter_value>}.
        The possible parameters are:
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
    >>> from hana_ml.algorithms.apl.gradient_boosting_classification \
        import GradientBoostingClassifier
    >>> from hana_ml.dataframe import ConnectionContext, DataFrame

    Connecting to SAP HANA

    >>> CONN = ConnectionContext('HDB_HOST', HDB_PORT, 'HDB_USER', 'HDB_PASS')
    >>> # -- Creates hana_ml DataFrame
    >>> hana_df = DataFrame(CONN,
                            'SELECT "id", "class", "capital-gain", '
                            '"native-country" from APL_SAMPLES.CENSUS')

    Creating and fitting the model

    >>> model = GradientBoostingClassifier()
    >>> model.fit(hana_df, label='native-country', key='id')

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
    {'BalancedErrorRate': 0.9761904761904762, 'BalancedClassificationRate': 0.023809523809523808,
    ...

    >>> # Performance metrics of the model for each class
    >>> model.get_metrics_per_class()
    {'Precision': {'Cambodia': 0.0, 'Canada': 0.0, 'China': 0.0, 'Columbia': 0.0...

    >>> model.get_feature_importances()
    {'Gain': OrderedDict([('class', 0.7713800668716431), ('capital-gain', 0.22861991822719574)])}

    Making predictions

    >>> # Default output
    >>> applyout_df = model.predict(hana_df)
    >>> applyout_df.collect().head(3) # returns the output as a pandas DataFrame
        id     TRUE_LABEL      PREDICTED  PROBABILITY
    0   30  United-States  United-States     0.89051
    1   63  United-States  United-States     0.89051
    2   66  United-States  United-States     0.89051
    >>> # All probabilities
    >>> model.set_params(extra_applyout_settings={'APL/ApplyExtraMode': 'AllProbabilities'})
    >>> applyout_df = model.predict(hana_df)
    >>> applyout_df.collect().head(3) # returns the output as a pandas DataFrame
              id     TRUE_LABEL      PREDICTED      PROBA_?     PROBA_Cambodia  ...
    35194  19272  United-States  United-States    0.016803            0.000595  ...
    20186  39624  United-States  United-States    0.017564            0.001063  ...
    43892  38759  United-States  United-States    0.019812            0.000353  ...
    >>> # Individual Contributions
    >>> model.set_params(extra_applyout_settings={'APL/ApplyExtraMode': 'Individual Contributions'})
    >>> applyout_df = model.predict(hana_df)
    >>> applyout_df.collect().head(3) # returns the output as a pandas DataFrame
       id     TRUE_LABEL      PREDICTED  gb_contrib_class  gb_contrib_capital-gain  ...
    0  30  United-States  United-States         -0.025366                -0.014416  ...
    1  63  United-States  United-States         -0.025366                -0.014416  ...
    2  66  United-States  United-States         -0.025366                -0.014416  ...

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
        'cutting_strategy': 'APL/CuttingStrategy',
        # Common to AGB
        'early_stopping_patience': 'APL/EarlyStoppingPatience',
        'eval_metric': 'APL/EvalMetric',
        'learning_rate': 'APL/LearningRate',
        'max_depth': 'APL/MaxDepth',
        'max_iterations': 'APL/MaxIterations',
        'number_of_jobs': 'APL/NumberOfJobs',
        'interactions': 'APL/Interactions',
        'interactions_max_kept': 'APL/InteractionsMaxKept',
        # Heritates from K2R
        'variable_selection_max_nb_of_final_variables':
            'APL/VariableSelectionMaxNbOfFinalVariables',
    }
    # pylint: disable=too-many-arguments
    def __init__(self,
                 conn_context=None,
                 early_stopping_patience=None,
                 eval_metric=None,  #'MultiClassLogLoss',
                 learning_rate=None,
                 max_depth=None,
                 max_iterations=None,
                 number_of_jobs=None,
                 variable_storages=None,
                 variable_value_types=None,
                 variable_missing_strings=None,
                 extra_applyout_settings=None,
                 ** other_params):
        super(GradientBoostingClassifier, self).__init__(
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
        self._model_type = 'multiclass'

    def set_params(self, **parameters):
        """
        Sets attributes of the current model.

        Parameters
        ----------
        parameters: dict
            The names and values of the attributes to change
        """
        if 'eval_metric' in parameters:
            val = parameters.pop('eval_metric')
            valid_vals = ['MultiClassLogLoss', 'MultiClassClassificationError']
            if val not in valid_vals:
                raise ValueError("Invalid eval_metric. The value must be among " + str(valid_vals))
            self.eval_metric = self._arg('eval_metric', val, str)
        return super(GradientBoostingClassifier, self).set_params(**parameters)

    def get_metrics_per_class(self):
        """
        Returns the performance for each class.

        Returns
        -------
        A dictionary.

        Example
        -------

        >>> data = DataFrame(conn, 'SELECT * from IRIS_MULTICLASSES')
        >>> model = GradientBoostingClassifier(conn)
        >>> model.fit(data=data, key='ID', label='LABEL')
        >>> model.get_metrics_per_class()
        {
        'Precision': {
            'setosa': 1.0,
            'versicolor': 1.0,
            'virginica': 0.9743589743589743
        },
        'Recall': {
            'setosa': 1.0,
            'versicolor': 0.9714285714285714,
            'virginica': 1.0
        },
        'F1Score': {
            'setosa': 1.0,
            'versicolor': 0.9855072463768115,
            'virginica': 0.9870129870129869
        }
        """
        if not hasattr(self, 'indicators_'):
            raise FitIncompleteError(
                "The indicators table was not found. Please fit the model.")
        df_ind = self.indicators_
        cond = "VARIABLE like 'gb_%' and DETAIL is not null and to_varchar(DETAIL)!='?'"
        df_ind = df_ind.filter(cond)   # hana DataFrame
        df_ind = df_ind.collect()  # to pd.DataFrame
        # Create the dictionary to be returned
        # converts df_ind.VALUE str to float if possible
        ret2 = {}  # 2 levels dictionary {metrics/classes: value}
        for _, row in df_ind.iterrows():
            metric_name = row.loc['KEY']
            old_v = row.loc['VALUE']
            class_v = row.loc['DETAIL']
            try:
                new_v = float(old_v)
            except ValueError:
                new_v = old_v
            class_v = str(class_v)
            per_class = ret2.get(metric_name, {})
            per_class[class_v] = new_v
            ret2[metric_name] = per_class
        return ret2

    def _get_indiv_contrib_applyconf(self):
        """
        Gets the apply configuration for 'Individual Contributions' output.
        Returns
        -------
        A pandas dataframe for operation_config table
        """

        return pd.DataFrame([('APL/ApplyExtraMode', 'Individual Contributions', None),
                            ])

class GradientBoostingBinaryClassifier(_GradientBoostingClassifierBase):
    """
    SAP HANA APL Gradient Boosting Binary Classifier algorithm.
    It is very similar to GradientBoostingClassifier, the multiclass classifier.
    Its particularity lies in the provided metrics which are specific to binary classification.

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
        The possible values are 'LogLoss','AUC' and 'ClassificationError'.
        Please refer to APL documentation for default value.
    learning_rate: float, optional
        The weight parameter controlling the model regularization to avoid overfitting risk.
        A small value improves the model generalization to unseen dataset at the expense of the
        computational cost.
        Please refer to APL documentation for default value.
    max_depth: int, optional
        The maximum depth of the decision trees added as a base learner to the model at each
        boosting iteration.
        The default value is 4.
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
        Specifies the variable value type (continuous, nominal, ordinal).
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
            - TRUE_LABEL: the class label if provided in the dataset
            - PREDICTED: the predicted label
            - PROBABILITY: the probability of the prediction(confidence)
        - {'APL/ApplyExtraMode': 'Individual Contributions'}: the individual contributions of each
        variable to the score. The output is:
            - <KEY>: the key column if provided in the dataset
            - TRUE_LABEL: the class label if provided in the dataset
            - gb_contrib_<VAR1>: the contribution of the variable VAR1 to the score
            ...
            - gb_contrib_<VARN>: the contribution of the variable VARN to the score
            - gb_contrib_constant_bias: the constant bias contribution to the score
    other_params: dict optional
        Corresponds to advanced settings.
        The dictionary contains {<parameter_name>: <parameter_value>}.
        The possible parameters are:
            - 'correlations_lower_bound'
            - 'correlations_max_kept'
            - 'cutting_strategy'
            - 'target_key'
            - 'interactions'
            - 'interactions_max_kept'
            - 'variable_selection_max_nb_of_final_variables'
        See *Common APL Aliases for Model Training* in the `SAP HANA APL Reference Guide
        <https://help.sap.com/viewer/p/apl>`_.
    other_train_apl_aliases: dict, optional
        Contains the APL alias for model training.
        The list of possible aliases depends on the APL version.
        Please refer to HANA APL documentation about aliases.

    Attributes
    ----------
    label: str
      The target column name. This attribute is set when the fit() method is called.
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
    >>> from hana_ml.algorithms.apl.gradient_boosting_classification \
        import GradientBoostingBinaryClassifier
    >>> from hana_ml.dataframe import ConnectionContext, DataFrame

    Connecting to SAP HANA

    >>> CONN = ConnectionContext(HDB_HOST, HDB_PORT, HDB_USER, HDB_PASS)
    >>> # -- Creates hana_ml DataFrame
    >>> hana_df = DataFrame(CONN, 'SELECT * from APL_SAMPLES.CENSUS')

    Creating and fitting the model

    >>> model = GradientBoostingBinaryClassifier()
    >>> model.fit(hana_df, label='class', key='id')

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
    {'LogLoss': 0.2567069689038737, 'PredictivePower': 0.8529, 'PredictionConfidence': 0.9759,
    ...}

    >>> model.get_feature_importances()
    {'Gain': OrderedDict([('relationship', 0.3866586685180664),
                          ('education-num', 0.1502334326505661)...

    Making predictions

    >>> # Default output
    >>> applyout_df = model.predict(hana_df)
    >>> applyout_df.collect().sample(3) # returns the output as a pandas DataFrame
              id  TRUE_LABEL  PREDICTED  PROBABILITY
    44903  41211           0          0    0.871326
    47878  36020           1          1    0.993455
    17549   6601           0          1    0.673872

    >>> # Individual Contributions
    >>> model.set_params(extra_applyout_settings={'APL/ApplyExtraMode': 'Individual Contributions'})
    >>> applyout_df = model.predict(hana_df)
    >>> applyout_df.collect().sample(3) # returns the output as a pandas DataFrame
          id  TRUE_LABEL  gb_contrib_age  gb_contrib_workclass  gb_contrib_fnlwgt  ...
    0  18448           0       -1.098452             -0.001238           0.060850  ...
    1  18457           0       -0.731512             -0.000448           0.020060  ...
    2  18540           0       -0.024523              0.027065           0.158083  ...

    Saving the model in the schema named 'MODEL_STORAGE'
    Please see model_storage class for further features of model storage.

    >>> from hana_ml.model_storage import ModelStorage
    >>> model_storage = ModelStorage(connection_context=CONN, schema='MODEL_STORAGE')
    >>> model.name = 'My model name'
    >>> model_storage.save_model(model=model, if_exists='replace')

    Reloading the model for new predictions
    >>> model2 = model_storage.load_model(name='My model name')
    >>> out2 = model2.predict(data=hana_df)

    """
    APL_ALIAS_KEYS = {
        'model_type': 'APL/ModelType',
        'cutting_strategy': 'APL/CuttingStrategy',
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
        'target_key': 'APL/TargetKey',
        'interactions': 'APL/Interactions',
        'interactions_max_kept': 'APL/InteractionsMaxKept',
        'variable_selection_max_nb_of_final_variables':
            'APL/VariableSelectionMaxNbOfFinalVariables',
    }

    # pylint: disable=too-many-arguments
    def __init__(self,
                 conn_context=None,
                 early_stopping_patience=None,
                 eval_metric=None, ##'LogLoss',
                 learning_rate=None,
                 max_depth=None,
                 max_iterations=None,
                 number_of_jobs=None,
                 variable_storages=None,
                 variable_value_types=None,
                 variable_missing_strings=None,
                 extra_applyout_settings=None,
                 ** other_params):
        super(GradientBoostingBinaryClassifier, self).__init__(
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
        self._model_type = 'binary classification'

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
            valid_vals = ['LogLoss', 'AUC', 'ClassificationError']
            if val not in valid_vals:
                raise ValueError("Invalid eval_metric. The value must be among " + str(valid_vals))
            self.eval_metric = self._arg('eval_metric', val, str)
        return super(GradientBoostingBinaryClassifier, self).set_params(**parameters)

    def get_performance_metrics(self):
        """
        Returns the performance metrics of the last trained model.

        Returns
        -------
        A dictionary with metric name as key and metric value as value.

        Example
        -------

        >>> data = DataFrame(conn, 'SELECT * from APL_SAMPLES.CENSUS')
        >>> model = GradientBoostingBinaryClassifier(conn)
        >>> model.fit(data=data, key='id', label='class')
        >>> model.get_performance_metrics()
        {'AUC': 0.9385, 'PredictivePower': 0.8529, 'PredictionConfidence': 0.9759,...}
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

    def _get_indiv_contrib_applyconf(self):
        """
        Gets the apply configuration for 'Individual Contributions' output.
        Settings are different depending on the subclass (multinomial classification, binary
        classification, or regression).
        Returns
        -------
        A pandas dataframe for operation_config table
        """
        return pd.DataFrame([
            ('APL/ApplyExtraMode', 'Advanced Apply Settings', None),
            ('APL/ApplyDecision', 'true', None),
            ('APL/ApplyContribution', 'all', None),
        ])
