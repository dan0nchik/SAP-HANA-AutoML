"""
This module contains Python wrapper for SAP HANA PAL unified-regression.

The following classes are available:
    * :class:`UnifiedRegression`
"""

#pylint: disable=too-many-lines
#pylint: disable=line-too-long
#pylint: disable=too-many-locals
#pylint: disable=too-many-arguments
#pylint: disable=ungrouped-imports
#pylint: disable=relative-beyond-top-level
import logging
import uuid
import pandas as pd
from hdbcli import dbapi
from hana_ml.visualizers.model_report import (
    ParameterReportBuilder,
    UnifiedRegressionReportBuilder
)
from hana_ml.ml_base import try_drop
from hana_ml.ml_exceptions import FitIncompleteError
from .sqlgen import trace_sql
from .pal_base import (
    PALBase,
    ParameterTable,
    require_pal_usable,
    pal_param_register,
    call_pal_auto,
    ListOfStrings,
    ListOfTuples
)



logger = logging.getLogger(__name__) #pylint: disable=invalid-name

class UnifiedRegression(PALBase):#pylint: disable=too-many-instance-attributes
    """
    The Python wrapper for SAP HANA PAL unified-regression function.

    Compared with the original regression interfaces,
    new features supported are listed below:
        - Regression algorithms easily switch
        - Dataset automatic partition
        - Model evaluation procedure provided
        - More metrics supported

    Parameters
    ----------

    func : str

        The name of a specified regression algorithm.

        The following algorithms(case-insensitive) are supported:

            - 'DecisionTree'
            - 'HybridGradientBoostingTree'
            - 'LinearRegression'
            - 'RandomDecisionTree'
            - 'MLP'
            - 'SVM'
            - 'GLM'
            - 'GeometricRegression'
            - 'PolynomialRegression'
            - 'ExponentialRegression'
            - 'LogarithmicRegression'

    **kwargs : keyword arguments

        Arbitrary keyword arguments and please referred to the responding algorithm for the parameters' key-value pair.

        **Note that some parameters are disabled in the regression algorithm!**

            - **'DecisionTree'** : :class:`~hana_ml.algorithms.pal.trees.DecisionTreeRegressor`

              - Disabled parameters: output_rules
              - Parameters removed from initialization but can be specified in fit(): categorical_variable

            - **'HybridGradientBoostingTree'** : :class:`~hana_ml.algorithms.pal.trees.HybridGradientBoostingRegressor`

              - Disabled parameters: calculate_importance
              - Parameters removed from initialization but can be specified in fit(): categorical_variable

            - **'LinearRegression'** : :class:`~hana_ml.algorithms.pal.linear_model.LinearRegression`

              - Disabled parameters: pmml_export
              - Parameters removed from initialization but can be specified in fit(): categorical_variable

            - **'RandomDecisionTree'** : :class:`~hana_ml.algorithms.pal.trees.RDTRegressor`

              - Disabled parameters: calculate_oob
              - Parameters removed from initialization but can be specified in fit(): categorical_variable

            - **'MLP'** : :class:`~hana_ml.algorithms.pal.neural_network.MLPRegressor`

              - Disabled parameters: functionality
              - Parameters removed from initialization but can be specified in fit(): categorical_variable

            - **'SVM'** : :class:`~hana_ml.algorithms.pal.svm.SVR`

              - Parameters removed from initialization but can be specified in fit(): categorical_variable

            - **'GLM'** : :class:`~hana_ml.algorithms.pal.regression.GLM`

              - Disabled parameters: output_fitted
              - Parameters removed from initialization but can be specified in fit(): categorical_varaible

            - **'GeometricRegression'** : :class:`~hana_ml.algorithms.pal.regression.BiVariateGeometricRegression`

              - Disabled parameters: pmml_export

            - **'PolynomialRegression'** : :class:`~hana_ml.algorithms.pal.regression.PolynomialRegression`

              - Disabled parameters: pmml_export

            - **'ExponentialRegression'** : :class:`~hana_ml.algorithms.pal.regression.ExponentialRegression`

              - Disabled parameters: pmml_export

            - **'LogarithmicRegression'** : :class:`~hana_ml.algorithms.pal.regression.BiVariateNaturalLogarithmicRegression`

              - Disabled parameters: pmml_export

    Attributes
    ----------

    model_ : DataFrame

        Model content.

    statistics_ : DataFrame

        Names and values of statistics.

    optim_param_ : DataFrame

        Provides optimal parameters selected.

        Available only when parameter selection is triggered.

    partition_ : DataFrame

        Partition result of training data.

        Available only when training data has an ID column and random partition is applied.

    Examples
    --------
    Training data for regression:

    >>> data_tbl.collect()
       ID    X1 X2  X3       Y
    0   0  0.00  A   1  -6.879
    1   1  0.50  A   1  -3.449
    2   2  0.54  B   1   6.635
    3   3  1.04  B   1  11.844
    4   4  1.50  A   1   2.786
    5   5  0.04  B   2   2.389
    6   6  2.00  A   2  -0.011
    7   7  2.04  B   2   8.839
    8   8  1.54  B   1   4.689
    9   9  1.00  A   2  -5.507

    Create a UnifiedRegression instance for linear regression problem:

    >>> mlr_params = dict(solver = 'qr',
                          adjusted_r2=False,
                          thread_ratio=0.5)

    >>> umlr = UnifiedRegression(func='LinearRegression', **mlr_params)

    Fit the UnifiedRegression instance with the aforementioned training data:

    >>> par_params = dict(partition_method='random',
                          training_percent=0.7,
                          partition_random_state=2,
                          output_partition_result=True)

    >>> umlr.fit(data = data_tbl,
                 key = 'ID',
                 label = 'Y',
                 **par_params)

    Check the resulting statistics on testing data:

    >>> umlr.statistics_.collect()
            STAT_NAME          STAT_VALUE
    0       TEST_EVAR   0.871459247598903
    1        TEST_MAE  2.0088082000000003
    2       TEST_MAPE  12.260003987804756
    3  TEST_MAX_ERROR   5.329849599999999
    4        TEST_MSE   9.551661310681718
    5         TEST_R2  0.7774293644548433
    6       TEST_RMSE    3.09057621013974
    7      TEST_WMAPE  0.7188006440839695

    Data for prediction:

    >>> data_pred.collect()
       ID       X1 X2  X3
    0   0    1.690  B   1
    1   1    0.054  B   2
    2   2  980.123  A   2
    3   3    1.000  A   1
    4   4    0.563  A   1

    Perform prediction:

    >>> pred_res = mlr.predict(data = data_pred, key = 'ID')
    >>> pred_res.collect()
       ID        SCORE UPPER_BOUND LOWER_BOUND REASON
    0   0     8.719607        None        None   None
    1   1     1.416343        None        None   None
    2   2  3318.371440        None        None   None
    3   3    -2.050390        None        None   None
    4   4    -3.533135        None        None   None


    Data for scoring:

    >>> data_score.collect()
       ID       X1 X2  X3    Y
    0   0    1.690  B   1  1.2
    1   1    0.054  B   2  2.1
    2   2  980.123  A   2  2.4
    3   3    1.000  A   1  1.8
    4   4    0.563  A   1  1.0

    Perform scoring:

    >>> score_res = umlr.score(data = data_score, key = "ID", label = 'Y')

    Check the statistics on scoring data:

    >>> score_res[1].collect()
       STAT_NAME          STAT_VALUE
    0       EVAR  -6284768.906191169
    1        MAE   666.5116459919999
    2       MAPE   278.9837795885635
    3  MAX_ERROR  3315.9714402299996
    4        MSE   2199151.795823181
    5         R2   -7854112.55651136
    6       RMSE  1482.9537402842952
    7      WMAPE   392.0656741129411
    """
    func_dict = {
        'decisiontree' : 'DT',
        'hybridgradientboostingtree' : 'HGBT',
        'linearregression' : 'MLR',
        'randomforest' : 'RDT',
        'randomdecisiontree' : 'RDT',
        'svm' : 'SVM',
        'mlp' : 'MLP',
        'glm' : 'GLM',
        'geometricregression': 'GEO',
        'polynomialregression' : 'POL',
        'exponentialregression' : 'EXP',
        'logarithmicregression' : 'LOG'}
    __cv_dict = {'resampling_method' : ('RESAMPLING_METHOD',
                                        {'cv' : 'cv', 'bootstrap' : 'bootstrap'}),
                 'random_state' : ('SEED', int),
                 'fold_num' : ('FOLD_NUM', int),
                 'repeat_times' : ('REPEAT_TIMES', int),
                 'search_strategy' : ('PARAM_SEARCH_STRATEGY', str,
                                      {'random' : 'random', 'grid' : 'grid'}),
                 'param_search_strategy' : ('PARAM_SEARCH_STRATEGY', str,
                                            {'random' : 'random', 'grid' : 'grid'}),
                 'random_search_times' : ('RANDOM_SEARCH_TIMES', int),
                 'timeout' : ('TIMEOUT', int),
                 'progress_indicator_id' : ('PROGRESS_INDICATOR_ID', str)}
    __param_grid_dict = {'param_values' : ('_VALUES', dict),
                         'param_range' : ('_RANGE', dict)}
    __activation_map = {'tanh' : 1,
                        'linear' : 2,
                        'sigmoid_asymmetric' : 3,
                        'sigmoid_symmetric' : 4,
                        'gaussian_asymmetric' : 5,
                        'gaussian_symmetric' : 6,
                        'elliot_asymmetric' : 7,
                        'elliot_symmetric' : 8,
                        'sin_asymmetric' : 9,
                        'sin_symmetric' : 10,
                        'cos_asymmetric' : 11,
                        'cos_symmetric' : 12,
                        'relu' : 13}
    __param_dict = {
        'adjusted_r2' : ('ADJUSTED_R2', bool),
        'decomposition' : ('DECOMPOSITION', int, {'lu' : 0, 'qr' : 1, 'svd' : 2, 'cholesky' : 5}),
        'thread_ratio' : ('THREAD_RATIO', float)}
    map_dict = {
        'GEO' : __param_dict,
        'LOG' : __param_dict,
        'EXP' : __param_dict,
        'POL' : dict({'degree' : ('DEGREE', int),
                      'degree_values' : ('DEGREE_VALUES', list),
                      'degree_range' : ('DEGREE_RANGE', list),
                      'evaluation_metric' : ('EVALUATION_METRIC', {'rmse' : 'RMSE'})},
                     **__param_dict,),
        'GLM' : {
            'family' : ('FAMILY', dict(gaussian='gaussian',
                                       normal='gaussian',
                                       poisson='poisson',
                                       binomial='binomial',
                                       gamma='gamma',
                                       inversegaussin='inversegaussian',
                                       negativebinomial='negativebinomial',
                                       ordinal='ordinal')),
            'link' : ('LINK', dict(identity='identity',
                                   log='log',
                                   logit='logit',
                                   probit='probit',
                                   comploglog='comploglog',
                                   reciprocal='inverse',
                                   inverse='inverse',
                                   inversesquare='inversesquare',
                                   sqrt='sqrt')),
            'solver' : ('SOLVER', dict(irls='irls', nr='nr', cd='cd')),
            'significance_level' : ('SIGNIFICANCE_LEVEL', float),
            'quasilikelihood' : ('QUASILIKELIHOOD', bool),
            'group_response' : ('GROUP_RESPONSE', bool),
            'handle_missing_fit' : ('HANDLE_MISSING_FIT', {'skip' : 1, 'abort' : 0, 'fill_zero' : 2}),
            'max_iter' : ('MAX_ITER', int),
            'tol' : ('TOL', float),
            'enet_alpha' : ('ALPHA', float),
            'enet_lambda' : ('LAMBDA', float),
            'num_lambda' : ('NUM_LAMBDA', int),
            'ordering' : ('ORDERING', ListOfStrings),
            'thread_ratio' : ('THREAD_RATIO', float),
            'evaluation_metric' : ('EVALUATION_METRIC',
                                   {'rmse' : 'RMSE', 'mae' : 'MAE', 'error_rate' :'ERROR_RATE'})
        },
        'MLR' : {
            'adjusted_r2' : ('ADJUSTED_R2', bool),
            'solver' : ('SOLVER', int, {'qr' : 1, 'svd' : 2, 'cyclical' : 4, 'cholesky' : 5, 'admm' : 6}),
            'alpha_to_enter' : ('ALPHA_TO_ENTER', float),
            'alpha_to_remove' : ('ALPHA_TO_REMOVE', float),
            'bp_test' : ('BP_TEST', bool),
            'dw_test' : ('DW_TEST', bool),
            'enet_alpha' : ('ALPHA', float),
            'enet_lambda' : ('LAMBDA', float),
            'handle_missing' : ('HANDLE_MISSING', bool),
            'mandatory_feature' : ('MANDATORY_FEATURE', ListOfStrings),
            'ks_test' : ('KS_TEST', bool),
            'max_iter' : ('MAX_ITER', int),
            'intercept' : ('INTERCEPT', bool),
            'pho' : ('PHO', float),
            'reset_test' : ('RESET_TEST', int),
            'stat_inf' : ('STAT_INF', bool),
            'tol' : ('TOL', float),
            'thread_ratio' : ('THREAD_RATIO', float),
            'var_select' : ('VAR_SELECT', {'no' : 0, 'forward' : 1, 'backward' : 2, 'stepwise' : 3}),
            'evaluation_metric' : ('EVALUATION_METRIC', {'rmse' : 'RMSE'})
            },
        'MLP' : {
            'activation' : ('ACTIVATION',
                            int,
                            __activation_map),
            'activation_options' : ('ACTIVATION_OPTIONS', ListOfStrings),
            'output_activation' : ('OUTPUT_ACTIVATION',
                                   int,
                                   __activation_map),
            'output_activation_options' : ('OUTPUT_ACTIVATION_OPTIONS', ListOfStrings),
            'hidden_layer_size' : ('HIDDEN_LAYER_SIZE', (list, tuple)),
            'hidden_layer_size_options' : ('HIDDEN_LAYER_SIZE_OPTIONS', ListOfTuples),
            'max_iter' : ('MAX_ITER', int),
            'training_style' : ('TRAINING_STYLE', int, {'batch' : 0, 'stochastic' : 1}),
            'learning_rate' : ('LEARNING_RATE', float),
            'momentum' : ('MOMENTUM', float),
            'batch_size' : ('BATCH_SIZE', int),
            'normalization' : ('NORMALIZATION', int, {'no' : 0, 'z-transform' : 1, 'scalar' : 2}),
            'weight_init' : ('WEIGHT_INIT',
                             int,
                             {'all-zeros' : 0,
                              'normal' : 1,
                              'uniform' : 2,
                              'variance-scale-normal' : 3,
                              'variance-scale-uniform' : 4}),
            'thread_ratio' : ('THREAD_RATIO', float),
            'evaluation_metric' : ('EVALUATION_METRIC', {'rmse' : 'RMSE'})},
        'DT' : {
            'allow_missing_dependent' : ('ALLOW_MISSING_LABEL', bool),
            'percentage' : ('PERCENTAGE', float),
            'min_records_of_parent' : ('MIN_RECORDS_PARENT', int),
            'min_records_of_leaf' : ('MIN_RECORDS_LEAF', int),
            'max_depth' : ('MAX_DEPTH', int),
            'split_threshold' : ('SPLIT_THRESHOLD', float),
            'discretization_type' : ('DISCRETIZATION_TYPE', int, {'mdlpc' :0, 'equal_freq' :1}),
            'max_branch' : ('MAX_BRANCH', int),
            'merge_threshold' : ('MERGE_THRESHOLD', float),
            'use_surrogate' : ('USE_SURROGATE', bool),
            'model_format' : ('MODEL_FORMAT', int, {'json' :1, 'pmml' :2}),
            'thread_ratio' : ('THREAD_RATIO', float),
            'evaluation_metric' : ('EVALUATION_METRIC',
                                   {'rmse' : 'RMSE', 'mae' : 'MAE'})},
        'RDT' : {
            'n_estimators' : ('N_ESTIMATORS', int),
            'max_features' : ('MAX_FEATURES', int),
            'min_samples_leaf' : ('MIN_SAMPLES_LEAF', int),
            'max_depth' : ('MAX_DEPTH', int),
            'split_threshold' : ('SPLIT_THRESHOLD', float),
            'random_state' : ('SEED', int),
            'thread_ratio' : ('THREAD_RATIO', float),
            'allow_missing_dependent' : ('ALLOW_MISSING_LABEL', bool),
            'sample_fraction' : ('SAMPLE_FRACTION', float),
            'compression' : ('COMPRESSION', bool),
            'max_bits' : ('MAX_BITS', int),
            'quantize_rate' : ('QUANTIZE_RATE', float)},
        'HGBT' : {
            'n_estimators' : ('N_ESTIMATORS', int),
            'random_state' : ('SEED', int),
            'subsample' : ('SUBSAMPLE', float),
            'max_depth' : ('MAX_DEPTH', int),
            'split_threshold' : ('SPLIT_THRESHOLD', float),
            'learning_rate' : ('LEARNING_RATE', float),
            'split_method' : ('SPLIT_METHOD', {'exact' :'exact',
                                               'sketch' :'sketch',
                                               'sampling' :'sampling'}),
            'sketch_eps' : ('SKETCH_ESP', float),
            'min_sample_weight_leaf' : ('MIN_SAMPLES_WEIGHT_LEAF', float),
            'ref_metric' : ('REF_METRIC', ListOfStrings),
            'min_samples_leaf' : ('MIN_SAMPLES_LEAF', int),
            'max_w_in_split' : ('MAX_W_IN_SPLIT', float),
            'col_subsample_split' : ('COL_SUBSAMPLE_SPLIT', float),
            'col_subsample_tree' : ('COL_SUBSAMPLE_TREE', float),
            'lamb' : ('LAMB', float),
            'alpha' : ('ALPHA', float),
            'base_score' : ('BASE_SCORE', float),
            'adopt_prior' : ('START_FROM_AVERAGE', bool),
            'thread_ratio' : ('THREAD_RATIO', float),
            'evaluation_metric' : ('EVALUATION_METRIC',
                                   {'rmse' : 'RMSE', 'mae' : 'MAE'})},
        'SVM' : {
            'c' : ('SVM_C', float),
            'kernel' : ('KERNEL_TYPE', int, {'linear' :0, 'poly' :1, 'rbf' :2, 'sigmoid' :3}),
            'degree' : ('POLY_DEGREE', int),
            'gamma' : ('RBF_GAMMA', float),
            'coef_lin' : ('COEF_LIN', float),
            'coef_const' : ('COEF_CONST', float),
            'shrink' : ('SHRINK', bool),
            'regression_eps' : ('REGRESSION_EPS', float),
            'compression' : ('COMPRESSION', bool),
            'max_bits' : ('MAX_BITS', int),
            'max_quantization_iter' : ('MAX_QUANTIZATION_ITER', int),
            'tol' : ('TOL', float),
            'evaluation_seed' : ('EVALUATION_SEED', int),
            'scale_label' : ('SCALE_LABEL', bool),
            'scale_info' : ('SCALE_INFO', int, {'no' :0, 'standardization' :1, 'rescale' :2}),
            'handle_missing' : ('HANDLE_MISSING', bool),
            'category_weight' : ('CATEGORY_WEIGHT', float),
            'thread_ratio' : ('THREAD_RATIO', float),
            'evaluation_metric' : ('EVALUATION_METRIC', {'rmse' : 'RMSE'})}
    }
    __partition_dict = dict(no=0, predefined=1, random=2)

    def __init__(self,
                 func,
                 **kwargs):
        super(UnifiedRegression, self).__init__()
        setattr(self, 'hanaml_parameters', pal_param_register())
        self.func = self._arg('Function name', func, self.func_dict)
        self.params = dict(**kwargs)
        self.__pal_params = {}
        if self.func in ['DT', 'HGBT', 'SVM', 'MLR', 'GLM', 'MLP']:
            func_map = dict(dict(self.map_dict[self.func],
                                 **dict(self.__cv_dict)),
                            **dict(self.__param_grid_dict))
        elif self.func == 'POL':
            func_map = dict(self.map_dict[self.func], **dict(self.__cv_dict))
        else:
            func_map = self.map_dict[self.func]
        for parm in self.params:
            if parm in func_map.keys():
                parm_val = self.params[parm]
                arg_map = func_map[parm]
                if arg_map[1] == ListOfStrings and isinstance(parm_val, str):
                    parm_val = [parm_val]
                if len(arg_map) == 2:
                    self.__pal_params[arg_map[0]] = (self._arg(parm, parm_val, arg_map[1]), arg_map[1])
                else:
                    self.__pal_params[arg_map[0]] = (self._arg(parm, parm_val, arg_map[2]), arg_map[1])
            else:
                err_msg = "'{}' is not a valid parameter name for initializing a {} model".format(parm, func)
                logger.error(err_msg)
                raise KeyError(err_msg)
        self.model_ = None
        self.statistics_ = None
        self.optimal_param_ = None
        self.partition_ = None

        self.__report_builder = None
        self.__param_rows = None

    def update_cv_params(self, name, value, typ):
        """
        Update parameters for model-evaluation/parameter-selection.
        """
        if name in self.__cv_dict.keys():
            self.__pal_params[self.__cv_dict[name][0]] = (value, typ)

    def __map_param(self, name, value, typ):#pylint:disable=no-self-use
        tpl = ()
        if typ in [int, bool]:
            tpl = (name, value, None, None)
        elif typ == float:
            tpl = (name, None, value, None)
        elif typ in [str, ListOfStrings]:
            tpl = (name, None, None,
                   value.upper() if '_METRIC' in name else value)
        elif isinstance(typ, dict):
            val = value
            if isinstance(val, (int, float)):
                tpl = (name, val, None, None)
            else:
                tpl = (name, None, None, val)
        return tpl

    @trace_sql
    def fit(self,#pylint: disable=too-many-branches, too-many-statements, unused-argument
            data,
            key=None,
            features=None,
            label=None,
            purpose=None,
            partition_method=None,
            partition_random_state=None,
            training_percent=None,
            output_partition_result=None,
            categorical_variable=None,
            build_report=True):
        """
        Fit function for unified regression.

        Parameters
        ----------

        data : DataFrame
            Training data.

        key : str, optional
            Name of the ID column. If ``key`` is not provided, it is assumed
            that the input has no ID column.

        features : list of str, optional
            Names of the feature columns.

            If ``features`` is not provided, it defaults to all non-ID,
            non-label columns.

        label : str or list of str, optional
            Name of the dependent variable.

            Should be a list of two strings for GLM models with ``family`` being 'binomial'.

            If ``label`` is not provided, it defaults to:

              - the first non-key column of ``data``, when ``func`` parameter from initialization function
                takes the following values:
                - 'GeometricRegression', 'PolynomialRegression', 'LinearRegression',
                  'ExponentialRegression', 'GLM' (except when ``family`` is 'binomial')
              - the first two non-key columsn of ``data``, when ``func`` parameter in initialization function
                takes the value of 'GLM' and ``familly`` is specified as 'binomial'.


        purpose : str, optional
            Indicates the name of purpose column which is used for predefined data partition.

            The meaning of value in the column for each data instance is shown below:

                - 1 : training
                - 2 : testing

            Mandatory and valid only when ``partition_method`` is 'predefined'..

        partition_method : {'no', 'predefined', 'random'}, optional
            Defines the way to divide the dataset.
                - 'no' : no partition.
                - 'predefined' : predefined partition.
                - 'random' : random partition.

            Defaults to 'no'.

        partition_random_state : int, optional
            Indicates the seed used to initialize the random number generator for data partition.

            Valid only when ``partition_method`` is set to 'random'.

                - 0 : Uses the system time.
                - Not 0 : Uses the specified seed.

            Defaults to 0.

        training_percent : float, optional
            The percentage of data used for training.

            Value range: 0 <= value <= 1.

            Defaults to 0.8.

        output_partition_result : bool, optional
            Specifies whether or not to outupt the partition result of ``data`` in data partition table.

            Valid only when ``key`` is provided and ``partition_method`` is set to 'random'.

            Defaults to False.

        categorical_variable : str or list of str, optional
            Specifies INTEGER column(s) that should be treated as categorical.

            Other INTEGER columns will be treated as continuous.

            No default value.
        """
        conn = data.connection_context
        require_pal_usable(conn)
        key = self._arg('key', key, str)
        partition_method = self._arg('partition_method', partition_method, self.__partition_dict)
        partition_random_state = self._arg('partition_random_state', partition_random_state, int)
        training_percent = self._arg('training_percent', training_percent, float)
        output_partition_result = self._arg('output_partition_result', output_partition_result, bool)
        purpose = self._arg('purpose', purpose, str, partition_method == 1)
        if partition_method != 1:#purpose ineffective when partition method is not 1
            purpose = None
        if isinstance(features, str):
            features = [features]
        features = self._arg('features', features, ListOfStrings)
        label = self._arg('label', label, str)
        cols = data.columns
        if key is not None:
            id_col = [key]
            cols.remove(key)
        else:
            id_col = []
        if purpose is not None:
            cols.remove(purpose)
            purpose = [purpose]
        else:
            purpose = []
        if label is None:
            if self.func == 'GLM' and 'group_response' in self.params.keys() and self.params['group_response'] is True:#pylint:disable=line-too-long
                label = cols[-2:]
            else:
                label = cols[-1]
        if isinstance(label, (list, tuple)):
            for lab in label:
                cols.remove(lab)
        else:
            cols.remove(label)
        if isinstance(label, str):
            label = [label]
        if features is None:
            features = cols
        data_ = data[id_col + features + label + purpose]
        unique_id = str(uuid.uuid1()).replace('-', '_').upper()
        param_rows = [('FUNCTION', None, None, self.func),
                      ('KEY', key is not None, None, None),
                      ('PARTITION_METHOD', partition_method, None, None),
                      ('PARTITION_RANDOM_SEED', partition_random_state, None, None),
                      ('PARTITION_TRAINING_PERCENT', None, training_percent, None),
                      ('OUTPUT_PARTITION_RESULT', output_partition_result, None, None)]
        if self.func == 'SVM':
            param_rows.extend([('EVALUATION_METRIC', None, None, 'RMSE')])
        for name in self.__pal_params:
            value, typ = self.__pal_params[name]
            if isinstance(value, (list, tuple)):
                if name == 'HIDDEN_LAYER_SIZE':
                    value = ', '.join([str(v) for v in value])
                    param_rows.extend([(name, None, None, value)])
                elif name == 'HIDDEN_LAYER_SIZE_OPTIONS':
                    value = ', '.join([str(v) for v in value])
                    value = value.replace('(', '"').replace(')', '"')
                    value = value.replace('[', '"').replace(']', '"')
                    value = '{' + value + '}'
                    param_rows.extend([(name, None, None, value)])
                elif name in ['ACTIVATION_OPTIONS', 'OUTPUT_ACTIVATION_OPTIONS']:
                    value = ', '.join([str(self.__activation_map[v]) for v in value])
                    value = '{' + value + '}'
                    param_rows.extend([(name, None, None, value)])
                elif name == 'ORDERING':
                    tpl = [('ORDERING', None, None, ', '.join(value))]
                    param_rows.extend(tpl)
                elif name == 'DEGREE_RANGE':
                    tpl = [('DEGREE_RANGE', None, None, str(value))]
                    param_rows.extend(tpl)
                elif name == 'DEGREE_VALUES':
                    tpl = [('DEGREE_VALUES', None, None,
                            '{' + ','.join([str(x) for x in value])) + '}']
                else:
                    for val in value:
                        tpl = [self.__map_param(name, val, typ)]
                        param_rows.extend(tpl)
            elif typ == dict:
                if name == '_RANGE':
                    for var in value:
                        rge = [str(v) for v in value[var]]
                        rge_str = '[' + ((',' if len(rge) == 3 else ',,'). join(rge)) + ']'
                        tpl = [(self.map_dict[self.func][var][0] + name, None, None, rge_str)]
                        param_rows.extend(tpl)
                elif name == '_VALUES':
                    for var in value:
                        vvr = [str(v) for v in value[var]]
                        vvr_str = '{' + ','.join(vvr) + '}'
                        tpl = [(self.map_dict[self.func][var][0] + name, None, None, vvr_str)]
                        param_rows.extend(tpl)
            else:
                tpl = [self.__map_param(name, value, typ)]
                param_rows.extend(tpl)
        if isinstance(categorical_variable, str):
            categorical_variable = [categorical_variable]
        categorical_variable = self._arg('categorical_variable', categorical_variable, ListOfStrings)
        if categorical_variable is not None:
            param_rows.extend([('CATEGORICAL_VARIABLE', None, None, var) for var in categorical_variable])
        outputs = ['MODEL', 'STATS', 'OPT_PARAM', 'PARTITION', 'PLACE_HOLDER1', 'PLACE_HOLDER2']
        outputs = ['#PAL_UNIFIED_REGRESSION_{}_{}_{}'.format(tbl, self.id, unique_id)
                   for tbl in outputs]
        model_tbl, stats_tbl, opt_param_tbl, partition_tbl, _, _ = outputs
        try:
            call_pal_auto(conn,
                          'PAL_UNIFIED_REGRESSION',
                          data_,
                          ParameterTable().with_data(param_rows),
                          *outputs)
        except dbapi.Error as db_err:
            logger.error(str(db_err))
            try_drop(conn, outputs)
            raise
        #pylint: disable=attribute-defined-outside-init
        self.model_ = conn.table(model_tbl)
        self.statistics_ = conn.table(stats_tbl)
        self.optimal_param_ = conn.table(opt_param_tbl)
        self.partition_ = conn.table(partition_tbl)

        self.__param_rows = param_rows
        if build_report:
            self.build_report()

    @trace_sql
    def predict(self, data, key,
                features=None,
                model=None,
                thread_ratio=None,
                prediction_type=None,
                significance_level=None,
                handle_missing=None,
                block_size=None):#pylint:disable=unused-argument
        """
        Predict with the regression model.

        Parameters
        ----------
        data :  DataFrame
            Data to be predicted.

        key : str
            Name of the ID column.

        features : ListOfStrings, optional
            Names of feature columns in data for prediction.

            Defaults all non-ID columns in `data` if not provided.

        model : DataFrame
            Fitted regression model.

            Defaults to self.model_.

        thread_ratio : float, optional
            Controls the proportion of available threads to use for prediction.

            The value range is from 0 to 1, where 0 indicates a single thread,
            and 1 indicates up to all available threads.

            Values between 0 and 1 will use that percentage of available threads.

            Values outside this range tell PAL to heuristically determine the number of threads to use.

            Defaults to the PAL's default value.

        prediction_type : str, optional
            Specifies the type of prediction. Valid options include:

                - 'response' : direct response (with link)
                - 'link' : linear response (without link)

            Valid only for GLM models.

            Defaults to 'response'.

        significance_level : float, optional
            Specifies significance level for the confidence interval and prediction interval.

            Valid only for GLM models where IRLS method is applied.

            Defaults to 0.05.

        handle_missing : str, optional
            Specifies the way to handle missing values. Valid options include:

                - 'skip' : skip(i.e. remove) rows with missing values
                - 'fill_zero' : replace missing values with 0.

            Valid only for GLM models.

            Defaults to 'fill_zero'.

        block_size : int, optional
            Specifies the number of data loaded per time during scoring.

                - 0: load all data once
                - Others: the specified number

            This parameter is for reducing memory consumption, especially as the predict data is huge,
            or it consists of a large number of missing independent variables. However, you might lose some efficiency.

            Valid only for RandomDecisionTree(RDT) models.

            Defaults to 0.

        Returns
        -------
            DataFrame
                - Prediction result, structured as follows:

                    -  1st column : ID
                    -  2nd column : SCORE, i.e. predicted value
                    -  3rd column : UPPER_BOUND, upper bound of predicted values
                    -  4th column : LOWER_BOUND, lower bound of predicted values
                    -  5th column : REASON, interpretation of prediction results
        """
        type_dict = dict(response='response', link='link')
        if model is None and getattr(self, 'model_') is None:
            raise FitIncompleteError("Model not initialized. Perform a fit first.")
        conn = data.connection_context
        require_pal_usable(conn)
        cols = data.columns
        key = self._arg('key', key, str, required=True)
        cols.remove(key)
        if isinstance(features, str):
            features = [features]
        features = self._arg('features', features, ListOfStrings)
        if features is None:
            features = cols
        data_ = data[[key] + features]
        prediction_type = self._arg('prediction_type', prediction_type, type_dict)
        thread_ratio = self._arg('thread_ratio', thread_ratio, float)
        significance_level = self._arg('significance_level', significance_level, float)
        handle_missing = self._arg('handle_missing', handle_missing, dict(skip=1, fill_zero=2))
        block_size = self._arg('block_size', block_size, int)
        if model is None:
            model = self.model_
        unique_id = str(uuid.uuid1()).replace('-', '_').upper()
        outputs = ['#PAL_UNIFIED_REGR_PREDICT_{}_TBL_{}_{}'.format(tb, self.id, unique_id) for tb in ['RESULT', 'PH']]
        param_rows = [('FUNCTION', None, None, self.func),
                      ('THREAD_RATIO', None, thread_ratio, None),
                      ('TYPE', None, None, prediction_type),
                      ('SIGNIFICANCE_LEVEL', None, significance_level, None),
                      ('HANDLE_MISSING', handle_missing, None, None),
                      ('BLOCK_SIZE', block_size, None, None)]
        try:
            call_pal_auto(conn,
                          'PAL_UNIFIED_REGRESSION_PREDICT',
                          data_,
                          model,
                          ParameterTable().with_data(param_rows),
                          *outputs)
        except dbapi.Error as db_err:
            logger.exception(str(db_err))
            try_drop(conn, outputs)
            raise
        return conn.table(outputs[0])

    @trace_sql
    def score(self, data, key,
              features=None,
              label=None,
              model=None,
              prediction_type=None,
              significance_level=None,
              handle_missing=None,
              thread_ratio=None,
              block_size=None):
        """
        Users can use the score function to evaluate the model quality.
        In the Unified regression, statistics and metrics are provided to show the model quality.


        Parameters
        ----------
        data :  DataFrame
            Data for scoring.

        key : str
            Name of the ID column.

        features : ListOfString or str, optional
            Names of feature columns.

            Defaults to all non-ID, non-label columns if not provided.

        label : str, optional
            Name of the label column.

            Defaults to the last non-ID column if not provided.

        model : DataFrame
            Fitted regression model.

            Defaults to self.model_.

        thread_ratio : float, optional
            Controls the proportion of available threads to use for prediction.

            The value range is from 0 to 1, where 0 indicates a single thread,
            and 1 indicates up to all available threads.

            Values between 0 and 1 will use that percentage of available threads.

            Values outside this range tell PAL to heuristically determine the number of threads to use.

            Defaults to the PAL's default value.

        prediction_type : str, optional
            Specifies the type of prediction. Valid options include:

                - 'response' : direct response (with link)
                - 'link' : linear response (without link)

            Valid only for GLM models.

            Defaults to 'response'.

        significance_level : float, optional
            Specifies significance level for the confidence interval and prediction interval.

            Valid only for GLM models where IRLS method is applied.

            Defaults to 0.05.

        handle_missing : str, optional
            Specifies the way to handle missing values. Valid options include:

                - 'skip' : skip rows with missing values
                - 'fill_zero' : replace missing values with 0.

            Valid only for GLM models.

            Defaults to 'fill_zero'.

        block_size : int, optional
            Specifies the number of data loaded per time during scoring.

                - 0: load all data once
                - Others: the specified number

            This parameter is for reducing memory consumption, especially as the predict data is huge,
            or it consists of a large number of missing independent variables. However, you might lose some efficiency.

            Valid only for RandomDecisionTree models.

            Defaults to 0.

        Returns
        -------
        DataFrame
            - Prediction result, structured as follows:

                -  1st column : ID
                -  2nd column : SCORE, i.e. predicted values
                -  3rd column : UPPER_BOUND, upper bound of predicted values
                -  4th column : LOWER_BOUND, lower bound of predicted values
                -  5th column : REASON, interpretation of prediction results

            - Statistics results

                Names and values of related statistics.
        """
        type_dict = dict(response='response', link='link')
        if model is None and getattr(self, 'model_') is None:
            raise FitIncompleteError("Model not initialized. Perform a fit first.")
        conn = data.connection_context
        require_pal_usable(conn)
        cols = data.columns
        key = self._arg('key', key, str, required=True)
        cols.remove(key)
        if isinstance(features, str):
            features = [features]
        features = self._arg('features', features, ListOfStrings)
        label = self._arg('label', label, str)
        if label is None:
            label = cols[-1]
        cols.remove(label)
        if features is None:
            features = cols
        thread_ratio = self._arg('thread_ratio', thread_ratio,
                                 float)
        prediction_type = self._arg('prediction_type', prediction_type, type_dict)
        significance_level = self._arg('significance_level', significance_level, float)
        handle_missing = self._arg('handle_missing', handle_missing, dict(skip=1, fill_zero=2))
        block_size = self._arg('block_size', block_size, int)
        data_ = data[[key] + features + [label]]
        if model is None:
            model = self.model_
        unique_id = str(uuid.uuid1()).replace('-', '_').upper()
        outputs = ['RESULT', 'STATS']
        outputs = ['#PAL_UNIFIED_REGRESSION_SCORE_{}_TBL_{}_{}'.format(name, self.id, unique_id)
                   for name in outputs]
        result_tbl, stats_tbl = outputs
        param_rows = [('FUNCTION', None, None, self.func),
                      ('THREAD_RATIO', None, thread_ratio, None),
                      ('TYPE', None, None, prediction_type),
                      ('SIGNIFICANCE_LEVEL', None, significance_level, None),
                      ('HANDLE_MISSING', handle_missing, None, None),
                      ('BLOCK_SIZE', block_size, None, None)]
        # SQLTRACE
        conn.sql_tracer.set_sql_trace(self, self.__class__.__name__, 'Score')
        try:
            call_pal_auto(conn,
                          'PAL_UNIFIED_REGRESSION_SCORE',
                          data_,
                          model,
                          ParameterTable().with_data(param_rows),
                          *outputs)
        except dbapi.Error as db_err:
            logger.exception(str(db_err))
            try_drop(conn, outputs)
            raise
        return conn.table(result_tbl), conn.table(stats_tbl)

    def build_report(self):
        """
        Build model report.
        """
        try:
            if self.__param_rows is None:
                raise Exception('To build a report, you must call the fit method firstly.')

            if self.__report_builder is None:
                self.__report_builder = UnifiedRegressionReportBuilder()

            parameter_df = pd.DataFrame(data=self.__param_rows, columns=ParameterReportBuilder.get_table_columns())
            self.__report_builder \
                .set_statistic_table(self.statistics_) \
                .set_parameter_table(parameter_df) \
                .set_optimal_parameter_table(self.optimal_param_.collect()) \
                .build()
        except Exception as err:
            logger.error(str(err))
            raise

    def generate_html_report(self, filename):
        """
        Save model report as a html file.

        Parameters
        ----------
        filename : str
            Html file name.
        """
        if self.__report_builder is None:
            raise Exception('To generate a report, you must call the build_report method firstly.')

        self.__report_builder.generate_html_report(filename)

    def generate_notebook_iframe_report(self):
        """
        Render model report as a notebook iframe.
        """
        if self.__report_builder is None:
            raise Exception('To generate a report, you must call the build_report method firstly.')

        self.__report_builder.generate_notebook_iframe_report()
