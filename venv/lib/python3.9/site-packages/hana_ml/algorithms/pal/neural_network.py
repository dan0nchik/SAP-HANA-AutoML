"""
This module contains Python wrappers for PAL Multi-layer Perceptron algorithm.

The following classes are available:

    * :class:`MLPClassifier`
    * :class:`MLPRegressor`
"""

#pylint: disable=too-many-arguments, too-many-lines
#pylint: disable=relative-beyond-top-level
#pylint: disable=line-too-long, unused-variable
import logging
import uuid
import itertools
from hdbcli import dbapi

from hana_ml.dataframe import quotename
from hana_ml.ml_exceptions import FitIncompleteError
from hana_ml.ml_base import try_drop
from hana_ml.algorithms.pal.pal_base import (
    PALBase,
    ParameterTable,
    execute_logged,
    _TEXT_TYPES,
    _INT_TYPES,
    ListOfTuples,
    ListOfStrings,
    pal_param_register,
    require_pal_usable,
    call_pal_auto
)
from .sqlgen import trace_sql
from . import metrics

logger = logging.getLogger(__name__) #pylint: disable=invalid-name


class _MLPBase(PALBase):#pylint: disable=too-many-instance-attributes, too-few-public-methods
    """
    Base class for Multi-layer Perceptron.
    """
    activation_map = {
        'tanh' : 1,
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
        'relu' : 13
    }
    #functionality_map = {'classification': 0, 'regression': 1}
    style_map = {'batch' : 0, 'stochastic' : 1}
    norm_map = {'no': 0, 'z-transform' : 1, 'scalar' : 2}
    weight_map = {'all-zeros' : 0, 'normal' : 1, 'uniform' : 2,
                  'variance-scale-normal' : 3, 'variance-scale-uniform' : 4}
    resampling_method_list = ('cv', 'stratified_cv', 'bootstrap', 'stratified_bootstrap')
    evaluation_metric_map = {'accuracy' : 'ACCURACY', 'f1_score' : 'F1_SCORE',
                             'auc_onevsrest' : 'AUC_1VsRest', 'auc_pairwise': 'AUC_pairwise',
                             'rmse' : 'RMSE'}
    search_strategy_list = ('grid', 'random')
    #mand_params_map = {'activation' : 'HIDDEN_LAYER_ACTIVE_FUNC',
    #                   'output_activation' : 'OUTPUT_LAYER_ACTIVE_FUNC',
    #                   'hidden_layer_size' : 'HIDDEN_LAYER_SIZE'}
    range_params_map = {'learning_rate' : 'LEARNING_RATE',
                        'momentum' : 'MOMENTUM_FACTOR',
                        'batch_size' : 'MINI_BATCH_SIZE'}
    #range_params = ('learning_rate', 'momentum', 'batch_size')
    def __init__(self,#pylint: disable=too-many-arguments, too-many-branches, too-many-statements, too-many-locals
                 functionality,
                 activation=None,
                 activation_options=None,
                 output_activation=None,
                 output_activation_options=None,
                 hidden_layer_size=None,
                 hidden_layer_size_options=None,
                 max_iter=None,
                 training_style=None,
                 learning_rate=None,
                 momentum=None,
                 batch_size=None,
                 normalization=None,
                 weight_init=None,
                 categorical_variable=None,
                 resampling_method=None,
                 evaluation_metric=None,
                 fold_num=None,
                 repeat_times=None,
                 search_strategy=None,
                 random_search_times=None,
                 random_state=None,
                 timeout=None,
                 progress_indicator_id=None,
                 param_values=None,
                 param_range=None,
                 thread_ratio=None):
        super(_MLPBase, self).__init__()
        setattr(self, 'hanaml_parameters', pal_param_register())
        #type checking
        self.activation = self._arg('activation', activation,
                                    self.activation_map)
        act_options = self._arg('activation_options', activation_options,
                                ListOfStrings)
        if self.activation is None and act_options is None:
            msg = ("'activation' and 'activation_options' cannot both be None.")
            logger.error(msg)
            raise ValueError(msg)
        if act_options is not None:
            self.activation_options = []
            for act in act_options:
                if act not in list(self.activation_map.keys()):#pylint:disable=bad-option-value
                    msg = ("'{}' is an invalid activation function.".format(act))
                    logger.error(msg)
                    raise ValueError(msg)
                self.activation_options.append(self.activation_map[act])
            self.activation_options = str(self.activation_options).replace('[', '{').replace(']', '}')#pylint:disable=line-too-long
        else:
            self.activation_options = None
        self.output_activation = self._arg('output_activation', output_activation,
                                           self.activation_map)
        out_act_options = self._arg('output_activation_options',
                                    output_activation_options,
                                    ListOfStrings)
        if self.output_activation is None and out_act_options is None:
            msg = ("'output_activation' and 'output_activation_options' "+
                   "cannot both be None.")
            logger.error(msg)
            raise ValueError(msg)
        if out_act_options is not None:
            self.output_activation_options = []
            for act_out in out_act_options:
                if act_out not in list(self.activation_map.keys()):#pylint:disable=bad-option-value
                    msg = ("'{}' is an invalid activation function".format(act_out)+
                           " for output layer.")
                    logger.error(msg)
                    raise ValueError(msg)
                self.output_activation_options.append(self.activation_map[act_out])
            self.output_activation_options = str(self.output_activation_options).replace('[', '{').replace(']', '}')#pylint:disable=line-too-long
        else:
            self.output_activation_options = None
        if hidden_layer_size is not None:
            if isinstance(hidden_layer_size, (tuple, list)) and all(isinstance(x, _INT_TYPES) for x in hidden_layer_size):#pylint:disable=line-too-long
                self.hidden_layer_size = ','.join(str(x) for x in hidden_layer_size)
            else:
                msg = "Parameter 'hidden_layer_size' must be type of tuple/list of int."
                logger.error(msg)
                raise TypeError(msg)
        else:
            self.hidden_layer_size = None
        hls_options = self._arg('hidden_layer_size_options', hidden_layer_size_options,
                                ListOfTuples)
        if hls_options is not None:
            #self.hidden_layer_size_options = []
            for hls_option in hls_options:
                if not all(isinstance(x, _INT_TYPES) for x in hls_option):
                    msg = ("Valid option of 'hidden_layer_size' must be "+
                           "tuple of int, while provided options contain"+
                           " values of invalid types.")
                    logger.error(msg)
                    raise TypeError(msg)
            hls_options = str(hls_options).replace('[', '{').replace(']', '}')
            self.hidden_layer_size_options = hls_options.replace('(', '"').replace(')', '"')
        else:
            self.hidden_layer_size_options = None
        if self.hidden_layer_size is None and self.hidden_layer_size_options is None:
            msg = ("'hidden_layer_size' and 'hidden_layer_size_options' "+
                   "cannot both be None.")
            logger.error(msg)
            raise ValueError(msg)
        self.max_iter = self._arg('max_iter', max_iter, int)
        self.functionality = functionality
        if training_style is not None:
            self.training_style = self._arg('training_style', training_style.lower(),
                                            self.style_map)
        else:
            self.training_style = None
        if learning_rate is None and training_style.lower() == 'stochastic':
            msg = "When 'training_style' is 'stochastic', 'learning_rate' cannot be None."
            logger.error(msg)
            raise ValueError(msg)
        #if learning_rate is not None and training_style.lower() != 'stochastic':
        #    msg = ('Parameter learning rate is invalid ' +
        #           'when training_style is not stochastic.')
        #    logger.warn(msg)
        #    raise ValueError(msg)
        self.learning_rate = self._arg('learning_rate', learning_rate, float)
        if momentum is None and training_style == 'stochastic':
            msg = "When 'training_style' is 'stochastic', 'momentum' cannot be None."
            logger.error(msg)
            raise ValueError(msg)
        #if momentum is not None and training_style.lower() != 'stochastic':
        #    msg = ('Parameter momentum is invalid ' +
        #           'when training_style is not stochastic.')
        #    logger.warn(msg)
        #    raise ValueError(msg)
        self.momentum = self._arg('momentum', momentum, float)
        #if batch_size is not None and training_style.lower() != 'stochastic':
        #    msg = ('Parameter batch_size is only valid ' +
        #           'when training_style is stochastic.')
        #    logger.error(msg)
        #    raise ValueError(msg)
        self.batch_size = self._arg('batch_size', batch_size, int)
        self.normalization = self._arg('normalization', normalization, self.norm_map)
        self.weight_init = self._arg('weight_init', weight_init, self.weight_map)
        if categorical_variable is not None:
            if isinstance(categorical_variable, _TEXT_TYPES):
                categorical_variable = [categorical_variable]
            if isinstance(categorical_variable, list) and all(isinstance(x, _TEXT_TYPES)for x in categorical_variable):#pylint:disable=line-too-long
                self.categorical_variable = categorical_variable
            else:
                msg = "Parameter 'categorial_variable' must be type of str or list of str."
                logger.error(msg)
                raise TypeError(msg)
        else:
            self.categorical_variable = None
        self.thread_ratio = self._arg('thread_ratio', thread_ratio, float)
        self.resampling_method = self._arg('resampling_method', resampling_method, str)
        if self.resampling_method is not None:

            if self.resampling_method not in self.resampling_method_list:#pylint:disable=line-too-long, bad-option-value
                msg = ("Resampling method '{}' is not available ".format(self.resampling_method)+
                       "for model evaluation/parameter selection in MLP" +
                       " classification and regression.")
                logger.error(msg)
                raise ValueError(msg)
            if self.functionality == 1 and self.resampling_method in ('cv', 'boostrap'):
                msg = ("Resampling method '{}' is invalid ".format(self.resampling_method)+
                       "for model evaluation/parameter selection in MLP regression.")
                logger.error(msg)
                raise ValueError(msg)

        self.evaluation_metric = self._arg('evaluation_metric', evaluation_metric,
                                           self.evaluation_metric_map)
        if self.evaluation_metric is not None:
            if self.functionality == 0 and self.evaluation_metric == 'RMSE':#pylint:disable=line-too-long, bad-option-value
                msg = ("Metric 'rmse' is invalid for classification.")
                logger.error(msg)
                raise ValueError(msg)
            if self.functionality == 1 and self.evaluation_metric in ('ACCURACY', 'F1_SCORE', 'AUC_1VSRest', 'AUC_pairwise'):#pylint:disable=line-too-long, bad-option-value
                msg = ("Metric '{}' is invalid ".format(evaluation_metric)+
                       "for regression.")
                logger.error(msg)
                raise ValueError(msg)
        self.fold_num = self._arg('fold_num', fold_num, int)
        self.repeat_times = self._arg('repeat_times', repeat_times, int)
        if self.resampling_method in ('cv', 'stratified_cv') and self.fold_num is None:
            msg = ("'fold_num' cannot be None when 'resampling_method'"+
                   " is set to 'cv' or 'stratified_cv'.")
            logger.error(msg)
            raise ValueError(msg)
        #need some check work here
        self.search_strategy = self._arg('search_strategy', search_strategy, str)
        if self.search_strategy is not None:
            if self.search_strategy not in self.search_strategy_list:
                msg = ("Search strategy '{}' is invalid ".format(self.search_strategy)+
                       "for parameter selection.")
                logger.error(msg)
                raise ValueError(msg)
        self.random_search_times = self._arg('random_search_times', random_search_times,
                                             int)
        if self.search_strategy == 'random' and self.random_search_times is None:
            msg = ("'random_search_times' cannot be None when 'search_strategy'"+
                   " is set to 'random'.")
            logger.error(msg)
            raise ValueError(msg)
        self.random_state = self._arg('random_state', random_state, int)
        self.timeout = self._arg('timeout', timeout, int)
        self.progress_indicator_id = self._arg('progress_indicator_id', progress_indicator_id, str)
        if isinstance(param_range, dict):
            param_range = [(x, param_range[x]) for x in param_range]
        if isinstance(param_values, dict):
            param_values = [(x, param_values[x]) for x in param_values]
        self.param_values = self._arg('param_values', param_values, ListOfTuples)
        #Validating the input values for parameter selection
        #param values and range valid only when search strategy being specified
        #print("Current training style is {}......".format(self.training_style))
        if self.param_values is not None and self.search_strategy is not None:
            for x in self.param_values:#pylint:disable=invalid-name
                if len(x) != 2:#pylint:disable=bad-option-value
                    msg = ("Each tuple that specifies the values of a parameter should"+
                           " contain exactly 2 elements: 1st is parameter name,"+
                           " 2nd is a list of valid values.")
                    logger.error(msg)
                    raise ValueError(msg)
                if x[0] not in list(self.range_params_map.keys()):
                    msg = ("Specifying the values of '{}' for ".format(x[0])+
                           "parameter selection is invalid.")
                    logger.error(msg)
                    raise ValueError(msg)
                if self.training_style == 1 and  x[0] == 'batch_size':
                    if not (isinstance(x[1], list) and all(isinstance(t, _INT_TYPES) for t in x[1])):#pylint:disable=line-too-long
                        msg = "Valid values of 'batch_size' must be a list of int."
                        logger.error(msg)
                        raise TypeError(msg)
                if self.training_style == 1:
                    if not (isinstance(x[1], list) and all(isinstance(t, (float, int)) for t in x[1])):#pylint:disable=line-too-long
                        msg = ("Valid values of '{}' ".format(x[0])+
                               "must be a list of numericals.")
                        logger.error(msg)
                        raise TypeError(msg)
                #else:
                #    print("Verified {} values......".format(x[0]))
        self.param_range = self._arg('param_range', param_range, ListOfTuples)
        if self.search_strategy is not None:
            if self.search_strategy == 'grid':
                rsz = [3]
            else:
                rsz = [2, 3]
        if self.param_range is not None and self.search_strategy is not None:
            for x in self.param_range:#pylint:disable=invalid-name
                if len(x) != 2:#pylint:disable=bad-option-value
                    msg = ("Each tuple that specifies the range of a parameter should contain"+
                           " exactly 2 elements: 1st is parameter name, 2nd is value range.")
                    logger.error(msg)
                    raise ValueError(msg)
                if x[0] not in list(self.range_params_map.keys()):
                    msg = ("Parameter '{}' is invalid for ".format(x[0])+
                           "range specification in parameter selection.")
                    logger.error(msg)
                    raise ValueError(msg)
                if x[0] == 'batch_size':
                    if not(isinstance(x[1], list) and len(x[1]) in rsz and all(isinstance(t, int) for t in x[1])):#pylint:disable=line-too-long
                        msg = ("The provided range of 'batch_size' is either not "+
                               "a list of int, or it contains wrong number of values.")
                        logger.error(msg)
                        raise TypeError(msg)
                if not (isinstance(x[1], list) and len(x[1]) in rsz and all(isinstance(t, (float, int)) for t in x[1])):#pylint:disable=line-too-long
                    msg = ("The provided range of '{}' is either not ".format(x[0])+
                           "a list of numericals, or it contains the wrong number of values.")
                    logger.error(msg)
                    raise TypeError(msg)

    @trace_sql
    def _fit(self, data, key=None, features=None, label=None, categorical_variable=None):#pylint: disable=too-many-locals, too-many-statements, too-many-branches
        conn = data.connection_context
        require_pal_usable(conn)
        has_id = False
        index_col = None
        #Do we need type check for key column and also check its existence in df?
        if key is not None:
            has_id = True
            index_col = key
        cols_left = data.columns
        if label is None:
            label = data.columns[-1]
        if isinstance(label, _TEXT_TYPES):
            label = [label]
        cols_left = [x for x in cols_left if x not in label]
        if has_id:
            cols_left.remove(index_col)
        if features is None:
            features = cols_left
        used_cols = [col for col in itertools.chain([index_col], features, label)
                     if col is not None]
        if categorical_variable is not None:
            if isinstance(categorical_variable, _TEXT_TYPES):
                categorical_variable = [categorical_variable]
            if not (isinstance(categorical_variable, list) and all(isinstance(x, _TEXT_TYPES)for x in categorical_variable)):#pylint:disable=line-too-long
                msg = "Parameter 'categorial_variable' must be type of str or list of str."
                logger.error(msg)
                raise TypeError(msg)

        unique_id = str(uuid.uuid1()).replace('-', '_').upper()
        outputs = model_tbl, log_tbl, stat_tbl, optimal_tbl = [
            '#MLP_{}_TBL_{}_{}'.format(name, self.id, unique_id) for name in ['MODEL',
                                                                              'TRAINING_LOG',
                                                                              'STATISTICS',
                                                                              'OPTIMAL_PARAM']
            ]
        training_df = data[used_cols]
        param_rows = [('HIDDEN_LAYER_ACTIVE_FUNC', self.activation, None, None),
                      ('HIDDEN_LAYER_ACTIVE_FUNC_VALUES', None, None, self.activation_options),
                      ('OUTPUT_LAYER_ACTIVE_FUNC', self.output_activation, None, None),
                      ('OUTPUT_LAYER_ACTIVE_FUNC_VALUES', None, None, self.output_activation_options),
                      ('HIDDEN_LAYER_SIZE', None, None, self.hidden_layer_size),
                      ('HIDDEN_LAYER_SIZE_VALUES', None, None, self.hidden_layer_size_options),
                      ('HAS_ID', int(has_id), None, None),
                      ('MAX_ITERATION', self.max_iter, None, None),
                      ('FUNCTIONALITY', self.functionality, None, None),
                      ('TRAINING_STYLE', self.training_style, None, None),
                      ('LEARNING_RATE', None, self.learning_rate, None),
                      ('MOMENTUM_FACTOR', None, self.momentum, None),
                      ('MINI_BATCH_SIZE', self.batch_size, None, None),
                      ('NORMALIZATION', self.normalization, None, None),
                      ('WEIGHT_INIT', self.weight_init, None, None),
                      ('THREAD_RATIO', None, self.thread_ratio, None),
                      ('RESAMPLING_METHOD', None, None, self.resampling_method),
                      ('EVALUATION_METRIC', None, None, self.evaluation_metric),
                      ('FOLD_NUM', self.fold_num, None, None),
                      ('REPEAT_TIMES', self.repeat_times, None, None),
                      ('PARAM_SEARCH_STRATEGY', None, None, self.search_strategy),
                      ('RANDOM_SEARCH_TIMES', self.random_search_times, None, None),
                      ('SEED', self.random_state, None, None),
                      ('TIMEOUT', self.timeout, None, None)]

        if self.categorical_variable is not None:
            param_rows.extend([('CATEGORICAL_VARIABLE', None, None, variable)
                               for variable in self.categorical_variable])
        if categorical_variable is not None:
            param_rows.extend([('CATEGORICAL_VARIABLE', None, None, variable)
                               for variable in categorical_variable])

        param_rows.extend([('DEPENDENT_VARIABLE', None, None, name)
                           for name in label])
        if self.param_values is not None and self.search_strategy is not None:
            for x in self.param_values:#pylint:disable=invalid-name
                values = str(x[1]).replace('[', '{').replace(']', '}')
                param_rows.extend([quotename(self.range_params_map[x[0]]+"_VALUES"),
                                   None, None, values])
        if self.param_range is not None and self.search_strategy is not None:
            for x in self.param_range:#pylint:disable=invalid-name
                range_ = str(x[1])
                if len(x[1]) == 2 and self.training_style == 'stochastic':
                    range_ = range_.replace(',', ',,')
                param_rows.extend([(quotename(self.range_params_map[x[0]]+"_RANGE"),
                                    None, None, range_)])

        try:
            call_pal_auto(conn,
                          'PAL_MULTILAYER_PERCEPTRON',
                          training_df,
                          ParameterTable().with_data(param_rows),
                          *outputs)
        except dbapi.Error as db_err:
            logger.exception(str(db_err))
            try_drop(conn, outputs)
            raise

        self.model_ = conn.table(model_tbl)#pylint:disable=attribute-defined-outside-init
        self.stats_ = conn.table(stat_tbl)#pylint:disable=attribute-defined-outside-init
        self.train_log_ = conn.table(log_tbl)#pylint:disable=attribute-defined-outside-init
        self.optim_param_ = conn.table(optimal_tbl)#pylint:disable=attribute-defined-outside-init

    @trace_sql
    def _predict(self, data, key, features=None, thread_ratio=None):#pylint: disable=too-many-locals
        conn = data.connection_context
        require_pal_usable(conn)
        if not hasattr(self, 'model_'):
            raise FitIncompleteError("Model not initialized. Perform a fit first.")
        key = self._arg('key', key, str, True)
        #if features is not None:
        #    if isinstance(features, _TEXT_TYPES):
        #        features = [features]
        #    if not (isinstance(featurs, list) and (isinstance(x, _TEXT_TYPES) for x in features)):
        #        msg = ("The parameter 'features' must be of type str or list of str.")
        #        logger.error(msg)
        #        raise TypeError(msg)
        features = self._arg('features', features, ListOfStrings)
        thread_ratio = self._arg('thread_ratio', thread_ratio, float)
        unique_id = str(uuid.uuid1())
        unique_id = unique_id.replace('-', '_').upper()
        mlp_type = 'MLPCLASSIFIER' if self.functionality == 0 else 'MLPREGRESSOR'
        # 'key' is necessary for prediction data(PAL default: key, Feature)
        cols = data.columns
        cols.remove(key)
        if features is None:
            features = cols
        data_ = data[[key] + features]
        param_tbl, result_tbl, soft_max_tbl = [
            '#{}_{}_TBL_{}_{}'.format(mlp_type, name, self.id, unique_id)
            for name in ['PREDICT_CONTROL',
                         'PREDICT_RESULT',
                         'SOFT_MAX'
                        ]
            ]
        out_tables = [result_tbl, soft_max_tbl]
        param_rows = [('THREAD_RATIO', None, thread_ratio, None)]

        try:
            call_pal_auto(conn,
                          'PAL_MULTILAYER_PERCEPTRON_PREDICT',
                          data_,
                          self.model_,
                          ParameterTable(param_tbl).with_data(param_rows),
                          *out_tables)
        except dbapi.Error as db_err:
            logger.exception(str(db_err))
            try_drop(conn, out_tables)
            raise
        return (conn.table(result_tbl),
                conn.table(soft_max_tbl))


class MLPClassifier(_MLPBase):
    """
    Multi-layer perceptron (MLP) Classifier.

    Parameters
    ----------

    activation : str

        Specifies the activation function for the hidden layer.

        Valid activation functions include:
          - 'tanh',
          - 'linear',
          - 'sigmoid_asymmetric',
          - 'sigmoid_symmetric',
          - 'gaussian_asymmetric',
          - 'gaussian_symmetric',
          - 'elliot_asymmetric',
          - 'elliot_symmetric',
          - 'sin_asymmetric',
          - 'sin_symmetric',
          - 'cos_asymmetric',
          - 'cos_symmetric',
          - 'relu'

        Should not be specified only if ``activation_options`` is provided.

    activation_options : list of str, optional

        A list of activation functions for parameter selection.

        See ``activation`` for the full set of valid activation functions.

    output_activation : str

        Specifies the activation function for the output layer.

        Valid activation functions same as those in ``activation``.

        Should not be specified only if ``outupt_activation_options`` is provided.

    output_activation_options : list of str, optional

        A list of activation functions for the output layer for parameter selection.

        See ``activation`` for the full set of activation functions.

    hidden_layer_size : list of int or tuple of int

        Sizes of all hidden layers.

        Should not be specified only if ``hidden_layer_size_options`` is provided.

    hidden_layer_size_options : list of tuples, optional

        A list of optional sizes of all hidden layers for parameter selection.

    max_iter : int, optional

        Maximum number of iterations.

        Defaults to 100.

    training_style : {'batch', 'stochastic'}, optional

        Specifies the training style.

        Defaults to 'stochastic'.

    learning_rate : float, optional

        Specifies the learning rate.
        Mandatory and valid only when ``training_style`` is 'stochastic'.

    momentum : float, optional

        Specifies the momentum for gradient descent update.
        Mandatory and valid only when ``training_style`` is 'stochastic'.

    batch_size : int, optional

        Specifies the size of mini batch.
        Valid only when ``training_style`` is 'stochastic'.

        Defaults to 1.

    normalization : {'no', 'z-transform', 'scalar'}, optional

        Defaults to 'no'.

    weight_init : {'all-zeros', 'normal', 'uniform', 'variance-scale-normal', 'variance-scale-uniform'}, optional

        Specifies the weight initial value.

        Defaults to 'all-zeros'.

    categorical_variable : str or list of str, optional

        Specifies column name(s) in the data table used as category variable.

        Valid only when column is of INTEGER type.

    thread_ratio : float, optional

        Controls the proportion of available threads to use for training.

        The value range is from 0 to 1, where 0 indicates a single thread,
        and 1 indicates up to all available threads.

        Values between 0 and 1
        will use that percentage of available threads.

        Values outside this range tell PAL to heuristically determine the number of threads to use.

        Defaults to 0.

    resampling_method : {'cv','stratified_cv', 'bootstrap', 'stratified_bootstrap'}, optional

        Specifies the resampling method for model evaluation or parameter
        selection.

        If not specified, neither model evaluation or parameter selection shall
        be triggered.

    evaluation_metric : {'accuracy','f1_score', 'auc_onevsrest', 'auc_pairwise'}, optional

        Specifies the evaluation metric for model evaluation or parameter selection.

    fold_num : int, optional

        Specifies the fold number for the cross-validation.

        Mandatory and valid only when ``resampling_method`` is set 'cv' or 'stratified_cv'.

    repeat_times : int, optional

        Specifies the number of repeat times for resampling.

        Defaults to 1.

    search_strategy : {'grid', 'random'}, optional

        Specifies the method for parameter selection.
        If not provided, parameter selection will not be activated.

    random_search_times : int, optional

        Specifies the number of times to randomly select candidate parameters.

        Mandatory and valid only when ``search_strategy`` is set to 'random'.

    random_state : int, optional

        Specifies the seed for random generation.

        When 0 is specified, system time is used.

        Defaults to 0.

    timeout : int, optional

        Specifies maximum running time for model evalation/parameter selection,
        in seconds.

        No timeout when 0 is specified.

        Defaults to 0.

    progress_id : str, optional

        Sets an ID of progress indicator for model evaluation/parameter selection.

        If not provided, no progress indicator is activated.

    param_values : dict or list of tuples, optional

        Specifies the values of following parameters for model parameter selection:

            ``learning_rate``, ``momentum``, ``batch_size``.

        If input is list of tuples, then each tuple must contain exactly two elements:

            - 1st element is the parameter name(str type),
            - 2nd element is a list of valid values for that parameter.

        If input is dict, then for each element, the key must be parameter name, while value
        be a list of valid values for the corresponding parameter.

        A simple example for illustration:

            [('learning_rate', [0.1, 0.2, 0.5]), ('momentum', [0.2, 0.6])],

        or

            dict(learning_rate=[0.1, 0.2, 0.5], momentum=[0.2, 0.6]).

        Valid only when ``resampling_method`` and ``search_strategy`` are both specified,
        and ``training_style`` is 'stochastic'.

    param_range : list of tuple, optional

        Specifies the range of the following parameters for model parameter selection:

            ``learning_rate``, ``momentum``, ``batch_size``.

        If input is a list of tuples, the each tuple should contain exactly two elements:

            - 1st element is the parameter name(str type),
            - 2nd element is a list that specifies the range of that parameter as follows:
              first value is the start value, seond value is the step, and third value is the end value.
              The step value can be omitted, and will be ignored, if ``search_strategy`` is set to 'random'.

        Otherwise, if input is a dict, then for each element the key should be parameter name, while value
        specifies the range of that parameter.

        Valid only when ``resampling_method`` and ``search_strategy`` are both specified,
        and ``traininig_style`` is 'stochastic'.

    Attributes
    ----------

    model_ : DataFrame

        Model content.

    train_log_ : DataFrame

        Provides mean squared error between predicted values and target
        values for each iteration.

    stats_ : DataFrame

        Names and values of statistics.

    optim_param_ : DataFrame

        Provides optimal parameters selected.

        Available only when parameter selection is triggered.


    Examples
    --------

    Training data:

    >>> df.collect()
       V000  V001 V002  V003 LABEL
    0     1  1.71   AC     0    AA
    1    10  1.78   CA     5    AB
    2    17  2.36   AA     6    AA
    3    12  3.15   AA     2     C
    4     7  1.05   CA     3    AB
    5     6  1.50   CA     2    AB
    6     9  1.97   CA     6     C
    7     5  1.26   AA     1    AA
    8    12  2.13   AC     4     C
    9    18  1.87   AC     6    AA

    Training the model:

    >>> mlpc = MLPClassifier(hidden_layer_size=(10,10),
    ...                      activation='tanh', output_activation='tanh',
    ...                      learning_rate=0.001, momentum=0.0001,
    ...                      training_style='stochastic',max_iter=100,
    ...                      normalization='z-transform', weight_init='normal',
    ...                      thread_ratio=0.3, categorical_variable='V003')
    >>> mlpc.fit(data=df)

    Training result may look different from the following results due
    to model randomness.

    >>> mlpc.model_.collect()
       ROW_INDEX                                      MODEL_CONTENT
    0          1  {"CurrentVersion":"1.0","DataDictionary":[{"da...
    1          2  t":0.2700182926188939},{"from":13,"weight":0.0...
    2          3  ht":0.2414416413305134},{"from":21,"weight":0....
    >>> mlpc.train_log_.collect()
        ITERATION     ERROR
    0           1  1.080261
    1           2  1.008358
    2           3  0.947069
    3           4  0.894585
    4           5  0.849411
    5           6  0.810309
    6           7  0.776256
    7           8  0.746413
    8           9  0.720093
    9          10  0.696737
    10         11  0.675886
    11         12  0.657166
    12         13  0.640270
    13         14  0.624943
    14         15  0.609432
    15         16  0.595204
    16         17  0.582101
    17         18  0.569990
    18         19  0.558757
    19         20  0.548305
    20         21  0.538553
    21         22  0.529429
    22         23  0.521457
    23         24  0.513893
    24         25  0.506704
    25         26  0.499861
    26         27  0.493338
    27         28  0.487111
    28         29  0.481159
    29         30  0.475462
    ..        ...       ...
    70         71  0.349684
    71         72  0.347798
    72         73  0.345954
    73         74  0.344071
    74         75  0.342232
    75         76  0.340597
    76         77  0.338837
    77         78  0.337236
    78         79  0.335749
    79         80  0.334296
    80         81  0.332759
    81         82  0.331255
    82         83  0.329810
    83         84  0.328367
    84         85  0.326952
    85         86  0.325566
    86         87  0.324232
    87         88  0.322899
    88         89  0.321593
    89         90  0.320242
    90         91  0.318985
    91         92  0.317840
    92         93  0.316630
    93         94  0.315376
    94         95  0.314210
    95         96  0.313066
    96         97  0.312021
    97         98  0.310916
    98         99  0.309770
    99        100  0.308704

    Prediction:

    >>> pred_df.collect()
    >>> res, stat = mlpc.predict(data=pred_df, key='ID')

    Prediction result may look different from the following results due to model randomness.

    >>> res.collect()
       ID TARGET     VALUE
    0   1      C  0.472751
    1   2      C  0.417681
    2   3      C  0.543967
    >>> stat.collect()
       ID CLASS  SOFT_MAX
    0   1    AA  0.371996
    1   1    AB  0.155253
    2   1     C  0.472751
    3   2    AA  0.357822
    4   2    AB  0.224496
    5   2     C  0.417681
    6   3    AA  0.349813
    7   3    AB  0.106220
    8   3     C  0.543967

    Model Evaluation:

    >>> mlpc = MLPClassifier(activation='tanh',
    ...                      output_activation='tanh',
    ...                      hidden_layer_size=(10,10),
    ...                      learning_rate=0.001,
    ...                      momentum=0.0001,
    ...                      training_style='stochastic',
    ...                      max_iter=100,
    ...                      normalization='z-transform',
    ...                      weight_init='normal',
    ...                      resampling_method='cv',
    ...                      evaluation_metric='f1_score',
    ...                      fold_num=10,
    ...                      repeat_times=2,
    ...                      random_state=1,
    ...                      progress_indicator_id='TEST',
    ...                      thread_ratio=0.3)
    >>> mlpc.fit(data=df, label='LABEL', categorical_variable='V003')

    Model evaluation result may look different from the following result due to randomness.

    >>> mlpc.stats_.collect()
                STAT_NAME                                         STAT_VALUE
    0             timeout                                              FALSE
    1     TEST_1_F1_SCORE                       1, 0, 1, 1, 0, 1, 0, 1, 1, 0
    2     TEST_2_F1_SCORE                       0, 0, 1, 1, 0, 1, 0, 1, 1, 1
    3  TEST_F1_SCORE.MEAN                                                0.6
    4   TEST_F1_SCORE.VAR                                           0.252631
    5      EVAL_RESULTS_1  {"candidates":[{"TEST_F1_SCORE":[[1.0,0.0,1.0,...
    6     solution status  Convergence not reached after maximum number o...
    7               ERROR                                 0.2951168443145714

    Parameter selection:

    >>> act_opts=['tanh', 'linear', 'sigmoid_asymmetric']
    >>> out_act_opts = ['sigmoid_symmetric', 'gaussian_asymmetric', 'gaussian_symmetric']
    >>> layer_size_opts = [(10, 10), (5, 5, 5)]
    >>> mlpc = MLPClassifier(activation_options=act_opts,
    ...                      output_activation_options=out_act_opts,
    ...                      hidden_layer_size_options=layer_size_opts,
    ...                      learning_rate=0.001,
    ...                      batch_size=2,
    ...                      momentum=0.0001,
    ...                      training_style='stochastic',
    ...                      max_iter=100,
    ...                      normalization='z-transform',
    ...                      weight_init='normal',
    ...                      resampling_method='stratified_bootstrap',
    ...                      evaluation_metric='accuracy',
    ...                      search_strategy='grid',
    ...                      fold_num=10,
    ...                      repeat_times=2,
    ...                      random_state=1,
    ...                      progress_indicator_id='TEST',
    ...                      thread_ratio=0.3)
    >>> mlpc.fit(data=df, label='LABEL', categorical_variable='V003')

    Parameter selection result may look different from the following result due to randomness.

    >>> mlpc.stats_.collect()
                STAT_NAME                                         STAT_VALUE
    0             timeout                                              FALSE
    1     TEST_1_ACCURACY                                               0.25
    2     TEST_2_ACCURACY                                           0.666666
    3  TEST_ACCURACY.MEAN                                           0.458333
    4   TEST_ACCURACY.VAR                                          0.0868055
    5      EVAL_RESULTS_1  {"candidates":[{"TEST_ACCURACY":[[0.50],[0.0]]...
    6      EVAL_RESULTS_2  PUT_LAYER_ACTIVE_FUNC=6;HIDDEN_LAYER_ACTIVE_FU...
    7      EVAL_RESULTS_3  FUNC=2;"},{"TEST_ACCURACY":[[0.50],[0.33333333...
    8      EVAL_RESULTS_4  rs":"HIDDEN_LAYER_SIZE=10, 10;OUTPUT_LAYER_ACT...
    9               ERROR                                  0.684842661926971
    >>> mlpc.optim_param_.collect()
                     PARAM_NAME  INT_VALUE DOUBLE_VALUE STRING_VALUE
    0         HIDDEN_LAYER_SIZE        NaN         None      5, 5, 5
    1  OUTPUT_LAYER_ACTIVE_FUNC        4.0         None         None
    2  HIDDEN_LAYER_ACTIVE_FUNC        3.0         None         None
    """

    #pylint: disable=too-many-arguments, too-many-locals
    def __init__(self, activation=None, activation_options=None,
                 output_activation=None, output_activation_options=None,
                 hidden_layer_size=None, hidden_layer_size_options=None,
                 max_iter=None, training_style='stochastic', learning_rate=None, momentum=None,
                 batch_size=None, normalization=None, weight_init=None, categorical_variable=None,
                 resampling_method=None, evaluation_metric=None, fold_num=None,
                 repeat_times=None, search_strategy=None, random_search_times=None,
                 random_state=None, timeout=None, progress_indicator_id=None,
                 param_values=None, param_range=None, thread_ratio=None):
        super(MLPClassifier, self).__init__(activation=activation,
                                            activation_options=activation_options,
                                            output_activation=output_activation,
                                            output_activation_options=output_activation_options,
                                            hidden_layer_size=hidden_layer_size,
                                            hidden_layer_size_options=hidden_layer_size_options,
                                            max_iter=max_iter,
                                            functionality=0,
                                            training_style=training_style,
                                            learning_rate=learning_rate,
                                            momentum=momentum,
                                            batch_size=batch_size,
                                            normalization=normalization,
                                            weight_init=weight_init,
                                            categorical_variable=categorical_variable,
                                            resampling_method=resampling_method,
                                            evaluation_metric=evaluation_metric,
                                            fold_num=fold_num,
                                            repeat_times=repeat_times,
                                            search_strategy=search_strategy,
                                            random_search_times=random_search_times,
                                            random_state=random_state,
                                            timeout=timeout,
                                            progress_indicator_id=progress_indicator_id,
                                            param_values=param_values,
                                            param_range=param_range,
                                            thread_ratio=thread_ratio)

    def fit(self, data, key=None, features=None, label=None, categorical_variable=None):
        """
        Fit the model when the training dataset is given.

        Parameters
        ----------

        data : DataFrame

            DataFrame containing the data.

        key : str, optional

            Name of the ID column. If ``key`` is not provided, it is assumed
            that the input has no ID column.

        features : list of str, optional

            Names of the feature columns.
            If ``features`` is not provided, it defaults to all
            the non-ID and non-label columns.

        label : str, optional

            Name of the label column.
            If ``label`` is not provided, it defaults to the last column.

        categorical_variable : str or list of str, optional

            Specifies INTEGER column(s) specified that should be treated as categorical.
            Other INTEGER columns will be treated as continuous.
        """
        key = self._arg('key', key, str)
        features = self._arg('features', features, ListOfStrings)
        label = self._arg('label', label, str)
        return self._fit(data, key, features, label, categorical_variable)

    def predict(self, data, key, features=None, thread_ratio=None):
        """
        Predict using the multi-layer perceptron model.

        Parameters
        ----------

        data : DataFrame

            DataFrame containing the data.

        key : str

            Name of the ID column.

        features : list of str, optional

            Names of the feature columns.
            If ``features`` is not provided, it defaults to all
            the non-ID columns.

        thread_ratio : float, optional

            Controls the proportion of available threads to be used for prediction.

            The value range is from 0 to 1, where 0 indicates a single thread,
            and 1 indicates up to all available threads.

            Values between 0 and 1 will use that percentage of available threads.

            Values outside this range tell PAL to heuristically determine the number of threads to use.

            Defaults to 0.

        Returns
        -------

        DataFrame

            Predicted classes, structured as follows:

              - ID column, with the same name and type as ``data`` 's ID column.
              - TARGET, type NVARCHAR, predicted class name.
              - VALUE, type DOUBLE, softmax value for the predicted class.

            Softmax values for all classes, structured as follows:

              - ID column, with the same name and type as ``data`` 's ID column.
              - CLASS, type NVARCHAR, class name.
              - VALUE, type DOUBLE, softmax value for that class.
        """
        return super(MLPClassifier, self)._predict(data, key, features, thread_ratio)

    def score(self, data, key, features=None, label=None, thread_ratio=None):
        """
        Returns the accuracy on the given test data and labels.

        Parameters
        ----------

        data : DataFrame

            DataFrame containing the data.

        key : str

            Name of the ID column.

        features : list of str, optional

            Names of the feature columns.

            If ``features`` is not provided, it defaults to all
            the non-ID and non-label columns.

        label : str, optional

            Name of the label column.

            If ``label`` is not provided, it defaults to the last column.

        Returns
        -------

        float

            Scalar value of accuracy after comparing the predicted result
            and original label.
        """
        if not hasattr(self, 'model_'):
            raise FitIncompleteError("Model not initialized. Perform a fit first.")
        key = self._arg('key', key, str, True)
        features = self._arg('features', features, ListOfStrings)
        label = self._arg('label', label, str)
        unique_id = str(uuid.uuid1())
        unique_id = unique_id.replace('-', '_').upper()
        cols = data.columns
        cols.remove(key)
        if label is None:
            label = cols[-1]
        cols.remove(label)
        if features is None:
            features = cols

        prediction, _ = self.predict(data=data, key=key,
                                     features=features, thread_ratio=thread_ratio)
        prediction = prediction.select(key, 'TARGET').rename_columns(['ID_P', 'PREDICTION'])

        actual = data.select(key, label).rename_columns(['ID_A', 'ACTUAL'])
        joined = actual.join(prediction, 'ID_P=ID_A').select('ACTUAL', 'PREDICTION')

        return metrics.accuracy_score(joined,
                                      label_true='ACTUAL',
                                      label_pred='PREDICTION')


class MLPRegressor(_MLPBase):
    '''
    Multi-layer perceptron (MLP) Regressor.

    Parameters
    ----------
    activation : str

        Specifies the activation function for the hidden layer.

        Valid activation functions include:
          - 'tanh',
          - 'linear',
          - 'sigmoid_asymmetric',
          - 'sigmoid_symmetric',
          - 'gaussian_asymmetric',
          - 'gaussian_symmetric',
          - 'elliot_asymmetric',
          - 'elliot_symmetric',
          - 'sin_asymmetric',
          - 'sin_symmetric',
          - 'cos_asymmetric',
          - 'cos_symmetric',
          - 'relu'

        Should not be specified only if ``activation_options`` is provided.

    activation_options : list of str, optional

        A list of activation functions for parameter selection.

        See ``activation`` for the full set of valid activation functions.

    output_activation : str

        Specifies the activation function for the output layer.

        Valid choices of activation function same as  those in ``activation``.

        Should not be specified only if ``output_activation_options`` is provided.

    output_activation_options : list of str, conditionally mandatory

        A list of activation functions for the output layer for parameter selection.

        See ``activation`` for the full set of activation functions for output layer.

    hidden_layer_size : list of int or tuple of int

        Sizes of all hidden layers.

        Should not be specified only if ``hidden_layer_size_options`` is provided.

    hidden_layer_size_options : list of tuples, optional

        A list of optional sizes of all hidden layers for parameter selection.

    max_iter : int, optional

        Maximum number of iterations.

        Defaults to 100.

    training_style :  {'batch', 'stochastic'}, optional

        Specifies the training style.

        Defaults to 'stochastic'.

    learning_rate : float, optional

        Specifies the learning rate.

        Mandatory and valid only when ``training_style`` is 'stochastic'.

    momentum : float, optional

        Specifies the momentum for gradient descent update.

        Mandatory and valid only when ``training_style`` is 'stochastic'.

    batch_size : int, optional

        Specifies the size of mini batch.

        Valid only when ``training_style`` is 'stochastic'.

        Defaults to 1.

    normalization : {'no', 'z-transform', 'scalar'}, optional

        Defaults to 'no'.

    weight_init : {'all-zeros', 'normal', 'uniform', 'variance-scale-normal', 'variance-scale-uniform'}, optional

        Specifies the weight initial value.

        Defaults to 'all-zeros'.

    categorical_variable : str or list of str, optional

        Specifies column name(s) in the data table used as category variable.

        Valid only when column is of INTEGER type.

    thread_ratio : float, optional

        Controls the proportion of available threads to use for training.

        The value range is from 0 to 1, where 0 indicates a single thread,
        and 1 indicates up to all available threads.

        Values between 0 and 1 will use that percentage of available threads.

        Values outside this range tell PAL to heuristically determine the number of threads to use.

        Defaults to 0.

    resampling_method : {'cv', 'bootstrap'}, optional

        Specifies the resampling method for model evaluation or parameter selection.

        If not specified, neither model evaluation or parameter selection shall
        be triggered.

    evaluation_metric : {'rmse'}, optional

        Specifies the evaluation metric for model evaluation or parameter selection.

    fold_num : int, optional

        Specifies the fold number for the cross-validation.

        Mandatory and valid only when ``resampling_method`` is set 'cv'.

    repeat_times : int, optional

        Specifies the number of repeat times for resampling.

        Defaults to 1.

    search_strategy : {'grid', 'random'}, optional

        Specifies the method for parameter selection.

        If not provided, parameter selection will not be activated.

    random_searhc_times : int, optional

        Specifies the number of times to randomly select candidate parameters.

        Mandatory and valid only when ``search_strategy`` is set to 'random'.

    random_state : int, optional

        Specifies the seed for random generation.

        When 0 is specified, system time is used.

        Defaults to 0.

    timeout : int, optional

        Specifies maximum running time for model evalation/parameter selection,
        in seconds.

        No timeout when 0 is specified.

        Defaults to 0.

    progress_id : str, optional

        Sets an ID of progress indicator for model evaluation/parameter selection.

        If not provided, no progress indicator is activated.

    param_values : dict or list of tuples, optional

        Specifies the values of following parameters for model parameter selection:

            ``learning_rate``, ``momentum``, ``batch_size``.

        If input is list of tuples, then each tuple must contain exactly two elements:

            - 1st element is the parameter name(str type),
            - 2nd element is a list of valid values for that parameter.

        Otherwise, if input is dict, then for each element, the key must be a parameter name, while value
        be a list of valid values for that parameter.

        A simple example for illustration:

            [('learning_rate', [0.1, 0.2, 0.5]), ('momentum', [0.2, 0.6])],

        or

		    dict(learning_rate=[0.1, 0.2, 0.5], momentum=[0.2, 0.6]).

        Valid only when ``resampling_method`` and ``search_strategy`` are both specified,
        and ``training_style`` is 'stochastic'.

    param_range : dict or list of tuple, optional

        Sets the range of the following parameters for model parameter selection:

            ``learning_rate``, ``momentum``, ``batch_size``.

        If input is a list of tuples, the each tuple should contain exactly two elements:

            - 1st element is the parameter name(str type),
            - 2nd element is a list that specifies the range of that parameter as follows:
              first value is the start value, seond value is the step, and third value is the end value.
              The step value can be omitted, and will be ignored, if ``search_strategy`` is set to 'random'.

        Otherwise, if input is a dict, then for each element the key should be parameter name, while value
        specifies the range of that parameter.

        Valid only when ``resampling_method`` and ``search_strategy`` are both specified,
        and ``traininig_style`` is 'stochastic'.

    Attributes
    ----------

    model_ : DataFrame

        Model content.

    train_log_ : DataFrame

        Provides mean squared error between predicted values and target
        values for each iteration.

    stats_ : DataFrame

        Names and values of statistics.

    optim_param_ : DataFrame

        Provides optimal parameters selected.

        Available only when parameter selection is triggered.

    Examples
    --------

    Training data:

    >>> df.collect()
       V000  V001 V002  V003  T001  T002  T003
    0     1  1.71   AC     0  12.7   2.8  3.06
    1    10  1.78   CA     5  12.1   8.0  2.65
    2    17  2.36   AA     6  10.1   2.8  3.24
    3    12  3.15   AA     2  28.1   5.6  2.24
    4     7  1.05   CA     3  19.8   7.1  1.98
    5     6  1.50   CA     2  23.2   4.9  2.12
    6     9  1.97   CA     6  24.5   4.2  1.05
    7     5  1.26   AA     1  13.6   5.1  2.78
    8    12  2.13   AC     4  13.2   1.9  1.34
    9    18  1.87   AC     6  25.5   3.6  2.14

    Training the model:

    >>> mlpr = MLPRegressor(hidden_layer_size=(10,5),
    ...                     activation='sin_asymmetric',
    ...                     output_activation='sin_asymmetric',
    ...                     learning_rate=0.001, momentum=0.00001,
    ...                     training_style='batch',
    ...                     max_iter=10000, normalization='z-transform',
    ...                     weight_init='normal', thread_ratio=0.3)
    >>> mlpr.fit(data=df, label=['T001', 'T002', 'T003'])

    Training result may look different from the following results due
    to model randomness.

    >>> mlpr.model_.collect()
       ROW_INDEX                                      MODEL_CONTENT
    0          1  {"CurrentVersion":"1.0","DataDictionary":[{"da...
    1          2  3782583596893},{"from":10,"weight":-0.16532599...
    >>> mlpr.train_log_.collect()
         ITERATION       ERROR
    0            1   34.525655
    1            2   82.656301
    2            3   67.289241
    3            4  162.768062
    4            5   38.988242
    5            6  142.239468
    6            7   34.467742
    7            8   31.050946
    8            9   30.863581
    9           10   30.078204
    10          11   26.671436
    11          12   28.078312
    12          13   27.243226
    13          14   26.916686
    14          15   26.782915
    15          16   26.724266
    16          17   26.697108
    17          18   26.684084
    18          19   26.677713
    19          20   26.674563
    20          21   26.672997
    21          22   26.672216
    22          23   26.671826
    23          24   26.671631
    24          25   26.671533
    25          26   26.671485
    26          27   26.671460
    27          28   26.671448
    28          29   26.671442
    29          30   26.671439
    ..         ...         ...
    705        706   11.891081
    706        707   11.891081
    707        708   11.891081
    708        709   11.891081
    709        710   11.891081
    710        711   11.891081
    711        712   11.891081
    712        713   11.891081
    713        714   11.891081
    714        715   11.891081
    715        716   11.891081
    716        717   11.891081
    717        718   11.891081
    718        719   11.891081
    719        720   11.891081
    720        721   11.891081
    721        722   11.891081
    722        723   11.891081
    723        724   11.891081
    724        725   11.891081
    725        726   11.891081
    726        727   11.891081
    727        728   11.891081
    728        729   11.891081
    729        730   11.891081
    730        731   11.891081
    731        732   11.891081
    732        733   11.891081
    733        734   11.891081
    734        735   11.891081

    [735 rows x 2 columns]

    >>> pred_df.collect()
       ID  V000  V001 V002  V003
    0   1     1  1.71   AC     0
    1   2    10  1.78   CA     5
    2   3    17  2.36   AA     6

    Prediction:

    >>> res  = mlpr.predict(data=pred_df, key='ID')

    Result may look different from the following results due to model
    randomness.

    >>> res.collect()
       ID TARGET      VALUE
    0   1   T001  12.700012
    1   1   T002   2.799133
    2   1   T003   2.190000
    3   2   T001  12.099740
    4   2   T002   6.100000
    5   2   T003   2.190000
    6   3   T001  10.099961
    7   3   T002   2.799659
    8   3   T003   2.190000
    '''
    #pylint:disable=too-many-arguments, too-many-locals
    def __init__(self, activation=None, activation_options=None,
                 output_activation=None, output_activation_options=None,
                 hidden_layer_size=None, hidden_layer_size_options=None,
                 max_iter=None, training_style='stochastic', learning_rate=None, momentum=None,
                 batch_size=None, normalization=None, weight_init=None, categorical_variable=None,
                 resampling_method=None, evaluation_metric=None, fold_num=None,
                 repeat_times=None, search_strategy=None, random_search_times=None,
                 random_state=None, timeout=None, progress_indicator_id=None,
                 param_values=None, param_range=None, thread_ratio=None):
        super(MLPRegressor, self).__init__(activation=activation,
                                           activation_options=activation_options,
                                           output_activation=output_activation,
                                           output_activation_options=output_activation_options,
                                           hidden_layer_size=hidden_layer_size,
                                           hidden_layer_size_options=hidden_layer_size_options,
                                           max_iter=max_iter,
                                           functionality=1,
                                           training_style=training_style,
                                           learning_rate=learning_rate,
                                           momentum=momentum,
                                           batch_size=batch_size,
                                           normalization=normalization,
                                           weight_init=weight_init,
                                           categorical_variable=categorical_variable,
                                           resampling_method=resampling_method,
                                           evaluation_metric=evaluation_metric,
                                           fold_num=fold_num,
                                           repeat_times=repeat_times,
                                           search_strategy=search_strategy,
                                           random_search_times=random_search_times,
                                           random_state=random_state,
                                           timeout=timeout,
                                           progress_indicator_id=progress_indicator_id,
                                           param_values=param_values,
                                           param_range=param_range,
                                           thread_ratio=thread_ratio)

    def fit(self, data, key=None, features=None, label=None, categorical_variable=None):
        """
        Fit the model when given training dataset.

        Parameters
        ----------

        data : DataFrame

            DataFrame containing the data.

        key : str, optional

            Name of the ID column. If ``key`` is not provided, it is assume
            that the input has no ID column.

        features : list of str, optional

            Names of the feature columns.

            If ``features`` is not provided, it defaults to all
            the non-ID and non-label columns.

        label : str or list of str, optional

            Name of the label column, or list of names of multiple label
            columns.

            If ``label`` is not provided, it defaults to the last column.

        categorical_variable : str or list of str, optional

            Specifies INTEGER column(s) that should be treated as categorical.

            Other INTEGER columns will be treated as continuous.
        """

        key = self._arg('key', key, str)
        features = self._arg('features', features, ListOfStrings)
        #check for str or list of str:
        if label is not None:
            msg = "label should be a string or list of strings."
            if isinstance(label, list):
                if not all(isinstance(x, _TEXT_TYPES) for x in label):
                    logger.error(msg)
                    raise TypeError(msg)
            else:
                if not isinstance(label, _TEXT_TYPES):
                    logger.error(msg)
                    raise ValueError(msg)
        return self._fit(data, key, features, label, categorical_variable)

    def predict(self, data, key, features=None, thread_ratio=None):
        """
        Predict using the multi-layer perceptron model.

        Parameters
        ----------

        data : DataFrame

            DataFrame containing the data.

        key : str

            Name of the ID column.

        features : list of str, optional

            Names of the feature columns.
            If ``features`` is not provided, it defaults to all
            the non-ID columns.

        thread_ratio : float, optional

            Controls the proportion of available threads to be used for prediction.

            The value range is from 0 to 1, where 0 indicates a single thread,
            and 1 indicates up to all available threads.

            Values between 0 and 1 will use that percentage of available threads.

            Values outside this range tell PAL to heuristically determine the number of threads to use.

            Defaults to 0.

        Returns
        -------

        DataFrame

            Predicted results, structured as follows:

                - ID column, with the same name and type as ``data`` 's ID column.
                - TARGET, type NVARCHAR, target name.
                - VALUE, type DOUBLE, regression value.
        """
        pred_res, _ = super(MLPRegressor, self)._predict(data, key, features, thread_ratio)
        return pred_res

    def score(self, data, key, features=None, label=None, thread_ratio=None):#pylint: disable=too-many-locals
        #check for fit table existence
        """
        Returns the coefficient of determination R^2 of the prediction.

        Parameters
        ----------

        data : DataFrame

            DataFrame containing the data.

        key : str

            Name of the ID column.

        features : list of str, optional

            Names of the feature columns.

            If ``features`` is not provided, it defaults to all
            the non-ID and non-label columns.

        label : str or list of str, optional

            Name of the label column, or list of names of multiple label
            columns.

            If ``label`` is not provided, it defaults to the last column.

        Returns
        -------

        float

            Returns the coefficient of determination R2 of the prediction.
        """
        conn = data.connection_context
        require_pal_usable(conn)
        if not hasattr(self, 'model_'):
            raise FitIncompleteError("Model not initialized. Perform a fit first.")
        key = self._arg('key', key, str, True)
        features = self._arg('features', features, ListOfStrings)
        #check for str or list of str:
        if label is not None:
            msg = "label should be a string or list of strings."
            if isinstance(label, list):
                if not all(isinstance(x, _TEXT_TYPES) for x in label):
                    logger.error(msg)
                    raise ValueError(msg)
            else:
                if not isinstance(label, _TEXT_TYPES):
                    logger.error(msg)
                    raise ValueError(msg)
        #return scalar value of accuracy after calling predict
        unique_id = str(uuid.uuid1())
        unique_id = unique_id.replace('-', '_').upper()
        cols = data.columns
        cols.remove(key)
        mlp_type = 'MLPREGRESSOR'
        if label is None:
            label = cols[-1]
        if isinstance(label, _TEXT_TYPES):
            label = [label]
        if features is None:
            features = [x for x in cols if x not in label]
        val_df = data.select([key] + label)
        pred_res = self.predict(data, key, features, thread_ratio)
        ##create compare table with ID, original val, predicted_val
        comp_tbl = '#{}_COMPARE_TBL_{}_{}'.format(mlp_type, self.id, unique_id)
        try:
            with conn.connection.cursor() as cur:
                #reorganize pred_res like (ID, COL1, COL2...) instead of ID, COL_NAME, COL_VALUE
                temp_sql = [('MAX(CASE WHEN ("TARGET" = {0})' +
                             'THEN "VALUE" ELSE NULL END) ' +
                             'AS {1}').format("N'{}'".format(name.replace("'", "''")),
                                              quotename(name))
                            for name in label]
                temp_sql = ", ".join(temp_sql)
                pred_res_new_sql = ('SELECT {0}, {1} FROM ({2}) ' +
                                    'GROUP BY {0} ORDER BY {0}').format(quotename(key),
                                                                        temp_sql,
                                                                        pred_res.select_statement)
                comp_cols = ['ori.{0} as {1}, pred.{0} as {2}'.format(quotename(name),
                                                                      quotename('ORI_' + name),
                                                                      quotename('PRED_' + name))
                             for name in label]
                comp_cols = ', '.join(comp_cols)
                comp_tbl_sql = ('CREATE LOCAL TEMPORARY COLUMN TABLE {0} ' +
                                'AS (SELECT ori.{1}, {2} FROM'
                                ' ({3}) ori,' +
                                ' ({4}) AS pred WHERE ' +
                                'ori.{1} = ' +
                                'pred.{1});').format(quotename(comp_tbl),
                                                     quotename(key),
                                                     comp_cols,
                                                     val_df.select_statement,
                                                     pred_res_new_sql)
                execute_logged(cur, comp_tbl_sql)
                #construct sql for calculating U => ((y_true - y_pred) ** 2).sum()
                u_sql = ['SUM(POWER({0} - {1}, 2))'.format(quotename('ORI_' + name),
                                                           quotename('PRED_' + name))
                         for name in label]
                u_sql = ' + '.join(u_sql)
                #construct sql for calculating V => ((y_true - y_true.mean()) ** 2).sum()
                v_sql = [('SUM(POWER({0} - (SELECT AVG({0}) FROM {1}),' +
                          '2))').format(quotename('ORI_' + name),
                                        quotename(comp_tbl))
                         for name in label]
                v_sql = ' + '.join(v_sql)
                #construct sql for calculating R2 => 1 - U/V
                res_sql = ('SELECT 1- ({0}) / ({1}) FROM {2};').format(u_sql,
                                                                       v_sql,
                                                                       quotename(comp_tbl))
                execute_logged(cur, res_sql)
                r_2 = cur.fetchall()
                return r_2[0][0]
        except dbapi.Error as db_err:
            #logger.error('HANA error during MLPRegressor score.', exc_info=True)
            logger.error(str(db_err))
            raise
