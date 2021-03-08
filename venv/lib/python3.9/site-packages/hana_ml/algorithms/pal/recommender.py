#pylint:disable=too-many-lines
#pylint:disable=line-too-long
"""
This module contains Python API of PAL recommender system algorithms.
The following classes are available:

* :class:`ALS`
* :class:`FRM`
* :class:`FFMClassifier`
* :class:`FFMRegressor`
* :class:`FFMRanker`

"""
import logging
import sys
import uuid
from hdbcli import dbapi
from hana_ml.dataframe import quotename
from hana_ml.ml_exceptions import FitIncompleteError
from hana_ml.algorithms.pal.pal_base import (
    PALBase,
    _INT_TYPES,
    ParameterTable,
    ListOfTuples,
    ListOfStrings,
    pal_param_register,
    try_drop,
    require_pal_usable,
    call_pal_auto
)
logger = logging.getLogger(__name__)#pylint:disable=invalid-name
_INTEGER_TYPES = (int, long) if(sys.version_info.major == 2) else(int,)#pylint: disable=undefined-variable
_STRING_TYPES = (str, unicode) if(sys.version_info.major == 2) else(str,)#pylint: disable=undefined-variable

class ALS(PALBase):#pylint:disable=too-many-arguments, too-few-public-methods, too-many-instance-attributes
    """
    Class for recommender system, alternating least squares algorithm.

    Parameters
    ----------

    factor_num : int, optional
        Length of factor vectors in the model.

        Default to 8.
    random_state : int, optional
        Specifies the seed for random number generator.

          - 0: Uses the current time as the seed.
          - Others: Uses the specified value as the seed.

        Default to 0.
    lamb : float, optional
        Specifies the L2 regularization of the factors.

        Default to 1e-2
    thread_ratio : float, optional
        Controls the proportion of available threads that can be used.

        The value range is from 0 to 1, where 0 indicates a single thread,
        and 1 indicates up to all available threads.

		Values between 0 and 1 will use that percentage of available threads.

        Values outside this range tell PAL to heuristically determine the number of threads to use.

        Defaults to 0.
    max_iter : int, optional
        Specifies the maximum number of iterations for the ALS algorithm.

        Default to 20.
    tol : float, optional
        Specifies the tolerance for exiting the iterative algorithm.

        The algorithm exits if the value of cost function is not decreased more
        than this value since the last check.

        If ``tol`` is set to 0, there is no check, and the algorithm only exits on reaching the maximum number of iterations.

        Note that evaluations of cost function require additional calculations, and you can set this parameter to 0 to avoid it.

        Default to 0.
    exit_interval : int, optional
        Specifies the number of iterations between consective convergence checkings.

        Basically, the algorithm calculates cost function and checks every ``exit_interval`` iterations
        to see if the tolerance has been reached.

        Note that evaluations of cost function require additional calculations.

        Only valid when ``tol`` is not 0.

        Default to 5.
    implicit : bool, optional
        Specifies implicit/explicit ALS.

        Default to False.
    linear_solver : {'cholesky', 'cg'}, optional
        Specifies the linear system solver.

        Default to 'cholesky'.
    cg_max_iter : int, optional
        Specifies maximum number of iteration of cg solver.

        Only valid when ``linear_solver`` is specified.

        Default to 3.
    alpha : float, optional
        Used when computing the confidence level in implicit ALS.

        Only valid when ``implicit`` is set to True.

        Default to 1.0.
    resampling_method : {'cv', 'boostrap'}, optional
        Specifies the resampling method for model evaluation or parameter selection.

        If not specified, neither model evaluation nor parameters selection is activated.

        No default value.
    evaluation_metric : {'rmse'}, optional
        Specifies the evaluation metric for model evaluation or parameter selection.

        If not specified, neither model evaluation nor parameter selection is activated.

        No default value.
    fold_num : int, optional
        Specifies the fold number for the cross validation method.

        Mandatory and valid only when ``resampling_method`` is set as 'cv'.

        Default to 1.
    repeat_times : int, optional
        Specifies the number of repeat times for resampling.

        Default to 1.
    search_strategy : {'grid', 'random'}, optional
        Specifies the method to activate parameter selection.

        No default value.
    random_search_times : int, optional
        Specifies the number of times to randomly select candidate parameters for selection.

        Mandatory and valid when ``search_strategy`` is set as 'random'.

        No default value.
    timeout : int, optional
        Specifies maximum running time for model evaluation or parameter selection, in seconds.

        No timeout when 0 is specified.

        Default to 0.
    progress_indicator_id : str, optional
        Sets an ID of progress indicator for model evaluation or parameter selection.

        No progress indicator is active if no value is provided.

        No default value.
    param_values : dict or ListOfTuples, optional

        Specifies values of parameters to be selected.

        Input should be a dict or list of size-two tuples, with key/1st element of each tuple being the target parameter name,
        while value/2nd element being the a list of valued for selection.

        Valid only when ``resampling_method`` and ``search_strategy`` are both specified.

        Valid parameter names include : ``alpha``, ``factor_num``, ``lamb``.

        No default value.

    param_range : dict or ListOfTuples, optional

        Specifies ranges of parameters to be selected.

        Input should be a dict or list of size-two tuples, with key/1st element of each tuple being the name of the target parameter,
        and value/2nd element being a list that specifies the range of parameters with the following format:

          [start, step, end] or [start, end].

        Valid only Only when `resampling_method` and `search_strategy` are both specified.

        Valid parameter names include : ``alpha``, ``factor_num``, ``lamb``.

        No default value.

    Attributes
    ----------
    metadata_ : DataFrame
        Model metadata content.
    map_ : DataFrame
        Map info.
    factors_ : DataFrame
        Decomposed factors.
    optim_param_ : DataFrame
        Optimal parameters selected.
    stats_ : DataFrame
        Statistic values.
    iter_info_ : DataFrame
        Cost function value and RMSE of corresponding iterations.

    Examples
    --------
    Input dataframe for training:

    >>> df_train.collect()
        USER    MOVIE    FEEDBACK
         A      Movie1      4.8
         A      Movie2      4.0
         A      Movie4      4.0
         A      Movie5      4.0
         A      Movie6      4.8
         A      Movie8      3.8
         A      Bad_Movie   2.5
         B      Movie2      4.8
         B      Movie3      4.8
         B      Movie4      5.0
         B      Movie5      5.0
         B      Movie7      3.5
         B      Movie8      4.8
         B      Bad_Movie   2.8
         C      Movie1      4.1
         C      Movie2      4.2
         C      Movie4      4.2
         C      Movie5      4.0
         C      Movie6      4.2
         C      Movie7      3.2
         C      Movie8      3.0
         C      Bad_Movie   2.5
         D      Movie1      4.5
         D      Movie3      3.5
         D      Movie4      4.5
         D      Movie6      3.9
         D      Movie7      3.5
         D      Movie8      3.5
         D      Bad_Movie   2.5
         E      Movie1      4.5
         E      Movie2      4.0
         E      Movie3      3.5
         E      Movie4      4.5
         E      Movie5      4.5
         E      Movie6      4.2
         E      Movie7      3.5
         E      Movie8      3.5

    Creating ALS instance:

    >>> als = ALS(factor_num=2,
                  lamb=1e-2, max_iter=20, tol=1e-6,
                  exit_interval=5, linear_solver='cholesky', thread_ratio=0, random_state=1)

    Performing fit() on given dataframe:

    >>> als.fit(self.df_train)

    >>> als.factors_.collect().head(10)
             FACTOR_ID    FACTOR
        0           0  1.108775
        1           1 -0.582392
        2           2  1.355926
        3           3 -0.760969
        4           4  1.084126
        5           5  0.281749
        6           6  1.145244
        7           7  0.418631
        8           8  1.151257
        9           9  0.315342

    Performing predict() on given predicting dataframe:

    >>> res = als.predict(self.df_predict, thread_ratio=1, key='ID')

    >>> res.collect()
               ID USER      MOVIE  PREDICTION
        0   1    A         Movie3    3.868747
        1   2    A         Movie7    2.870243
        2   3    B         Movie1    5.787559
        3   4    B         Movie6    5.837218
        4   5    C         Movie3    3.323575
        5   6    D         Movie2    4.156372
        6   7    D         Movie5    4.325851
        7   8    E      Bad_Movie    2.545807
    """
    resampling_method_list = {'cv' : 0, 'bootstrap' : 1}
    evaluation_metric_list = {'rmse'}
    search_strat_list = {'grid': 0, 'random': 1}
    range_params_map = {'factor_num' : 'FACTOR_NUMBER',
                        'lamb' : 'REGULARIZATION',
                        'alpha' : 'ALPHA'}
    linear_solver_map = {'choleskey': 0, 'cg': 1}
    def __init__(self,#pylint: disable=too-many-arguments, too-many-locals, too-many-branches, too-many-statements
                 random_state=None,
                 max_iter=None,
                 tol=None,
                 exit_interval=None,
                 implicit=None,
                 linear_solver=None,
                 cg_max_iter=None,
                 thread_ratio=None,
                 resampling_method=None,
                 evaluation_metric=None,
                 fold_num=None,
                 repeat_times=None,
                 search_strategy=None,
                 random_search_times=None,
                 timeout=None,
                 progress_indicator_id=None,
                 param_values=None,
                 param_range=None,
                 factor_num=None,
                 lamb=None,
                 alpha=None):
        super(ALS, self).__init__()
        setattr(self, 'hanaml_parameters', pal_param_register())
        self.factor_num = self._arg('factor_num', factor_num, int)
        self.random_state = self._arg('random_state', random_state, int)
        self.lamb = self._arg('lamb', lamb, float)
        self.max_iter = self._arg('max_iter', max_iter, int)
        self.tol = self._arg('tol', tol, float)
        self.exit_interval = self._arg('exit_interval', exit_interval, int)
        #if self.tol is not None and tol == 0:
        #    if exit_interval is not None:
        #        msg = ("`exit_interval` should only be valid if tolerance is not set to 0.")
        #        raise ValueError(msg)
        self.implicit = self._arg('implicit', implicit, int)
        self.linear_solver = self._arg('linear_solver', linear_solver, self.linear_solver_map)
        #if self.linear is not None:
        #    if self.linear not in :
        #        msg = ("Linear solver '{}' is not available in ALS.".format(self.linear))
        #        logger.error(msg)
        #        raise ValueError(msg)
        self.cg_max_iter = self._arg('cg_max_iter', cg_max_iter, int)
        #if self.linear_solver != 1:
        #    if cg_max_iter is not None:
        #        msg = ("`cg_max_iter` should only be valid if `linear_solver` is set as 'cg'.")
        #        raise ValueError(msg)
        self.alpha = self._arg('alpha', alpha, float)
        #if self.implicit is not True:
        #    if alpha is not None:
        #        msg = ("`alpha` should only be valid if `implicit` is set as True.")
        #        raise ValueError(msg)
        self.thread_ratio = self._arg('thread_ratio', thread_ratio, float)
        self.resampling_method = self._arg('resampling_method', resampling_method, str)
        if self.resampling_method is not None:
            if self.resampling_method not in self.resampling_method_list:#pylint:disable=line-too-long, bad-option-value
                msg = ("Resampling method '{}' is not available ".format(self.resampling_method)+
                       "for model evaluation/parameter selection in ALS.")
                logger.error(msg)
                raise ValueError(msg)
        self.evaluation_metric = self._arg('evaluation_metric', evaluation_metric, str)
        if self.evaluation_metric is not None:
            if self.evaluation_metric.lower() not in self.evaluation_metric_list:
                msg = ("Evaluation metric '{}' is not available.".format(self.evaluation_metric))
                logger.error(msg)
                raise ValueError(msg)
        self.fold_num = self._arg('fold_num', fold_num, int)
        if self.resampling_method != 'cv':
            if self.fold_num is not None:
                msg = ("Fold number should only be valid if "+
                       "`resampling_method` is set as 'cv'.")
                raise ValueError(msg)
        self.repeat_times = self._arg('repeat_times', repeat_times, int)
        if self.resampling_method == 'cv' and self.fold_num is None:
            msg = ("`fold_num` cannot be None when `resampling_method` is set to 'cv'.")
            logger.error(msg)
            raise ValueError(msg)
        self.search_strategy = self._arg('search_strategy', search_strategy, str)
        if self.search_strategy is not None:
            if self.search_strategy not in self.search_strategy:
                msg = ("Search strategy `{}` is invalid ".format(self.search_strategy)+
                       "for parameter selection.")
                logger.error(msg)
                raise ValueError(msg)
        self.random_search_times = self._arg('random_search_times', random_search_times, int)
        if self.search_strategy == 'random' and self.random_search_times is None:
            msg = ("`random_search_times` cannot be None when"+
                   " `search_strategy` is set to 'random'.")
            logger.error(msg)
            raise ValueError(msg)
        if self.search_strategy != 'random' and self.random_search_times is not None:
            msg = ("`random_search_times` should only be valid when `search_strategy` is set as 'random'.")
            raise ValueError(msg)
        self.timeout = self._arg('timeout', timeout, int)
        self.progress_indicator_id = self._arg('progress_indicator_id', progress_indicator_id, str)
        if isinstance(param_range, dict):
            param_range = [(x, param_range[x]) for x in param_range]
        if isinstance(param_values, dict):
            param_values = [(x, param_values[x]) for x in param_values]
        self.param_values = self._arg('param_values', param_values, ListOfTuples)
        self.param_range = self._arg('param_range', param_range, ListOfTuples)
        self.model_ = None
        if self.search_strategy is None:
            if self.param_values is not None:
                msg = ("Specifying the values of `{}` ".format(self.param_values[0][0])+
                       "for non-parameter-search-strategy"+
                       " parameter selection is invalid.")
                logger.error(msg)
                raise ValueError(msg)
            if self.param_range is not None:
                msg = ("Specifying the range of `{}` for ".format(self.param_range[0][0])+
                       "non-parameter-search-strategy parameter selection is invalid.")
                logger.error(msg)
                raise ValueError(msg)
        else:
            value_list = []
            if self.alpha is not None:
                value_list.append("alpha")
            if self.factor_num is not None:
                value_list.append("factor_num")
            if self.lamb is not None:
                value_list.append("lamb")
            if self.param_values is not None:
                for x in self.param_values:#pylint:disable=invalid-name
                    if len(x) != 2:#pylint:disable=bad-option-value
                        msg = ("Each tuple that specifies the values of a parameter should"+
                               " contain exactly 2 elements: 1st is parameter name,"+
                               " 2nd is a list of valid values.")
                        logger.error(msg)
                        raise ValueError(msg)
                    if x[0] not in list(self.range_params_map.keys()):
                        msg = ("Specifying the values of `{}` for ".format(x[0])+
                               "parameter selection is invalid.")
                        logger.error(msg)
                        raise ValueError(msg)
                    if x[0] in value_list:
                        msg = ("Parameter `{}` is invalid for ".format(x[0])+
                               "re-specification in parameter selection.")
                        logger.error(msg)
                        raise ValueError(msg)
                    if x[0] == 'alpha' and self.implicit is not True:
                        msg = ("`alpha` should only be valid if `implicit` is set as True.")
                        raise ValueError(msg)
                    if (x[0] == 'factor_num') and not (isinstance(x[1], list) and all(isinstance(t, _INT_TYPES) for t in x[1])):
                        msg = "Valid values of `{}` must be a list of int.".format(x[0])
                        logger.error(msg)
                        raise TypeError(msg)
                    if (x[0] in ('alpha', 'lamb')) and not (isinstance(x[1], list) and all(isinstance(t, (int, float)) for t in x[1])):
                        msg = "Valid values of `{}` must be a list of numericals.".format(x[0])
                        logger.error(msg)
                        raise TypeError(msg)
                    value_list.append(x[0])

            if self.param_range is not None:
                rsz = [3] if self.search_strategy == 'grid'else [2, 3]
                for x in self.param_range:#pylint:disable=invalid-name
                    if len(x) != 2:#pylint:disable=bad-option-value
                        msg = ("Each tuple that specifies the range of a parameter should contain"+
                               " exactly 2 elements: 1st is parameter name, 2nd is value range.")
                        logger.error(msg)
                        raise ValueError(msg)
                    if x[0] not in list(self.range_params_map.keys()):
                        msg = ("Parameter `{}` is invalid for ".format(x[0])+
                               "range specification in parameter selection.")
                        logger.error(msg)
                        raise ValueError(msg)
                    if x[0] in value_list:
                        msg = ("Parameter `{}` is invalid for ".format(x[0])+
                               "re-specification in parameter selection.")
                        logger.error(msg)
                        raise ValueError(msg)
                    if x[0] == 'alpha' and self.implicit is not True:
                        msg = ("`alpha` should only be valid if `implicit` is set as True.")
                        raise ValueError(msg)
                    if (x[0] == 'factor_num') and not (isinstance(x[1], list) and len(x[1]) in rsz and all(isinstance(t, _INT_TYPES) for t in x[1])):
                        msg = ("The provided range of `{}` is either not ".format(x[0])+
                               "a list of int, or it contains wrong number of values.")
                        logger.error(msg)
                        raise TypeError(msg)
                    if (x[0] in ('alpha', 'lamb')) and not (isinstance(x[1], list) and len(x[1]) in rsz and all(isinstance(t, (int, float)) for t in x[1])):
                        msg = ("The provided range of `{}` is either not ".format(x[0])+
                               "a list of numericals, or it contains wrong number of values.")
                        logger.error(msg)
                        raise TypeError(msg)

    def fit(self, data, usr=None, item=None, feedback=None, key=None):#pylint: disable=too-many-locals, too-many-statements, too-many-branches, too-many-arguments
        """
        Fit the ALS model with input training data. Model parameters should be given by initializing the model first.

        Parameters
        ----------
        data : DataFrame
            Data to be fit.
        key : str, optional
            Name of the ID column.
        usr : list of str, optional
            Name of the usr column.
        item : str, optional
            Name of the item column.
        feedback : str, optional
            Name of the feedback column.
        """
        conn = data.connection_context
        require_pal_usable(conn)
        cols = data.columns
        usr = self._arg('usr', usr, str)
        item = self._arg('item', item, str)
        key = self._arg('key', key, str)
        feedback = self._arg('feedback', feedback, str)
        if key is not None:
            cols.remove(key)
        if usr is None:
            usr = cols[0]
        if item is None:
            item = cols[1]
        if feedback is None:
            feedback = cols[-1]
        cols_left = [usr, item, feedback]
        param_rows = [('FACTOR_NUMBER', self.factor_num, None, None),
                      ('SEED', self.random_state, None, None),
                      ('REGULARIZATION', None, self.lamb, None),
                      ('MAX_ITERATION', self.max_iter, None, None),
                      ('EXIT_THRESHOLD', None, self.tol, None),
                      ('IMPLICIT_TRAIN', self.implicit, None, None),
                      ('LINEAR_SYSTEM_SOLVER', self.linear_solver, None, None),
                      ('THREAD_RATIO', None, self.thread_ratio, None),
                      ('RESAMPLING_METHOD', None, None, self.resampling_method),
                      ('EVALUATION_METRIC', None, None,
                       self.evaluation_metric.upper() if self.evaluation_metric is not None else None),
                      ('REPEAT_TIMES', self.repeat_times, None, None),
                      ('PARAM_SEARCH_STRATEGY', None, None, self.search_strategy),
                      ('TIMEOUT', self.timeout, None, None),
                      ('PROGRESS_INDICATOR_ID', None, None, self.progress_indicator_id)
                     ]
        if self.tol is not None:
            param_rows.extend([('EXIT_INTERVAL', self.exit_interval, None, None)])
        if self.linear_solver is not None:
            param_rows.extend([('CG_MAX_ITERATION', self.cg_max_iter, None, None)])
        if self.implicit is not None:
            param_rows.extend([('ALPHA', None, self.alpha, None)])
        if self.resampling_method is not None:
            param_rows.extend([('FOLD_NUM', self.fold_num, None, None)])
        if self.search_strategy is not None:
            param_rows.extend([('RANDOM_SEARCH_TIMES', self.random_search_times, None, None)])
        if self.param_values is not None:
            for x in self.param_values:#pylint:disable=invalid-name
                values = str(x[1]).replace('[', '{').replace(']', '}')
                param_rows.extend([(quotename(self.range_params_map[x[0]]+"_VALUES"),
                                    None, None, values)])
        if self.param_range is not None:
            for x in self.param_range:#pylint:disable=invalid-name
                range_ = str(x[1])
                if len(x[1]) == 2 and self.search_strategy == 'random':
                    range_ = range_.replace(',', ',,')
                param_rows.extend([(quotename(self.range_params_map[x[0]]+"_RANGE"),
                                    None, None, range_)])
        unique_id = str(uuid.uuid1()).replace('-', '_').upper()
        tables = ['MODEL_METADATA', 'MODEL_MAP', 'MODEL_FACTORS', 'ITERATION_INFORMATION', 'STATISTICS', 'OPTIMAL_PARAMETER']
        tables = ["#PAL_ALS_{}_TBL_{}_{}".format(tbl, self.id, unique_id) for tbl in tables]
        metadata_tbl, map_tbl, factors_tbl, iter_info_tbl, stat_tbl, optim_param_tbl = tables
        try:
            call_pal_auto(conn,
                          "PAL_ALS",
                          data[cols_left],
                          ParameterTable().with_data(param_rows),
                          *tables)
        except dbapi.Error as db_err:
            logger.exception(str(db_err))
            try_drop(conn, tables)
            raise
        self.metadata_ = conn.table(metadata_tbl)#pylint:disable=attribute-defined-outside-init
        self.map_ = conn.table(map_tbl)#pylint:disable=attribute-defined-outside-init
        self.factors_ = conn.table(factors_tbl)#pylint:disable=attribute-defined-outside-init
        self.optim_param_ = conn.table(optim_param_tbl)#pylint:disable=attribute-defined-outside-init
        self.stats_ = conn.table(stat_tbl)#pylint:disable=attribute-defined-outside-init
        self.iter_info_ = conn.table(iter_info_tbl)#pylint:disable=attribute-defined-outside-init
        self.model_ = [self.metadata_, self.map_, self.factors_]

    def predict(self, data, key, usr=None, item=None, thread_ratio=None):#pylint:disable=too-many-arguments
        """
        Prediction for the input data with the trained ALS model.

        Parameters
        ----------
        data : DataFrame
            Data to be fit.
        key : str, optional
            Name of the ID column.
        usr : list of str, optional
            Name of the usr column.
        item : str, optional
            Name of the item column.
        thread_ratio : float, optional
            Specifies the upper limit of thread usage in proportion of current available threads.

            The valid range of the value is [0, 1].

            Default to 0.

        Returns
        -------
        DataFrame
            Prediction result of the missing values(e.g. user feedback) in the input data, structured as follows:

              - 1st column : Data ID
              - 2nd column : User name/ID
              - 3rd column : Item name/ID
              - 4th column : Predicted feedback values
        """
        conn = data.connection_context
        require_pal_usable(conn)
        if getattr(self, 'model_') is None:
        #if not hasattr(self, 'map_'):
            raise FitIncompleteError("Model not initialized. Perform a fit first.")
        cols = data.columns
        usr = self._arg('usr', usr, str)
        item = self._arg('item', item, str)
        key = self._arg('key', key, str, required=True)
        cols.remove(key)
        if usr is None:
            usr = cols[0]
        if item is None:
            item = cols[-1]
        cols_left = [key, usr, item]
        unique_id = str(uuid.uuid1()).replace('-', '_').upper()
        result_tbl = '#PAL_ALS_RESULT_TBL_{}_{}'.format(self.id, unique_id)
        param_rows = [('THREAD_RATIO', None, thread_ratio, None)]
        try:
            call_pal_auto(conn,
                          'PAL_ALS_PREDICT',
                          data[cols_left],
                          self.model_[0],
                          self.model_[1],
                          self.model_[2],
                          ParameterTable().with_data(param_rows),
                          result_tbl)
        except dbapi.Error as db_err:
            logger.exception(str(db_err))
            try_drop(conn, result_tbl)
            raise
        return conn.table(result_tbl)

class FRM(PALBase):#pylint:disable=too-many-arguments, too-few-public-methods, too-many-instance-attributes
    """
    class FRM
    Factorized Polynomial Regression Models of recommender system algorithms

    Parameters
    ----------

    solver : {'sgd', 'momentum', 'nag', 'adagrad'}, optional
        Specifies the method for solving the objective minimization problem.

        Default to 'sgd'.
    factor_num : int, optional
        Length of factor vectors in the model.

        Default to 8.
    init : float, optional
        Variance of the normal distribution used to initialize the model parameters.

        Default to 1e-2.
    random_state : int, optional
        Specifies the seed for random number generator.

            -  0: Uses the current time as the seed.
            -  Others: Uses the specified value as the seed.

        Note that due to the inherently randomicity of parallel sgc, models of different
        trainings might be different even with the same seed of random number generator.

        Default to 0.
    lamb : float, optional
        L2 regularization of the factors.

        Default to 1e-8.
    linear_lamb : float, optional
        L2 regularization of the factors.

        Default to 1e-10.
    thread_ratio : float, optional
        Controls the proportion of available threads that can be used.

        The value range is from 0 to 1, where 0 indicates a single thread,
        and 1 indicates up to all available threads.

        Values between 0 and 1 will use that percentage of available threads.

        Values outside this range tell PAL to heuristically determine the number of threads to use.

        Defaults to 0.
    max_iter : int, optional
        Specifies the maximum number of iterations for the ALS algorithm.

        Default value is 50.
    sgd_tol : float, optional
        Exit threshold.

        The algorithm exits when the cost function has not decreased
        more than this threshold in ``sgd_exit_interval`` steps.

        Default to 1e-5
    sgd_exit_interval : int, optional
        The algorithm exits when the cost function has not decreased
        more than ``sgd_tol`` in ``sgd_exit_interval`` steps.

        Default to 5.
    momentum : float, optional
        The momentum factor in method 'momentum' or 'nag'.

        Valid only when `method` is 'momentum' or 'nag'.

        Default to 0.9.
    resampling_method : {'cv', 'bootstrap'}, optional
        Specifies the resampling method for model evaluation or parameter selection.

        If not specified, neither model evaluation nor parameter selection is activated.

        No default value.
    evaluation_metric : {'rmse'}, optional
        Specifies the evaluation metric for model evaluation or parameter selection.

        If not specified, neither model evaluation nor parameter selection is activated.

        No default value.
    fold_num : int, optional
        Specifies the fold number for the cross validation method.

        Mandatory and valid only when ``resampling_method`` is set to 'cv'.

        Default to 1.
    repeat_times : int, optional
        Specifies the number of repeat times for resampling.

        Default to 1.
    search_strategy : {'grid', 'random'}, optional
        Specifies the method to activate parameter selection.

        No default value.
    random_search_times : int, optional
        Specifies the number of times to randomly select candidate parameters for selection.

        Mandatory and valid when PARAM_SEARCH_STRATEGY is set to random.

        No default value.
    timeout : int, optional
        Specifies maximum running time for model evaluation or parameter selection, in seconds.

        No timeout when 0 is specified.

        Default to 0.
    progress_indicator_id : str, optional
        Sets an ID of progress indicator for model evaluation or parameter selection.

        No progress indicator is active if no value is provided.

        No default value.
    param_values : dict or ListOfTuples, optional

        Specifies values of parameters to be selected.

        Input should be a dict or list of tuple of two elements, with the key/1st element being the parameter name,
        and value/2nd element being a list of values for selection.

        Valid only when ``resampling_method`` and ``search_strategy`` are both specified.

        Valid paramter names include : 'factor_num', 'lamb', 'linear_lamb', 'momentum'.

        No default value.
    param_range : dict or ListOfTuples, optional

        Specifies ranges of param to be selected.

        Input should be a dict or list of tuple of two elements , with key/1st element being the parameter name,
        and value/2nd element being a list of numericals indicating the range for selection.

        Valid only when ``resampling_method`` and ``search_strategy`` are both specified.

        Valid parameter names include:'factor_num', 'lamb', 'linear_lamb', 'momentum'.

        No default value.

    Attributes
    ----------
    metadata_ : DataFrame
        Model metadata content.
    model_ : DataFrame
        Model (Map, Weight)
    factors_ : DataFrame
        Decomposed factors.
    optim_param_ : DataFrame
        Optimal parameters selected.
    stats_ : DataFrame
        Statistic values
    iter_info_ : DataFrame
        Cost function value and RMSE of corresponding iteration.

    Examples
    --------
    Input dataframe for training:

    >>> df_train.collect()
        ID    USER    MOVIE     TIMESTAMP    FEEDBACK
         1      A     Movie1        3          4.8
         2      A     Movie2        3          4.0
         3      A     Movie4        1          4.0
         4      A     Movie5        2          4.0
         5      A     Movie6        3          4.8
         6      A     Movie8        2          3.8
         7      A     Bad_Movie     1          2.5
         8      B     Movie2        3          4.8
         9      B     Movie3        2          4.8
         1      B     Movie4        2          5.0
         1      B     Movie5        4          5.0
         1      B     Movie7        1          3.5
         1      B     Movie8        2          4.8
         1      B     Bad_Movie     3          2.8
         1      C     Movie1        2          4.1
         1      C     Movie2        4          4.2
         1      C     Movie4        3          4.2
         1      C     Movie5        1          4.0
         1      C     Movie6        4          4.2
         2      C     Movie7        3          3.2
         2      C     Movie8        1          3.0
         2      C     Bad_Movie     2          2.5
         2      D     Movie1        3          4.5
         2      D     Movie3        2          3.5
         2      D     Movie4        2          4.5
         2      D     Movie6        2          3.9
         2      D     Movie7        4          3.5
         2      D     Movie8        3          3.5
         2      D     Bad_Movie     3          2.5
         3      E     Movie1        2          4.5
         3      E     Movie2        2          4.0
         3      E     Movie3        2          3.5
         3      E     Movie4        4          4.5
         3      E     Movie5        3          4.5
         3      E     Movie6        2          4.2
         3      E     Movie7        4          3.5
         3      E     Movie8        3          3.5

    Input user dataframe for training:

    >>> usr_info.collect()
        USER            USER_SIDE_FEATURE
        -- There is no side information for user provided. --

    Input item dataframe for training:

    >>> item_info.collect()
        MOVIE       GENRES
        Movie1      Sci-Fi
        Movie2      Drama,Romance
        Movie3      Drama,Sci-Fi
        Movie4      Crime,Drama
        Movie5      Crime,Drama
        Movie6      Sci-Fi
        Movie7      Crime,Drama
        Movie8      Sci-Fi,Thriller
        Bad_Movie   Romance,Thriller


    Creating FRM instance:

    >>> frm = FRM(factor_num=2, solver='adagrad',
                  learning_rate=0, max_iter=100,
                  thread_ratio=0.5, random_state=1)

    Performing fit() on given dataframe:

    >>> frm.fit(self.df_train, self.usr_info, self.item_info, categorical_variable='TIMESTAMP')

    >>> frm.factors_.collect().head(10)
             FACTOR_ID    FACTOR
        0          0 -0.083550
        1          1 -0.083654
        2          2  0.582244
        3          3 -0.102799
        4          4 -0.441795
        5          5 -0.013341
        6          6 -0.099548
        7          7  0.245046
        8          8 -0.056534
        9          9 -0.342042

    Performing predict() on given predicting dataframe:

    >>> res = frm.predict(self.df_predict, self.usr_info, self.item_info, thread_ratio=0.5, key='ID')

    >>> res.collect()
               ID USER  ITEM  PREDICTION
        0   1    A  None    3.486804
        1   2    A     4    3.490246
        2   3    B     2    5.436991
        3   4    B     3    5.287031
        4   5    C     2    3.015121
        5   6    D     1    3.602543
        6   7    D     3    4.097683
        7   8    E     2    2.317224
    """
    solver_map = {'sgd': 0, 'momentum': 1, 'nag': 2, 'adagrad': 3}
    resampling_method_list = {'cv' : 0, 'bootstrap' : 1}
    evaluation_metric_list = {'rmse'}
    search_strat_list = {'grid': 'grid', 'random': 'random'}
    range_params_map = {'factor_num' : 'FACTOR_NUMBER',
                        'lamb' : 'REGULARIZATION',
                        'linear_lamb' : 'LINEAR_REGULARIZATION',
                        'momentum' : 'MOMENTUM'}
    def __init__(self,#pylint: disable=too-many-arguments, too-many-locals, too-many-branches, too-many-statements
                 solver=None,
                 factor_num=None,
                 init=None,
                 random_state=None,
                 learning_rate=None,
                 linear_lamb=None,
                 lamb=None,
                 max_iter=None,
                 sgd_tol=None,
                 sgd_exit_interval=None,
                 thread_ratio=None,
                 momentum=None,
                 resampling_method=None,
                 evaluation_metric=None,
                 fold_num=None,
                 repeat_times=None,
                 search_strategy=None,
                 random_search_times=None,
                 timeout=None,
                 progress_indicator_id=None,
                 param_values=None,
                 param_range=None):
        super(FRM, self).__init__()
        setattr(self, 'hanaml_parameters', pal_param_register())
        self.solver = self._arg('solver', solver, self.solver_map)
        #if self.method is not None:
        #    if self.method not in self.method_map:
        #        msg = ("Method '{}' is invalid ".format(method))
        #        raise ValueError(msg)
        self.factor_num = self._arg('factor_num', factor_num, int)
        self.init = self._arg('init', init, float)
        self.random_state = self._arg('random_state', random_state, int)
        self.learning_rate = self._arg('learning_rate', learning_rate, float)
        self.linear_lamb = self._arg('linear_lamb', linear_lamb, float)
        self.lamb = self._arg('lamb', lamb, float)
        self.max_iter = self._arg('max_iter', max_iter, int)
        self.sgd_tol = self._arg('sgd_tol', sgd_tol, float)
        self.sgd_exit_interval = self._arg('sgd_exit_interval', sgd_exit_interval, int)
        self.thread_ratio = self._arg('thread_ratio', thread_ratio, float)
        self.momentum = self._arg('momentum', momentum, float)
        #if self.solver is None or self.solver in (0, 3):
        #    if self.momentum is not None:
        #        msg = ("`momentum` should only be valid if method is set to 'momentum' or 'nag'.")
        #        raise ValueError(msg)
        self.resampling_method = self._arg('resampling_method', resampling_method, str)
        if self.resampling_method is not None:
            if self.resampling_method not in self.resampling_method_list:#pylint:disable=line-too-long, bad-option-value
                msg = ("Resampling method '{}' is not available ".format(self.resampling_method)+
                       "for model evaluation/parameter selection in FRM.")
                logger.error(msg)
                raise ValueError(msg)
        self.evaluation_metric = self._arg('evaluation_metric', evaluation_metric, str)
        if self.evaluation_metric is not None:
            if self.evaluation_metric not in self.evaluation_metric_list:
                msg = ("Evaluation metric '{}' is not available.".format(self.evaluation_metric))
                logger.error(msg)
                raise ValueError(msg)
        self.fold_num = self._arg('fold_num', fold_num, int)
        #if self.resampling_method != 'cv' and self.fold_num is not None:
        #    msg = ("`fold_num` should only be valid when 'resampling_method' is set as 'cv'.")
        #    raise ValueError(msg)
        if self.resampling_method == 'cv' and self.fold_num is None:
            msg = ("`fold_num` cannot be None when `resampling_method` is set to 'cv'.")
            logger.error(msg)
            raise ValueError(msg)
        self.repeat_times = self._arg('repeat_times', repeat_times, int)
        self.search_strategy = self._arg('search_strategy', search_strategy, self.search_strat_list)
        #if self.search_strategy is not None:
        #    if self.search_strategy not in self.search_strat_list:
        #        msg = ("Search strategy '{}' is invalid ".format(self.search_strategy)+
        #               "for parameter selection.")
        #        logger.error(msg)
        #        raise ValueError(msg)
        self.random_search_times = self._arg('random_search_times', random_search_times, int)
        if self.search_strategy == 'random' and self.random_search_times is None:
            msg = ("`random_search_times` cannot be None when `search_strategy` is set to 'random'.")
            logger.error(msg)
            raise ValueError(msg)
        if self.search_strategy != 'random' and self.random_search_times is not None:
            msg = ("`random_search_times` should only be valid when `search_strategy` is set as 'random'.")
            raise ValueError(msg)
        self.timeout = self._arg('timeout', timeout, int)
        self.progress_indicator_id = self._arg('progress_indicator_id', progress_indicator_id, str)
        if isinstance(param_range, dict):
            param_range = [(x, param_range[x]) for x in param_range]
        if isinstance(param_values, dict):
            param_values = [(x, param_values[x]) for x in param_values]
        self.param_values = self._arg('param_values', param_values, ListOfTuples)
        self.param_range = self._arg('param_range', param_range, ListOfTuples)
        self.model_ = None
        if self.search_strategy is None:
            if self.param_values is not None:
                msg = ("Specifying the values of `{}` ".format(self.param_values[0][0])+
                       "for non-parameter-search-strategy parameter selection is invalid.")
                logger.error(msg)
                raise ValueError(msg)
            if self.param_range is not None:
                msg = ("Specifying the range of `{}` for ".format(self.param_range[0][0])+
                       "non-parameter-search-strategy parameter selection is invalid.")
                logger.error(msg)
                raise ValueError(msg)
        else:
            value_list = []
            if self.factor_num is not None:
                value_list.append("factor_num")
            if self.linear_lamb is not None:
                value_list.append("linear_lamb")
            if self.lamb is not None:
                value_list.append("lamb")
            if self.momentum is not None:
                value_list.append("momentum")
            if self.param_values is not None:
                for x in self.param_values:#pylint:disable=invalid-name
                    if len(x) != 2:#pylint:disable=bad-option-value
                        msg = ("Each tuple that specifies the values of a parameter should"+
                               " contain exactly 2 elements: 1st is parameter name,"+
                               " 2nd is a list of valid values.")
                        logger.error(msg)
                        raise ValueError(msg)
                    if x[0] not in list(self.range_params_map.keys()):
                        msg = ("Specifying the values of `{}` for ".format(x[0])+
                               "parameter selection is invalid.")
                        logger.error(msg)
                        raise ValueError(msg)
                    if x[0] in value_list:
                        msg = ("Parameter `{}` is invalid for ".format(x[0])+
                               "re-specification in parameter selection.")
                        logger.error(msg)
                        raise ValueError(msg)
                    if x[0] == 'momentum' and self.solver not in (2, 1):
                        msg = ("`momentum` should only be valid if `solver` is set as 'momentum' or 'nag'.")
                        raise ValueError(msg)
                    if (x[0] == 'factor_num') and not (isinstance(x[1], list) and all(isinstance(t, _INT_TYPES) for t in x[1])):
                        msg = "Valid values of `{}` must be a list of int.".format(x[0])
                        logger.error(msg)
                        raise TypeError(msg)
                    if (x[0] in ('linear_lamb', 'lamb', 'momentum')) and not (isinstance(x[1], list) and all(isinstance(t, (int, float)) for t in x[1])):
                        msg = "Valid values of `{}` must be a list of numericals.".format(x[0])
                        logger.error(msg)
                        raise TypeError(msg)
                    value_list.append(x[0])

            if self.search_strategy is not None:
                rsz = [3] if self.search_strategy == 'grid'else [2, 3]
                for x in self.param_range:#pylint:disable=invalid-name
                    if len(x) != 2:#pylint:disable=bad-option-value
                        msg = ("Each tuple that specifies the range of a parameter should contain"+
                               " exactly 2 elements: 1st is parameter name, 2nd is value range.")
                        logger.error(msg)
                        raise ValueError(msg)
                    if x[0] not in list(self.range_params_map.keys()):
                        msg = ("Parameter `{}` is invalid for ".format(x[0])+
                               "range specification in parameter selection.")
                        logger.error(msg)
                        raise ValueError(msg)
                    if x[0] in value_list:
                        msg = ("Parameter `{}` is invalid for ".format(x[0])+
                               "re-specification in parameter selection.")
                        logger.error(msg)
                        raise ValueError(msg)
                    if x[0] == 'momentum' and self.solver not in (1, 2):
                        msg = ("`momentum` should only be valid if method is set as 'momentum' or 'nag'.")
                        raise ValueError(msg)
                    if (x[0] == 'factor_num') and not (isinstance(x[1], list) and len(x[1]) in rsz and all(isinstance(t, _INT_TYPES) for t in x[1])):
                        msg = ("The provided range of `{}` is either not ".format(x[0])+
                               "a list of int, or it contains wrong number of values.")
                        logger.error(msg)
                        raise TypeError(msg)
                    if (x[0] in ('linear_lamb', 'lamb', 'momentum')) and not (isinstance(x[1], list) and len(x[1]) in rsz and all(isinstance(t, (int, float)) for t in x[1])):
                        msg = ("The provided range of `{}` is either not ".format(x[0])+
                               "a list of numericals, or it contains wrong number of values.")
                        logger.error(msg)
                        raise TypeError(msg)

    def fit(self, data, usr_info, item_info, key=None, usr=None, item=None, feedback=None,#pylint: disable=too-many-locals, too-many-statements, too-many-branches, too-many-arguments
            features=None, usr_features=None, item_features=None,#pylint: disable=too-many-locals, too-many-statements, too-many-branches, too-many-arguments
            usr_key=None, item_key=None,#pylint: disable=too-many-locals, too-many-statements, too-many-branches, too-many-arguments
            categorical_variable=None, usr_categorical_variable=None, item_categorical_variable=None):#pylint: disable=too-many-locals, too-many-statements, too-many-branches, too-many-arguments
        """
        Fit the FRM model with input training data. Model parameters should be given by initializing the model first.

        Parameters
        ----------
        data : DataFrame
            Data to be fit.
        usr_info : DataFrame
            User side features.
        item_info : DataFrame
            Item side features.
        key : str, optional
            Name of the ID column.
        usr : list of str, optional
            Name of the usr column.
        item : str, optional
            Name of the item column.
        feedback : str, optional
            Name of the feedback column.
        features : str/listOfStrings, optional
            Global side features column name in the training dataframe.
        usr_features : str/listOfStrings, optional
            User side features column name in the training dataframe.
        item_features : str/listOfStrings, optional
            Item side features column name in the training dataframe.
        categorical_variable : str/ListofStrings, optional
            Indicates whether or not a column data is actually corresponding
            to a category variable even the data type of this column is INTEGER.

            By default, 'VARCHAR' or 'NVARCHAR' is category variable, and 'INTEGER'
            or 'DOUBLE' is continuous variable.
        usr_categorical_variable : str/ListofStrings, optional
            Name of user side feature columns that should be treated as categorical.
        item_categorical_variable : str/ListofStrings, optional
            Name of item side feature columns that should be treated as categorical.
        """
        conn = data.connection_context
        require_pal_usable(conn)
        if categorical_variable is not None:
            if isinstance(categorical_variable, str):
                categorical_variable = [categorical_variable]
            try:
                categorical_variable = self._arg('categorical_variable', categorical_variable, ListOfStrings)#pylint: disable=undefined-variable
            except:
                msg = ("'categorical_variable' must be list of string or string.")
                logger.error(msg)
                raise TypeError(msg)
        if usr_categorical_variable is not None:
            if isinstance(usr_categorical_variable, str):
                usr_categorical_variable = [usr_categorical_variable]
            try:
                usr_categorical_variable = self._arg('usr_categorical_variable', usr_categorical_variable, ListOfStrings)#pylint: disable=undefined-variable
            except:
                msg = ("'usr_categorical_variable' must be list of string or string.")
                logger.error(msg)
                raise TypeError(msg)
        if item_categorical_variable is not None:
            if isinstance(item_categorical_variable, str):
                item_categorical_variable = [item_categorical_variable]
            try:
                item_categorical_variable = self._arg('item_categorical_variable', item_categorical_variable, ListOfStrings)#pylint: disable=undefined-variable
            except:
                msg = ("'item_categorical_variable' must be list of string or string.")
                logger.error(msg)
                raise TypeError(msg)
        cols = data.columns
        if key is not None:
            cols.remove(key)
        if usr is None:
            usr = cols[0]
        cols.remove(usr)
        if item is None:
            item = cols[1]
        cols.remove(item)
        if feedback is None:
            feedback = cols[-1]
        cols.remove(feedback)
        if features is not None:
            if isinstance(features, str):
                features = [features]
            try:
                features = self._arg('features', features, ListOfStrings)#pylint: disable=undefined-variable
            except:
                msg = ("'features' must be list of string or string.")
                logger.error(msg)
                raise TypeError(msg)
        else:
            features = cols
        data_ = data[[usr] + [item] + features + [feedback]]
        if usr_features is not None:
            if isinstance(usr_features, str):
                usr_features = [usr_features]
            try:
                usr_features = self._arg('usr_features', usr_features, ListOfStrings)#pylint: disable=undefined-variable
            except:
                msg = ("'usr_features' must be list of string or string.")
                logger.error(msg)
                raise TypeError(msg)
        usr_cols = usr_info.columns
        if usr_key is not None:
            usr_cols.remove(usr_key)
        elif len(usr_cols) > 0:#pylint:disable=len-as-condition
            usr_cols.remove(usr_cols[0])
        if usr_features is not None:
            for var in usr_features:
                usr_cols.remove(var)

        if item_features is not None:
            if isinstance(item_features, str):
                item_features = [item_features]
            try:
                item_features = self._arg('item_features', item_features, ListOfStrings)#pylint: disable=attribute-defined-outside-init, undefined-variable
            except:
                msg = ("'item_features' must be list of string or string.")
                logger.error(msg)
                raise TypeError(msg)
        item_cols = item_info.columns
        if item_key is not None:
            item_cols.remove(item_key)
        elif len(item_cols) > 0:#pylint:disable=len-as-condition
            item_cols.remove(item_cols[0])
        if item_features is not None:
            for var in item_features:
                item_cols.remove(var)
        param_rows = [('FACTOR_NUMBER', self.factor_num, None, None),
                      ('SEED', self.random_state, None, None),
                      ('REGULARIZATION', None, self.lamb, None),
                      ('LINEAR_REGULARIZATION', None, self.linear_lamb, None),
                      ('MAX_ITERATION', self.max_iter, None, None),
                      ('SGD_EXIT_THRESHOLD', None, self.sgd_tol, None),
                      ('SGD_EXIT_INTERVAL', None, self.sgd_exit_interval, None),
                      ('TIMEOUT', self.timeout, None, None),
                      ('PARAM_SEARCH_STRATEGY', None, None, self.search_strategy),
                      ('RESAMPLING_METHOD', None, None, self.resampling_method),
                      ('EVALUATION_METRIC', None, None,
                       self.evaluation_metric.upper() if self.evaluation_metric is not None else None),
                      ('PROGRESS_INDICATOR_ID', None, None, self.progress_indicator_id),
                      ('LEARNING_RATE', None, self.learning_rate, None),
                      ('THREAD_RATIO', None, self.thread_ratio, None),
                      ('REPEAT_TIMES', self.repeat_times, None, None),
                      ('INITIALIZATION', None, self.init, None),
                      ('METHOD', self.solver, None, None),
                      ('FOLD_NUM', self.fold_num, None, None)
                     ]
        #if self.solver is not None:
        #    param_rows.extend([('METHOD', self.method, None, None)])
        #if self.resampling_method is not None:
        #    param_rows.extend([('FOLD_NUM', self.fold_num, None, None)])
        if categorical_variable is not None:
            param_rows.extend(('CATEGORICAL_VARIABLE', None, None, variable)
                              for variable in categorical_variable)
        if usr_categorical_variable is not None:
            param_rows.extend(('USER_CATEGORICAL_VARIABLE', None, None, variable)
                              for variable in usr_categorical_variable)
        if item_categorical_variable is not None:
            param_rows.extend(('ITEM_CATEGORICAL_VARIABLE', None, None, variable)
                              for variable in item_categorical_variable)
        if self.solver in (1, 2):
            param_rows.extend([('MOMENTUM', None, self.momentum, None)])
        if self.search_strategy is not None:
            param_rows.extend([('RANDOM_SEARCH_TIMES', self.random_search_times, None, None)])
        if usr_cols is not None and usr_features is not None:
            param_rows.extend(('USER_EXCLUDED_FEATURE', None, None, exc)
                              for exc in usr_cols)
        if item_cols is not None and item_features is not None:
            param_rows.extend(('ITEM_EXCLUDED_FEATURE', None, None, exc)
                              for exc in item_cols)
        if self.param_values is not None:
            for x in self.param_values:#pylint:disable=invalid-name
                values = str(x[1]).replace('[', '{').replace(']', '}')
                param_rows.extend([(quotename(self.range_params_map[x[0]]+"_VALUES"),
                                    None, None, values)])
        if self.param_range is not None:
            for x in self.param_range:#pylint:disable=invalid-name
                range_ = str(x[1])
                if len(x[1]) == 2 and self.search_strategy == 'random':
                    range_ = range_.replace(',', ',,')
                param_rows.extend([(quotename(self.range_params_map[x[0]]+"_RANGE"),
                                    None, None, range_)])
        unique_id = str(uuid.uuid1()).replace('-', '_').upper()
        tables = ['MODEL_METADATA', 'MODEL', 'MODEL_FACTORS', 'ITERATION_INFORMATION', 'STATISTICS', 'OPTIMAL_PARAMETER']
        tables = ["#PAL_FRM_{}_TBL_{}_{}".format(tbl, self.id, unique_id) for tbl in tables]
        metadata_tbl, model_tbl, factors_tbl, iter_info_tbl, stat_tbl, optim_param_tbl = tables
        try:
            call_pal_auto(conn,
                          "PAL_FRM",
                          data_,
                          usr_info,
                          item_info,
                          ParameterTable().with_data(param_rows),
                          *tables)
        except dbapi.Error as db_err:
            logger.exception(str(db_err))
            try_drop(conn, tables)
            raise
        self.metadata_ = conn.table(metadata_tbl)#pylint:disable=attribute-defined-outside-init
        self.model_ = conn.table(model_tbl)#pylint:disable=attribute-defined-outside-init
        self.factors_ = conn.table(factors_tbl)#pylint:disable=attribute-defined-outside-init
        self.optim_param_ = conn.table(optim_param_tbl)#pylint:disable=attribute-defined-outside-init
        self.stats_ = conn.table(stat_tbl)#pylint:disable=attribute-defined-outside-init
        self.iter_info_ = conn.table(iter_info_tbl)#pylint:disable=attribute-defined-outside-init

    def predict(self, data, usr_info, item_info, key, usr=None, item=None, features=None, thread_ratio=None):#pylint: disable=too-many-locals, too-many-statements, too-many-branches, too-many-arguments
        """
        Prediction for the input data with the trained FRM model.

        Parameters
        ----------
        data : DataFrame
            Data to be fit.
        key : str, optional
            Name of the ID column.
            If not provided, it defaults to the 1st column of ``data``.
        usr : list of str, optional
            Name of the column containing user name or user ID.
            If not provided, it defaults to 1st non-ID column of ``data``.
        item : str, optional
            Name of the item column.

            Name of the column containing item name or item ID.

            If not provided, it defaults to the 1st non-ID, non-usr column of ``data``.
        usr_info : DataFrame
            User side features.
        item_info : DataFrame
            Item side features.
        features : str/listOfStrings, optional
            Global side features column name in the training dataframe.
        thread_ratio : float, optional
            Specifies the upper limit of thread usage in proportion of current available threads.

            The valid range of the value is [0,1].

            Default to 0.

        Returns
        -------
        DataFrame
            Prediction result of FRM algorithm, structured as follows:

            -   1st column : Data ID
            -   2nd column : User name/ID
            -   3rd column : Item name/Id
            -   4th column : Predicted rating

        """
        conn = data.connection_context
        require_pal_usable(conn)
        if getattr(self, 'model_') is None:
        #if not hasattr(self, 'model_'):
            raise FitIncompleteError("Model not initialized. Perform a fit first.")
        cols = data.columns
        key = self._arg('key', key, str, required=True)
        cols.remove(key)
        if usr is None:
            usr = cols[0]
        cols.remove(usr)
        if item is None:
            item = cols[-1]
        cols.remove(item)
        if features is not None:
            if isinstance(features, str):
                features = [features]
            try:
                features = self._arg('features', features, ListOfStrings)#pylint: disable=undefined-variable
            except:
                msg = ("'features' must be list of string or string.")
                logger.error(msg)
                raise TypeError(msg)
        else:
            features = cols
        data_ = data[[key] + [usr] + [item] + features]
        unique_id = str(uuid.uuid1()).replace('-', '_').upper()
        result_tbl = '#PAL_FRM_RESULT_TBL_{}_{}'.format(self.id, unique_id)
        param_rows = [('THREAD_RATIO', None, thread_ratio, None)]
        try:
            call_pal_auto(conn, 'PAL_FRM_PREDICT', data_, usr_info, item_info,
                          self.metadata_, self.model_, self.factors_,
                          ParameterTable().with_data(param_rows), result_tbl)
        except dbapi.Error as db_err:
            logger.exception(str(db_err))
            try_drop(conn, result_tbl)
            raise
        return conn.table(result_tbl)

class _FFMBase(PALBase):#pylint: disable=too-many-instance-attributes, too-few-public-methods
    """
    Base class for Field-Aware Factorization Machine of recommender system algorithms.
    """
    handle_missing_map = {'remove': 1, 'skip' : 1, 'replace' : 2, 'fill_zero' : 2}
    def __init__(self,#pylint: disable=too-many-arguments, too-many-locals, too-many-branches, too-many-statements
                 functionality=None,
                 ordering=None,
                 normalise=None,
                 include_linear=None,
                 include_constant=None,
                 early_stop=None,
                 random_state=None,
                 factor_num=None,
                 max_iter=None,
                 train_size=None,
                 learning_rate=None,
                 linear_lamb=None,
                 poly2_lamb=None,
                 tol=None,
                 exit_interval=None,
                 handle_missing=None):
        super(_FFMBase, self).__init__()
        setattr(self, 'hanaml_parameters', pal_param_register())
        self.handle_missing = self._arg('handle_missing', handle_missing, self.handle_missing_map)
        if self.handle_missing is None:
            self.handle_missing = self.handle_missing_map['fill_zero']
        self.exit_interval = self._arg('exit_interval', exit_interval, int)
        self.tol = self._arg('tol', tol, float)
        self.learning_rate = self._arg('learning_rate', learning_rate, float)
        self.linear_lamb = self._arg('linear_lamb', linear_lamb, float)
        self.poly2_lamb = self._arg('poly2_lamb', poly2_lamb, float)
        self.random_state = self._arg('random_state', random_state, int)
        self.max_iter = self._arg('max_iter', max_iter, int)
        self.train_size = self._arg('train_size', train_size, float)
        self.factor_num = self._arg('factor_num', factor_num, int)
        self.normalise = self._arg('normalise', normalise, bool)
        self.ordering = self._arg('ordering', ordering, ListOfStrings)
        if ordering is not None:
            self.ordering = (', ').join(self.ordering)
        self.include_linear = self._arg('include_linear', include_linear, bool)
        self.early_stop = self._arg('early_stop', early_stop, bool)
        self.include_constant = self._arg('include_constant', include_constant, bool)
        self.functionality = self._arg('functionality', functionality, str)
        self.model_ = None

    def _fit(self, data, key=None, features=None, label=None, categorical_variable=None, delimiter=None):#pylint: disable=too-many-locals, too-many-statements, too-many-branches, too-many-arguments
        conn = data.connection_context
        require_pal_usable(conn)
        delimiter = self._arg('delimiter', delimiter, str)
        label = self._arg('label', label, str)
        if categorical_variable is not None:
            if isinstance(categorical_variable, str):
                categorical_variable = [categorical_variable]
            try:
                categorical_variable = self._arg('categorical_variable', categorical_variable, ListOfStrings)#pylint: disable=undefined-variable
            except:
                msg = ("`categorical_variable` must be list of string or string.")
                logger.error(msg)
                raise TypeError(msg)
        cols = data.columns
        if key is not None:
            cols.remove(key)
        de_col = cols[label] if label is not None else cols[-1]
        de_col_dtype = data.dtypes([de_col])[0][1]
        de_col_is_categorical = False
        if categorical_variable is not None:
            if de_col in categorical_variable:
                de_col_is_categorical = True

        if not(de_col_dtype in ('INT', 'DOUBLE') and not de_col_is_categorical) and self.functionality == 'regression':
            msg = ("Cannot do regression when response is not numeric.")
            logger.error(msg)
            raise ValueError(msg)
        if de_col_dtype == 'DOUBLE' and self.functionality == 'ranking':
            msg = ("Cannot do ranking when response is of double type.")
            logger.error(msg)
            raise ValueError(msg)

        cols.remove(de_col)
        if features is not None:
            if isinstance(features, str):
                features = [features]
            try:
                features = self._arg('features', features, ListOfStrings)#pylint: disable=undefined-variable
            except:
                msg = ("`features` must be list of string or string.")
                logger.error(msg)
                raise TypeError(msg)
        else:
            features = cols
        data_ = data[features + [de_col]]
        param_rows = [('DELIMITER', None, None, delimiter),
                      ('TASK', None, None, self.functionality),
                      ('DEPENDENT_VARIABLE', None, None, label),
                      ('NORMALISE', self.normalise, None, None),
                      ('ORDERING', None, None, self.ordering),
                      ('INCLUDE_CONSTANT', self.include_constant, None, None),
                      ('INCLUDE_LINEAR', self.include_linear, None, None),
                      ('EARLY_STOP', self.early_stop, None, None),
                      ('SEED', self.random_state, None, None),
                      ('K_NUM', self.factor_num, None, None),
                      ('MAX_ITERATION', self.max_iter, None, None),
                      ('TRAIN_RATIO', None, self.train_size, None),
                      ('LEARNING_RATE', None, self.learning_rate, None),
                      ('LINEAR_LAMBDA', None, self.linear_lamb, None),
                      ('POLY2_LAMBDA', None, self.poly2_lamb, None),
                      ('CONVERGENCE_CRITERION', None, self.tol, None),
                      ('CONVERGENCE_INTERVAL', self.exit_interval, None, None),
                      ('HANDLE_MISSING', self.handle_missing, None, None)
                     ]
        if categorical_variable is not None:
            param_rows.extend(('CATEGORICAL_VARIABLE', None, None, variable)
                              for variable in categorical_variable)
        unique_id = str(uuid.uuid1()).replace('-', '_').upper()
        tables = ['META', 'COEFFICIENT', 'STATISTICS', 'CROSS_VALIDATION']
        tables = ["#PAL_FFM_{}_TBL_{}_{}".format(tbl, self.id, unique_id) for tbl in tables]
        meta_tbl, coef_tbl, stat_tbl, cross_valid_tbl = tables
        try:
            call_pal_auto(conn,
                          "PAL_FFM",
                          data_,
                          ParameterTable().with_data(param_rows),
                          *tables)
        except dbapi.Error as db_err:
            logger.exception(str(db_err))
            try_drop(conn, tables)
            raise
        self.meta_ = conn.table(meta_tbl)#pylint:disable=attribute-defined-outside-init
        self.coef_ = conn.table(coef_tbl)#pylint:disable=attribute-defined-outside-init
        self.stats_ = conn.table(stat_tbl)#pylint:disable=attribute-defined-outside-init
        self.cross_valid_ = conn.table(cross_valid_tbl)#pylint:disable=attribute-defined-outside-init
        self.model_ = [self.meta_, self.coef_]

    def _predict(self, data, key, features=None, thread_ratio=None, handle_missing=None):#pylint: disable=too-many-locals, too-many-statements, too-many-branches, too-many-arguments
        conn = data.connection_context
        require_pal_usable(conn)
        if getattr(self, 'model_') is None:
        #if not hasattr(self, 'coef_'):
            raise FitIncompleteError("Model not initialized. Perform a fit first.")
        self.handle_missing = self._arg('handle_missing', handle_missing, self.handle_missing_map)
        if self.handle_missing is None:
            self.handle_missing = self.handle_missing_map['fill_zero']
        cols = data.columns
        key = self._arg('key', key, str, required=True)
        cols.remove(key)
        if features is not None:
            if isinstance(features, str):
                features = [features]
            try:
                features = self._arg('features', features, ListOfStrings)#pylint: disable=undefined-variable
            except:
                msg = ("`features` must be list of string or string.")
                logger.error(msg)
                raise TypeError(msg)
        else:
            features = cols
        data_ = data[[key] + features]
        param_rows = [('THREAD_RATIO', None, thread_ratio, None),
                      ('HANDLE_MISSING', self.handle_missing, None, None)]
        unique_id = str(uuid.uuid1()).replace('-', '_').upper()
        res_tbl = "#PAL_FFM_RESULT_TBL_{}_{}".format(self.id, unique_id)
        try:
            call_pal_auto(conn,
                          'PAL_FFM_PREDICT',
                          data_,
                          self.model_[0],
                          self.model_[1],
                          ParameterTable().with_data(param_rows),
                          res_tbl)
        except dbapi.Error as db_err:
            logger.exception(str(db_err))
            try_drop(conn, res_tbl)
            raise
        return conn.table(res_tbl)

class FFMClassifier(_FFMBase):
    """
    class FFMClassifier
    Field-Aware Factorization Machine with the task of classification.

    Parameters
    ----------

    factor_num : int, optional
        The factorisation dimensionality.
        Default to 4.
    random_state : int, optional
        Specifies the seed for random number generator.

          - 0: Uses the current time as the seed.
          - Others: Uses the specified value as the seed.

        Default to 0.
    train_size : float, optional
        The proportation of dataset used for training, and the remaining data set for validation.

        For example, 0.8 indicates that 80% for training, and the remaining 20% for validation.

        Default to 0.8 if number of instances not less than 40, 1.0 otherwise.
    max_iter : int, optional
        Specifies the maximum number of iterations for the alternative least square algorithm.

        Default to 20
    ordering : ListOfStrings, optional
        Specifies the categories orders for ranking.

        No default value.
    normalise : bool, optional
        Specifies whether to normalise each instance so that its L1 norm is 1.

        Default to True.
    include_constant : bool, optional
        Specifies whether to include the w0 constant part.

        Default to True.
    include_linear : bool, optional
        Specifies whether to include the linear part of regression model.

        Default to True.
    early_stop : bool, optional
        Specifies whether to early stop the SGD optimisation.

        Valid only if the value of ``thread_ratio`` is less than 1.

        Default to True.
    learning_rate : float, optional
        The learning rate for SGD iteration.

        Default to 0.2.
    linear_lamb : float, optional
        The L2 regularisation parameter for the linear coefficient vector.

        Default to 1e-5.
    poly2_lamb : float, optional
        The L2 regularisation parameter for factorized coefficient matrix of the quadratic term.

        Default to 1e-5.
    tol : float, optional
        The criterion to determine the convergence of SGD.

        Default to 1e-5.
    exit_interval : int, optional
        The interval of two iterations for comparison to determine the convergence.

        Default to 5.
    handle_missing : str, optional
        Specifies how to handle missing feature:

            - 'skip': skip rows with missing values.
            - 'fill_zero': replace missing values with 0.

        Default to 'fill_zero'.

    Attributes
    ----------
    metadata_ : DataFrame
        Model metadata content.
    coef_ : DataFrame
        DataFrame that provides the following information:
            - Feature name,
            - Field name,
            - The factorisation number,
            - The parameter value.
    stats_ : DataFrame
        Statistic values.
    cross_valid_ : DataFrame
        Cross validation content.

    Examples
    --------
    Input dataframe for classification training:

    >>> df_train_classification.collect()
        USER    MOVIE                  TIMESTAMP    CTR
        A      Movie1                   3          Click
        A      Movie2                   3          Click
        A      Movie4                   1          Not click
        A      Movie5                   2          Click
        A      Movie6                   3          Click
        A      Movie8                   2          Not click
        A      Movie0, Movie3           1          Click
        B      Movie2                   3          Click
        B      Movie3                   2          Click
        B      Movie4                   2          Not click
        B      null                     4          Not click
        B      Movie7                   1          Click
        B      Movie8                   2          Not click
        B      Movie0                   3          Not click
        C      Movie1                   2          Click
        C      Movie2, Movie5, Movie7   4          Not click
        C      Movie4                   3          Not click
        C      Movie5                   1          Not click
        C      Movie6                   null       Click
        C      Movie7                   3          Not click
        C      Movie8                   1          Click
        C      Movie0                   2          Click
        D      Movie1                   3          Click
        D      Movie3                   2          Click
        D      Movie4, Movie7           2          Click
        D      Movie6                   2          Click
        D      Movie7                   4          Not click
        D      Movie8                   3          Not click
        D      Movie0                   3          Not click
        E      Movie1                   2          Not click
        E      Movie2                   2          Click
        E      Movie3                   2          Click
        E      Movie4                   4          Click
        E      Movie5                   3          Click
        E      Movie6                   2          Not click
        E      Movie7                   4          Not click
        E      Movie8                   3          Not click
    Creating FFMClassifier instance:

    >>> ffm = FFMClassifier(linear_lamb=1e-5, poly2_lamb=1e-6, random_state=1,
                  factor_num=4, early_stop=1, learning_rate=0.2, max_iter=20, train_size=0.8)

    Performing fit() on given dataframe:

    >>> ffm.fit(data=self.df_train_classification, categorical_variable='TIMESTAMP')

    >>> ffm.stats_.collect()
            STAT_NAME          STAT_VALUE
        0         task      classification
        1  feature_num                  18
        2    field_num                   3
        3        k_num                   4
        4     category    Click, Not click
        5         iter                   3
        6      tr-loss  0.6409316561278655
        7      va-loss  0.7452354780967997

    Performing predict() on given predicting dataframe:

    >>> res = ffm.predict(data=self.df_predict, key='ID', thread_ratio=1)

    >>> res.collect()
            ID      SCORE  CONFIDENCE
        0   1  Not click    0.543537
        1   2  Not click    0.545470
        2   3      Click    0.542737
        3   4      Click    0.519458
        4   5      Click    0.511001
        5   6  Not click    0.534610
        6   7      Click    0.537739
        7   8  Not click    0.536781
        8   9  Not click    0.635412
    """
    def __init__(self, ordering=None, normalise=None, include_linear=None, include_constant=None,#pylint: disable=too-many-arguments, too-many-locals, too-many-branches, too-many-statements
                 early_stop=None, random_state=None, factor_num=None, max_iter=None, train_size=None, learning_rate=None,#pylint: disable=too-many-arguments, too-many-locals, too-many-branches, too-many-statements
                 linear_lamb=None, poly2_lamb=None, tol=None, exit_interval=None, handle_missing=None):#pylint: disable=too-many-arguments, too-many-locals, too-many-branches, too-many-statements

        super(FFMClassifier, self).__init__(functionality='classification',
                                            ordering=ordering,
                                            normalise=normalise,
                                            include_linear=include_linear,
                                            include_constant=include_constant,
                                            early_stop=early_stop,
                                            random_state=random_state,
                                            factor_num=factor_num,
                                            max_iter=max_iter,
                                            train_size=train_size,
                                            learning_rate=learning_rate,
                                            linear_lamb=linear_lamb,
                                            poly2_lamb=poly2_lamb,
                                            tol=tol,
                                            exit_interval=exit_interval,
                                            handle_missing=handle_missing)

    def fit(self, data, key=None, features=None, label=None, categorical_variable=None, delimiter=None):#pylint: disable=too-many-arguments, too-many-locals, too-many-branches, too-many-statements
        """
        Fit the FFMClassifier model with the input training data. Model parameters should be given by initializing the model first.

        Parameters
        ----------
        data : DataFrame
            Data to be fit.
        key : str, optional
            Name of the ID column.
        features : str/ListOfStrings, optional
            Name of the feature columns.
        delimiter : str, optional
            The delimiter to separate string features.

            For example, "China, USA" indicates two feature values "China" and "USA".

            Default to ','.
        label : str, optional
            Secifies the dependent variable.

            For classification, the label column can be any kind of data type.

            Default to last column name.
        categorical_variable : str/ListofStrings, optional
            Indicates whether or not a column data is actually corresponding
            to a category variable even the data type of this column is INTEGER.

            By default, 'VARCHAR' or 'NVARCHAR' is category variable, and 'INTEGER'
            or 'DOUBLE' is continuous variable.
        """
        self._fit(data, key, features, label, categorical_variable, delimiter)

    def predict(self, data, key, features=None, thread_ratio=None, handle_missing=None):#pylint: disable=too-many-locals, too-many-statements, too-many-branches, too-many-arguments
        """
        Prediction for the input data with the trained FFMClassifier model.

        Parameters
        ----------
        data : DataFrame
            Data to be fit.
        key : str, optional
            Name of the ID column.
        features : str/ListOfStrings, optional
            Global side features column name in the training dataframe.
        thread_ratio : float, optional
            The ratio of available threads.

              -   0: single thread
              -   0~1: percentage
              -   Others: heuristically determined

            Default to -1.
        handle_missing : str, optional
            Specifies how to handle missing feature:

                - 'skip': skip rows with missing values.
                - 'fill_zero': replace missing values with 0.

            Default to 'fill_zero'.

        Returns
        -------
        DataFrame
            Prediction result, structured as follows:

              - 1st column : ID
              - 2nd column : SCORE, i.e. predicted class labels
              - 3rd column : CONFIDENCE, the confidence for assigning class labels.
        """
        pred_res = super(FFMClassifier, self)._predict(data, key, features, thread_ratio, handle_missing)
        return pred_res

class FFMRegressor(_FFMBase):
    """
    class FFMRegressor
    Field-Aware Factorization Machine with the task of Regression.

    Parameters
    ----------

    factor_num : int, optional
        The factorisation dimensionality.
        Default to 4.
    random_state : int, optional
        Specifies the seed for random number generator.

        -   0: Uses the current time as the seed.
        -   Others: Uses the specified value as the seed.

        Default to 0.
    train_size : float, optional
        The proportion of data used for training, and the remaining data set for validation.

        For example, 0.8 indicates that 80% for training, and the remaining 20% for validation.

        Default to 0.8 if number of instances not less than 40, 1.0 otherwise.
    max_iter : int, optional
        Specifies the maximum number of iterations for the ALS algorithm.

        Default to 20
    ordering : ListOfStrings, optional
        Specifies the categories orders for ranking.

        No default value.
    normalise : bool, optional
        Specifies whether to normalise each instance so that its L1 norm is 1.

        Default to True.
    include_constant : bool, optional
        Specifies whether to include the constant part.

        Default to True.
    include_linear : bool, optional
        Specifies whether to include the linear part of the model.

        Default to True.
    early_stop : bool, optional
        Specifies whether to early stop the SGD optimisation.

        Valid only if the value of ``train_size`` is less than 1.

        Default to True.
    learning_rate : float, optional
        The learning rate for SGD iteration.

        Default to 0.2.
    linear_lamb : float, optional
        The L2 regularisation parameter for the linear coefficient vector.

        Default to 1e-5.
    poly2_lamb : float, optional
        The L2 regularisation parameter for factorized coefficient matrix of the quadratic term.

        Default to 1e-5.

    tol : float, optional
        The criterion to determine the convergence of SGD.

        Default to 1e-5.
    exit_interval : int, optional
        The interval of two iterations for comparison to determine the convergence.

        Default to 5.
    handle_missing : str, optional
        Specifies how to handle missing feature:

            -   'skip': remove rows with missing values.
            -   'fill_zero': replace missing values with 0.

        Default to 'fill_zero'.

    Attributes
    ----------
    metadata_ : DataFrame
        Model metadata content.
    coef_ : DataFrame
        The DataFrame inclusive of the following information:
            - Feature name,
            - Field name,
            - The factorisation number,
            - The parameter value.
    stats_ : DataFrame
        Statistic values.
    cross_valid_ : DataFrame
        Cross validation content.

    Examples
    --------
    Input dataframe for regresion training:

    >>> df_train_regression.collect()
        USER    MOVIE                  TIMESTAMP    CTR
        A      Movie1                   3          0
        A      Movie2                   3          5
        A      Movie4                   1          0
        A      Movie5                   2          1
        A      Movie6                   3          2
        A      Movie8                   2          0
        A      Movie0, Movie3           1          5
        B      Movie2                   3          4
        B      Movie3                   2          4
        B      Movie4                   2          0
        B      null                     4          3
        B      Movie7                   1          4
        B      Movie8                   2          0
        B      Movie0                   3          4
        C      Movie1                   2          3
        C      Movie2, Movie5, Movie7   4          2
        C      Movie4                   3          1
        C      Movie5                   1          0
        C      Movie6                   null       5
        C      Movie7                   3          0
        C      Movie8                   1          5
        C      Movie0                   2          3
        D      Movie1                   3          0
        D      Movie3                   2          5
        D      Movie4, Movie7           2          5
        D      Movie6                   2          5
        D      Movie7                   4          0
        D      Movie8                   3          1
        D      Movie0                   3          1
        E      Movie1                   2          1
        E      Movie2                   2          5
        E      Movie3                   2          3
        E      Movie4                   4          2
        E      Movie5                   3          5
        E      Movie6                   2          0
        E      Movie7                   4          2
        E      Movie8                   3          0
    Creating FFMRegressor instance:

    >>>  ffm = FFMRegressor(factor_num=4, early_stop=True, learning_rate=0.2, max_iter=20, train_size=0.8,
                            linear_lamb=1e-5, poly2_lamb=1e-6, random_state=1)

    Performing fit() on given dataframe:

    >>> ffm.fit(data=self.df_train_regression, categorical_variable='TIMESTAMP')

    >>> ffm.stats_.collect
            STAT_NAME          STAT_VALUE
        0         task          regression
        1  feature_num                  18
        2    field_num                   3
        3        k_num                   4
        4         iter                  15
        5      tr-loss  0.4503367758101421
        6      va-loss  1.6896813062750056

    Performing predict() on given prediction dataset:

    >>> res = ffm.predict(data=self.df_predict, key='ID', thread_ratio=1)

    >>> res.collect()
           ID                SCORE CONFIDENCE
        0   1    2.978197866860172       None
        1   2  0.43883354766746385       None
        2   3    3.765106298778723       None
        3   4   1.8874204073998788       None
        4   5    3.588371752514674       None
        5   6   1.3448502862740495       None
        6   7    5.268571202934171       None
        7   8   0.8713338730015039       None
        8   9    2.347070689885986       None
    """
    def __init__(self, ordering=None, normalise=None, include_linear=None, include_constant=None,#pylint: disable=too-many-arguments, too-many-locals, too-many-branches, too-many-statements
                 early_stop=None, random_state=None, factor_num=None, max_iter=None, train_size=None, learning_rate=None,#pylint: disable=too-many-arguments, too-many-locals, too-many-branches, too-many-statements
                 linear_lamb=None, poly2_lamb=None, tol=None, exit_interval=None, handle_missing=None):#pylint: disable=too-many-arguments, too-many-locals, too-many-branches, too-many-statements

        super(FFMRegressor, self).__init__(functionality='regression',
                                           ordering=ordering,
                                           normalise=normalise,
                                           include_linear=include_linear,
                                           include_constant=include_constant,
                                           early_stop=early_stop,
                                           random_state=random_state,
                                           factor_num=factor_num,
                                           max_iter=max_iter,
                                           train_size=train_size,
                                           learning_rate=learning_rate,
                                           linear_lamb=linear_lamb,
                                           poly2_lamb=poly2_lamb,
                                           tol=tol,
                                           exit_interval=exit_interval,
                                           handle_missing=handle_missing)

    def fit(self, data, key=None, features=None, label=None, categorical_variable=None, delimiter=None):#pylint: disable=too-many-arguments, too-many-locals, too-many-branches, too-many-statements
        """
        Fit the FFMRegressor model with the input training data. Model parameters should be given by initializing the model first.

        Parameters
        ----------
        data : DataFrame
            Data to be fit.
        key : str, optional
            Name of the ID column.
        features : str/ListOfStrings, optional
            Name of the feature columns.
        delimiter : str, optional
            The delimiter to separate string features.

            For example, "China, USA" indicates two feature values "China" and "USA".

            Default to ','.
        label : str, optional
            Secifies the dependent variable.

            For regression, the label column must have numerical data type.

            Default to last column name.
        categorical_variable : str/ListofStrings, optional
            Indicates whether or not a column data is actually corresponding
            to a category variable even the data type of this column is INTEGER.

            By default, 'VARCHAR' or 'NVARCHAR' is category variable, and 'INTEGER'
            or 'DOUBLE' is continuous variable.
        """
        self._fit(data, key, features, label, categorical_variable, delimiter)

    def predict(self, data, key, features=None, thread_ratio=None, handle_missing=None):#pylint: disable=too-many-locals, too-many-statements, too-many-branches, too-many-arguments
        """
        Prediction for the input data with the trained FFMRegressor model.

        Parameters
        ----------
        data : DataFrame
            Data to be fit.
        key : str, optional
            Name of the ID column.
        features : str/ListOfStrings, optional
            Global side features column name in the training dataframe.
        thread_ratio : float, optional
            The ratio of available threads.

              - 0: single thread
              - 0~1: percentage
              - Others: heuristically determined

            Default to -1.
        handle_missing : {'skip', 'fill_zero'}, optional
            Specifies how to handle missing feature:

              - 'skip': remove rows with missing values.
              - 'fill_zero': replace missing values with 0.

            Default to 'fill_zero'.

        Returns
        -------
        DataFrame
            Prediction result, structured as follows:

              - 1st column : ID
              - 2nd column : SCORE, i.e. predicted values
              - 3rd column : CONFIDENCE, all NULLs.
        """
        pred_res = super(FFMRegressor, self)._predict(data, key, features, thread_ratio, handle_missing)
        return pred_res

class FFMRanker(_FFMBase):
    """
    class FFMRanker
    Field-Aware Factorization Machine with the task of ranking.

    Parameters
    ----------

    factor_num : int, optional
        The factorisation dimensionality.

        Default to 4.
    random_state : int, optional
        Specifies the seed for random number generator.

          - 0: Uses the current time as the seed.
          - Others: Uses the specified value as the seed.

        Default to 0.
    train_size : float, optional
        The porportaion of data used for training, and the remaining data set for validation.

        For example, 0.8 indicates that 80% for training, and the remaining 20% for validation.

        Default to 0.8 if number of instances not less than 40, 1.0 otherwise.
    max_iter : int, optional
        Specifies the maximum number of iterations for the ALS algorithm.

        Default to 20.
    ordering : ListOfStrings, optional
        Specifies the categories orders for ranking.

        No default value.
    normalise : bool, optional
        Specifies whether to normalise each instance so that its L1 norm is 1.

        Default to True.
    include_linear : bool, optional
        Specifies whether to include the the linear part of the model.

        Default to True.
    early_stop : bool, optional
        Specifies whether to early stop the SGD optimisation.

        Valid only if the value of ``train_size`` is less than 1.

        Default to True.
    learning_rate : float, optional
        The learning rate for SGD iteration.

        Default to 0.2.
    linear_lamb : float, optional
        The L2 regularisation parameter for the linear coefficient vector.

        Default to 1e-5.
    poly2_lamb : float, optional
        The L2 regularisation parameter for factorized coefficient matrix of the quadratic term.

        Default to 1e-5.
    tol : float, optional
        The criterion to determine the convergence of SGD.

        Default to 1e-5.
    exit_interval : int, optional
        The interval of two iterations for comparison to determine the convergence.

        Default to 5.
    handle_missing : {'skip', 'fill_zero'}, optional
        Specifies how to handle missing feature:

          - 'skip': remove rows with missing values.
          - 'fill_zero': replace missing values with 0.

        Default to 'fill_zero'.

    Attributes
    ----------
    metadata_ : DataFrame
        Model metadata content.
    coef_ : DataFrame
        The DataFrame inclusive of the following information:
            - Feature name,
            - Field name,
            - The factorisation number,
            - The parameter value.
    stats_ : DataFrame
        Statistic values.
    cross_valid_ : DataFrame
        Cross validation content.

    Examples
    --------
    Input dataframe for regresion training:

    >>> df_train_ranker.collect()
        USER    MOVIE                  TIMESTAMP    CTR
          A     Movie1                   3          medium
          A     Movie2                   3          too high
          A     Movie4                   1          medium
          A     Movie5                   2          too low
          A     Movie6                   3          low
          A     Movie8                   2          low
          A     Movie0, Movie3           1          too high
          B     Movie2                   3          high
          B     Movie3                   2          high
          B     Movie4                   2          medium
          B     null                     4          medium
          B     Movie7                   1          high
          B     Movie8                   2          high
          B     Movie0                   3          high
          C     Movie1                   2          medium
          C     Movie2, Movie5, Movie7   4          low
          C     Movie4                   3          too low
          C     Movie5                   1          high
          C     Movie6                   null       too high
          C     Movie7                   3          high
          C     Movie8                   1          too high
          C     Movie0                   2          medium
          D     Movie1                   3          too high
          D     Movie3                   2          too high
          D     Movie4, Movie7           2          too high
          D     Movie6                   2          too high
          D     Movie7                   4          too high
          D     Movie8                   3          too low
          D     Movie0                   3          too low
          E     Movie1                   2          too low
          E     Movie2                   2          too high
          E     Movie3                   2          medium
          E     Movie4                   4          low
          E     Movie5                   3          too high
          E     Movie6                   2          low
          E     Movie7                   4          low
          E     Movie8                   3          too low

    Creating FFMRanker instance:

    >>>  ffm = FFMRanker(ordering=['too low', 'low', 'medium', 'high', 'too high'],
                         factor_num=4, early_stop=True, learning_rate=0.2, max_iter=20, train_size=0.8,
                         linear_lamb=1e-5, poly2_lamb=1e-6, random_state=1)

    Performing fit() on given dataframe:

    >>> ffm.fit(data=self.df_train_rank, categorical_variable='TIMESTAMP')

    >>> ffm.stats_.collect()
            STAT_NAME                            STAT_VALUE
        0         task                               ranking
        1  feature_num                                    18
        2    field_num                                     3
        3        k_num                                     4
        4     category  too low, low, medium, high, too high
        5         iter                                    14
        6      tr-loss                    1.3432013591533276
        7      va-loss                    1.5509792122994928

    Performing predict() on given predicting dataframe:

    >>> res = ffm.predict(data=self.df_predict, key='ID', thread_ratio=1)

    >>> res.collect()
           ID     SCORE  CONFIDENCE
        0   1      high    0.294206
        1   2    medium    0.209893
        2   3   too low    0.316609
        3   4      high    0.219671
        4   5  too high    0.222545
        5   6      high    0.385621
        6   7   too low    0.407695
        7   8   too low    0.295200
        8   9      high    0.282633
    """
    def __init__(self, ordering=None, normalise=None, include_linear=None,#pylint: disable=too-many-arguments, too-many-locals, too-many-branches, too-many-statements
                 early_stop=None, random_state=None, factor_num=None, max_iter=None, train_size=None, learning_rate=None,#pylint: disable=too-many-arguments, too-many-locals, too-many-branches, too-many-statements
                 linear_lamb=None, poly2_lamb=None, tol=None, exit_interval=None, handle_missing=None):#pylint: disable=too-many-arguments, too-many-locals, too-many-branches, too-many-statements

        super(FFMRanker, self).__init__(functionality='ranking',
                                        ordering=ordering,
                                        normalise=normalise,
                                        include_linear=include_linear,
                                        include_constant=True,
                                        early_stop=early_stop,
                                        random_state=random_state,
                                        factor_num=factor_num,
                                        max_iter=max_iter,
                                        train_size=train_size,
                                        learning_rate=learning_rate,
                                        linear_lamb=linear_lamb,
                                        poly2_lamb=poly2_lamb,
                                        tol=tol,
                                        exit_interval=exit_interval,
                                        handle_missing=handle_missing)

    def fit(self, data, key=None, features=None, label=None, categorical_variable=None, delimiter=None):#pylint: disable=too-many-locals, too-many-statements, too-many-branches, too-many-arguments
        """
        Fit the FFMRanker model with the input training data. Model parameters should be given by initializing the model first.

        Parameters
        ----------
        data : DataFrame
            Data to be fit.
        key : str, optional
            Name of the ID column.
        features : str/ListOfStrings, optional
            Name of the feature columns.
        delimiter : str, optional
            The delimiter to separate string features.

            For example, "China, USA" indicates two feature values "China" and "USA".

            Default to ','.
        label : str, optional
            Secifies the dependent variable.

            For ranking, the label column must have categorical data type.

            Default to last column name.
        categorical_variable : str/ListofStrings, optional
            Indicates whether or not a column data is actually corresponding
            to a category variable even the data type of this column is INTEGER.

            By default, 'VARCHAR' or 'NVARCHAR' is category variable, and 'INTEGER'
            or 'DOUBLE' is continuous variable.
        """
        self._fit(data, key, features, label, categorical_variable, delimiter)

    def predict(self, data, key, features=None, thread_ratio=None, handle_missing=None):#pylint: disable=too-many-locals, too-many-statements, too-many-branches, too-many-arguments
        """
        Prediction for the input data with the trained FFMRanker model.

        Parameters
        ----------
        data : DataFrame
            Data to be fit.
        key : str, optional
            Name of the ID column.
        features : str/ListOfStrings, optional
            Global side features column name in the training dataframe.
        thread_ratio : float, optional
            The ratio of available threads.

              - 0: single thread
              - 0~1: percentage
              - Others: heuristically determined

            Default to -1.
        handle_missing : str, optional
            Specifies how to handle missing feature:

              - 'skip': remove rows with missing values.
              - 'fill_zero': replace missing values with 0.

            Default to 'fill_zero'.

        Returns
        -------
        DataFrame
            Prediction result, structured as follows:

            -   1st column : ID
            -   2nd column : SCORE, i.e. predicted ranking
            -   3rd column : CONFIDENCE, the confidence for ranking.
        """
        pred_res = super(FFMRanker, self)._predict(data, key, features, thread_ratio, handle_missing)
        return pred_res
