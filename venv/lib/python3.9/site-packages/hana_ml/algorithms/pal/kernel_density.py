#pylint:disable=too-many-lines, too-many-locals
'''
This module contains PAL wrappers for kernel density estimation.

The following class is available:

    * :class:`KDE`
'''

import logging
import uuid
from hdbcli import dbapi
from hana_ml.ml_exceptions import FitIncompleteError#pylint:disable=ungrouped-imports
from hana_ml.ml_base import try_drop
from hana_ml.algorithms.pal.pal_base import (
    PALBase,
    ParameterTable,
    ListOfStrings,
    require_pal_usable,
    call_pal_auto
)
logger = logging.getLogger(__name__)#pylint: disable=invalid-name

class KDE(PALBase):#pylint:disable=too-few-public-methods,too-many-instance-attributes
    '''
    Perform Kernel Density to analogue with histograms whereas getting rid of its defects.

    Parameters
    ----------

    thread_ratio : float, optional
        Specifies the ratio of total number of threads that can be used by this function.

        The value range is from 0 to 1, where 0 means only using 1 thread, and 1 means using
        at most all the currently available threads.

        Values outside the range will be ignored and this function heuristically
        determines the number of threads to use.

        Default to 0.0.
    leaf_size : int, optional
        Number of samples in a KD tree or Ball tree leaf node.

        Only Valid when ``algorithm`` is 'kd-tree' or 'ball-tree'.

        Default to 30.
    kernel : {'gaussian', 'tophat', 'epanechnikov', 'exponential', 'linear', 'cosine'}, optional
        Kernel function type.

        Default to 'gaussian'.
    method : {'brute_force', 'kd_tree', 'ball_tree'}, optional(deprecated)
        Searching method.

        Default to 'brute_force'
    algorithm : {'brute-force', 'kd-tree', 'ball-tree'}, optional
        Specifies the searching method.

        Default to 'brute-force'.
    bandwidth : float, optional
        Bandwidth used during density calculation.

        0 means providing by optimizer inside, otherwise bandwidth is provided by end users.

        Only valid when data is one dimensional.

        Default to 0.
    distance_level : {'manhattan', 'euclidean', 'minkowski', 'chebyshev'}, optional
        Computes the distance between the train data and the test data point.

        Default to 'eculidean'.
    minkowski_power : float, optionl
        When you use the Minkowski distance, this parameter controls the value of power.

        Only valid when ``distance_level`` is 'minkowski'.

        Default to 3.0.
    rtol : float, optional
        The desired relative tolerance of the result.

        A larger tolerance generally leads to faster execution.

        Default to 1e-8.
    atol : float, optional
        The desired absolute tolerance of the result.

        A larger tolerance generally leads to faster execution.

        Default to 0.
    stat_info : bool, optional

        - False: STATISTIC table is empty

        - True: Statistic information is displayed in the STATISTIC table.

        Only valid when parameter selection is not specified.
    resampling_method : {'loocv'}, optional
        Specifies the resampling method for model evaluation or parameter selection,
        only 'loocv' is permitted.

        Once set, the mood is set to KDE_CROSS_VALIDATAION.

        ``evaluation_metric`` must be set together.

        No default value.
    evaluation_metric : {'nll'}, optional
        Specifies the evaluation metric for model evaluation or parameter selection,
        only 'nll' is supported.

        Should only be set when in KDE_CROSS_VALIDATAION mood.

        No default value.
    search_strategy : {'grid', 'random'}, optional
        Specifies the method to activate parameter selection.

        No default value.
    repeat_times : int, optional
        Specifies the number of repeat times for resampling.

        Default to 1.
    random_state : int, optional
        Specifies the seed for random generation. Use system time when 0 is specified.

        Default to 0.
    bandwidth_values : list, optional
        Specifies values of parameter ``bandwidth`` to be selected.

        Only valid when parameter selection is enabled.
    bandwidth_range : list, optional
        Specifies ranges of parameter ``bandwidth`` to be selected.

        Only valid when parameter selection is enabled.

    Attributes
    ----------
    stats_ : DataFrame
        Statistical info for model evaluation.
        Available only when model evaluation/parameter selection is triggered.

    optim_param_ : DataFrame
        Optimal parameters selected.
        Available only when model evaluation/parameter selection is triggered.

    Examples
    --------
    Data used for fitting a kernel density function:

    >>> df_train.collect()
            ID        X1        X2
        0   0 -0.425770 -1.396130
        1   1  0.884100  1.381493
        2   2  0.134126 -0.032224
        3   3  0.845504  2.867921
        4   4  0.288441  1.513337
        5   5 -0.666785  1.244980
        6   6 -2.102968 -1.428327
        7   7  0.769902 -0.473007
        8   8  0.210291  0.328431
        9   9  0.482323 -0.437962

    Data used for density value prediction:

    >>> df_pred.collect()
       ID        X1        X2
    0   0 -2.102968 -1.428327
    1   1 -2.102968  0.719797
    2   2 -2.102968  2.867921
    3   3 -0.609434 -1.428327
    4   4 -0.609434  0.719797
    5   5 -0.609434  2.867921
    6   6  0.884100 -1.428327
    7   7  0.884100  0.719797
    8   8  0.884100  2.867921

    Construct KDE instance:

    >>> kde = KDE(leaf_size=10, method='kd_tree', bandwidth=0.68129, stat_info=True)

    Fit a kernel density function:

    >>> kde.fit(data=df_train, key='ID')

    Peroform density prediction and check the results

    >>> res, stats = kde.predict(data=df_pred, key='ID')
    >>> res.collect()
       ID  DENSITY_VALUE
    0   0      -3.324821
    1   1      -5.733966
    2   2      -8.372878
    3   3      -3.123223
    4   4      -2.772520
    5   5      -4.852817
    6   6      -3.469782
    7   7      -2.556680
    8   8      -3.198531

    >>> stats_.collect()
       TEST_ID                            FITTING_IDS
    0        0  {"fitting ids":[0,1,2,3,4,5,6,7,8,9]}
    1        1  {"fitting ids":[0,1,2,3,4,5,6,7,8,9]}
    2        2  {"fitting ids":[0,1,2,3,4,5,6,7,8,9]}
    3        3  {"fitting ids":[0,1,2,3,4,5,6,7,8,9]}
    4        4  {"fitting ids":[0,1,2,3,4,5,6,7,8,9]}
    5        5  {"fitting ids":[0,1,2,3,4,5,6,7,8,9]}
    6        6  {"fitting ids":[0,1,2,3,4,5,6,7,8,9]}
    7        7  {"fitting ids":[0,1,2,3,4,5,6,7,8,9]}
    8        8  {"fitting ids":[0,1,2,3,4,5,6,7,8,9]}
    '''
    def __init__(self,#pylint:disable=too-many-arguments, too-many-locals, too-many-branches, too-many-statements
                 thread_ratio=None,
                 leaf_size=None,
                 kernel=None,
                 method=None,
                 distance_level=None,
                 minkowski_power=None,
                 atol=None,
                 rtol=None,
                 bandwidth=None,
                 resampling_method=None,
                 evaluation_metric=None,
                 bandwidth_values=None,
                 bandwidth_range=None,
                 stat_info=None,
                 random_state=None,
                 search_strategy=None,
                 repeat_times=None,
                 algorithm=None):
        super(KDE, self).__init__()
        self.key = None
        self.features = None
        self._training_data = None
        self.resampling_method_map = {'loocv': 'loocv'}
        self.evaluation_metric_map = {'nll': 'NLL'}
        self.search_strategy_map = {'grid': 'grid', 'random': 'random'}
        self.kernel_map = {'gaussian': 0, 'tophat': 1, 'epanechnikov': 2,
                           'exponential': 3, 'linear': 4, 'cosine': 5}
        self.method_map = {'brute_force': 0, 'brute-force': 0,
                           'kd_tree': 1, 'kd-tree': 1,
                           'ball_tree': 2, 'ball-tree': 2}
        self.distance_level_map = {'manhattan': 1, 'euclidean': 2, 'minkowski': 3, 'chebyshev': 4}
        self.thread_ratio = self._arg('thread_ratio', thread_ratio, float)
        self.leaf_size = self._arg('leaf_size', leaf_size, int)
        self.kernel = self._arg('kernel', kernel, self.kernel_map)
        self.method = self._arg('method', method, self.method_map)
        self.algorithm = self._arg('algorithm', algorithm, self.method_map)
        self.distance_level = self._arg('distance_level', distance_level, self.distance_level_map)
        self.minkowski_power = self._arg('minkowski_power', minkowski_power, float)
        self.atol = self._arg('atol', atol, float)
        self.rtol = self._arg('rtol', rtol, float)
        self.resampling_method = self._arg('resampling_method', resampling_method, self.resampling_method_map)#pylint:disable=line-too-long
        self.evaluation_metric = self._arg('evaluation_metric', evaluation_metric, self.evaluation_metric_map)#pylint:disable=line-too-long
        self.bandwidth = self._arg('bandwidth', bandwidth, float)
        self.bandwidth_values = self._arg('bandwidth_values', bandwidth_values, list)
        self.bandwidth_range = self._arg('bandwidth_range', bandwidth_range, list)
        self.stat_info = self._arg('stat_info', stat_info, bool)
        self.repeat_times = self._arg('repeat_times', repeat_times, int)
        self.search_strategy = self._arg('search_strategy', search_strategy, self.search_strategy_map)#pylint:disable=line-too-long
        self.random_state = self._arg('random_state', random_state, int)
        cv_count = 0
        for val in (self.resampling_method, self.evaluation_metric):
            if val is not None:
                cv_count += 1
        if cv_count not in (0, 2):
            msg = ("'resampling_method' and 'evaluation_metric' must be set together.")
            logger.error(msg)
            raise ValueError(msg)
        if self.resampling_method is None and self.search_strategy is not None:
            msg = ("'search_strategy' is invalid when resampling_method is not set.")
            logger.error(msg)
            raise ValueError(msg)
        if self.search_strategy is None:
            if self.bandwidth_values is not None:
                msg = ("'bandwidth_values' can only be specified "+
                       "when parameter selection is enabled.")
                logger.error(msg)
                raise ValueError(msg)
            if self.bandwidth_range is not None:
                msg = ("'bandwidth_range' can only be specified "+
                       "when parameter selection is enabled.")
                logger.error(msg)
                raise ValueError(msg)
        if self.search_strategy is not None:
            bandwidth_set_count = 0
            for bandwidth_set in (self.bandwidth, self.bandwidth_range, self.bandwidth_values):
                if bandwidth_set is not None:
                    bandwidth_set_count += 1
            if bandwidth_set_count > 1:
                msg = ("The following paramters cannot be specified together:" +
                       "'bandwidth', 'bandwidth_values', 'bandwidth_range'.")
                logger.error(msg)
                raise ValueError(msg)
            if self.bandwidth_values is not None:
                if not all(isinstance(t, (int, float)) for t in self.bandwidth_values):
                    msg = "Valid values of `bandwidth_values` must be a list of numerical values."
                    logger.error(msg)
                    raise TypeError(msg)

            if self.bandwidth_range is not None:
                rsz = [3] if self.search_strategy == 'grid'else [2, 3]
                if not len(self.bandwidth_range) in rsz or not all(isinstance(t, (int, float)) for t in self.bandwidth_range):#pylint:disable=line-too-long
                    msg = ("The provided `bandwidth_range` is either not "+
                           "a list of numerical values, or it contains wrong number of values.")
                    logger.error(msg)
                    raise TypeError(msg)

    def fit(self, data, key, features=None):#pylint:disable=too-many-statements, too-many-branches
        '''
        If parameter selection / model evaluation is enabled, perform it.
        Otherwise, just setting the training data set.

        Parameters
        ----------
        data : DataFrame
            Dataframe including the data of density distribution.
        key : str
            Name of the ID column in the dataframe.
        features : str/list of str, optional
            Name of the feature columns in the dataframe.

        Attributes
        ----------
        _training_data : DataFrame
            The traninig data for kernel density function fitting.
        '''
        conn = data.connection_context
        require_pal_usable(conn)
        self.key = self._arg('key', key, str)
        cols = data.columns
        cols.remove(self.key)
        if features is not None:
            if isinstance(features, str):
                features = [features]
            try:
                self.features = self._arg('features', features, ListOfStrings)#pylint: disable=undefined-variable
            except:
                msg = ("'features' must be list of string or string.")
                logger.error(msg)
                raise TypeError(msg)
        else:
            self.features = cols

        data_ = data[[self.key] + self.features]
        self._training_data = data_
        if self.resampling_method is not None:
            param_rows = [('THREAD_RATIO', None, self.thread_ratio, None),
                          ('BUCKET_SIZE', self.leaf_size, None, None),
                          ('KERNEL', self.kernel, None, None),
                          ('METHOD',
                           self.method if self.algorithm is None else self.algorithm,
                           None, None),
                          ('DISTANCE_LEVEL', self.distance_level, None, None),
                          ('MINKOWSKI_POWER', None, self.minkowski_power, None),
                          ('ABSOLUTE_RESULT_TOLERANCE', None, self.atol, None),
                          ('RELATIVE_RESULT_TOLERANCE', None, self.rtol, None),
                          ('RESAMPLING_METHOD', None, None, self.resampling_method),
                          ('EVALUATION_METRIC', None, None, self.evaluation_metric),
                          ('SEED', self.random_state, None, None),
                          ('REPEAT_TIMES', self.repeat_times, None, None),
                          ('PARAM_SEARCH_STRATEGY', None, None, self.search_strategy)]
            if self.bandwidth is not None:
                param_rows.extend([('BANDWIDTH', None, self.bandwidth, None)])
            if self.bandwidth_range is not None:
                val = str(self.bandwidth_range).replace('[', '[').replace(']', ']')
                param_rows.extend([('BANDWIDTH_RANGE', None, None, val)])
            if self.bandwidth_values is not None:
                val = str(self.bandwidth_values).replace('[', '{').replace(']', '}')
                param_rows.extend([('BANDWIDTH_VALUES', None, None, val)])
            unique_id = str(uuid.uuid1()).replace('-', '_').upper()
            tables = ['STATISTICS', 'OPTIMAL_PARAM']#pylint:disable=line-too-long
            tables = ["#PAL_KDE_CV_{}_TBL_{}_{}".format(tbl, self.id, unique_id) for tbl in tables]
            stats_tbl, optim_param_tbl = tables
            try:
                call_pal_auto(conn,
                              'PAL_KDE_CV',
                              data_,
                              ParameterTable().with_data(param_rows),
                              *tables)
            except dbapi.Error as db_err:
                logger.exception(str(db_err))
                try_drop(conn, tables)
                raise
            self.stats_ = conn.table(stats_tbl)#pylint:disable=attribute-defined-outside-init
            self.optim_param_ = conn.table(optim_param_tbl)#pylint:disable=attribute-defined-outside-init
            if(self.bandwidth_range is not None or self.bandwidth_values is not None):
                self.bandwidth = self.optim_param_.select('DOUBLE_VALUE').head().collect()['DOUBLE_VALUE'][0]#pylint:disable=line-too-long

    def predict(self, data, key, features=None):#pylint:disable=too-many-statements, too-many-branches
        '''
        Apply kernel density analysis.

        Parameters
        ----------
        data : DataFrame
            Dataframe including the data of density prediction.
        key : str
            Column of IDs of the data points for density prediction.

        Returns
        -------
        DataFrame
          - Result data table, i.e. predicted log-density values on all points in ``data``.

          - Statistics information table which reflects the support of prediction points
            over all training points.
        '''
        conn = data.connection_context
        require_pal_usable(conn)
        if self._training_data is None:
            raise FitIncompleteError("Model not initialized. Perform a fit first.")
        key = self._arg('key', key, str)
        cols = data.columns
        cols.remove(key)
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
        if len(features) != len(self.features):
            msg = ("Selected feature column number in training data"+
                   " and prediction data must be the same.")
            logger.error(msg)
            raise ValueError(msg)
        data_ = data[[key] + features]
        param_rows = [('THREAD_RATIO', None, self.thread_ratio, None),
                      ('BUCKET_SIZE', self.leaf_size, None, None),
                      ('KERNEL', self.kernel, None, None),
                      ('METHOD', self.method, None, None),
                      ('DISTANCE_LEVEL', self.distance_level, None, None),
                      ('MINKOWSKI_POWER', None, self.minkowski_power, None),
                      ('ABSOLUTE_RESULT_TOLERANCE', None, self.atol, None),
                      ('RELATIVE_RESULT_TOLERANCE', None, self.rtol, None),
                      ('BANDWIDTH', None, self.bandwidth, None),
                      ('STAT_INFO', self.stat_info, None, None)]
        unique_id = str(uuid.uuid1()).replace('-', '_').upper()
        tables = ['RESULT', 'STATISTICS']
        tables = ["#PAL_KDE_{}_TBL_{}_{}".format(tbl, self.id, unique_id) for tbl in tables]
        try:
            call_pal_auto(conn,
                          'PAL_KDE',
                          self._training_data,
                          data_,
                          ParameterTable().with_data(param_rows),
                          *tables)
        except dbapi.Error as db_err:
            logger.exception(str(db_err))
            try_drop(conn, tables)
            raise
        return conn.table(tables[0]), conn.table(tables[1])
