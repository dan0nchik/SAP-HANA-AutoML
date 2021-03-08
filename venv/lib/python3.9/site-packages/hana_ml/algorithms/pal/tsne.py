"""
This module contains Python API of PAL
T-distributed Stochastic Neighbour Embedding algorithm.
The following classes are available:
    * :class:`TSNE`
"""
import logging
import uuid
from hdbcli import dbapi
from .pal_base import (
    PALBase,
    ParameterTable,
    ListOfStrings,
    try_drop,
    require_pal_usable,
    call_pal_auto
)

logger = logging.getLogger(__name__)#pylint: disable=invalid-name

class TSNE(PALBase):#pylint: disable=too-many-instance-attributes, too-few-public-methods
    r"""
    Class for T-distributed Stochastic Neighbour Embedding.

    Parameters
    ----------
    thread_ratio : float, optional
        The ratio of available threads.

            - 0 : single thread
            - 0~1 : percentage

        Defaults to 0.0.
    n_iter : int, optional
        Specifies the maximum number of iterations for the TSNE algorithm.

        Default to 250.
    random_state : int, optional
        The seed for random number generate.

        Default to 0.
    exaggeration : float, optional
        Value to be multiplied on pij before 250 iterations.

        The natural clusters would be more separated with larger value, which means
        there would be more empty space on the map.

        Default to 12.0.
    angle : float, optional
        The legal value should be between 0.0 to 1.0.

		Setting it to 0.0 means using the "exact" method which would run :math:`O(N^2)` time,
		otherwise TSNE would employ Barnes-Hut approximation which would run :math:`O(N*log{N})`.

        This value is a tradeoff between accuracy and training speed for Barnes-Hut
        approximation.

        The training speed would be faster with higher value.

        Default to 0.5.
    n_components : int, optional
        Dimension of the embedded space.

		Values other than 2 and 3 are illegal.

        Default to 2.
    object_frequency : int, optional
        Frequency of calculating the objective function and putting the result
        into OBJECTIVES table.

        This parameter value should not be larger than the value assigned to ``n_iter``.

        Default to 50.
    learning_rate : float, optional
        Learning rate.

        Default to 200.0.
    perplexity : float, optional
        The perplexity is related to the number of nearest neighbors and  mentioned
        above.

        Larger value is suitable for large dataset.

        Make sure ``preplexity`` * 3 < [no. of samples]

        Default to 30.0.

    Examples
    --------
    Input dataframe for fit and predict:

    >>> df_train.collect()
           ID  ATT1  ATT2  ATT3  ATT4  ATT5
        0   1   1.0   2.0 -10.0 -20.0   3.0
        1   2   4.0   5.0 -30.0 -10.0   6.0
        2   3   7.0   8.0 -40.0 -50.0   9.0
        3   4  10.0  11.0 -25.0 -15.0  12.0
        4   5  13.0  14.0 -12.0 -24.0  15.0
        5   6  16.0  17.0  -9.0 -13.0  18.0

    Creating TSNE instance:

    >>> tsne = TSNE(self.conn, n_iter=500, n_components=3, angle=0,
                    object_frequency=50, random_state=30)

    Performing fit_predict() on given dataframe:

    >>> res, stats, obj = tsne.fit_predict(data=self.df_train, key='ID', perplexity=1.0)

    >>> res.collect()
            ID           x           y           z
        0   1    4.875853 -189.090497 -229.536424
        1   2  -67.675459  213.661740  178.397623
        2   3  -68.852910  162.710853  284.966271
        3   4  -68.056108  193.118052  220.275439
        4   5   76.524624 -189.850926 -227.625750
        5   6  123.184000 -190.549221 -226.477160

    >>> stats.collect()
           STAT_NAME           STAT_VALUE
        0     method                exact
        1       iter                  500
        2  objective  0.12310845438143747

    >>> obj.collect()
           ITER  OBJ_VALUE
        0    50  50.347530
        1   100  50.982194
        2   150  49.368419
        3   200  70.201283
        4   250  63.717535
        5   300   1.296687
        6   350   0.882636
        7   400   0.260532
        8   450   0.174178
        9   500   0.123108
    """
    def __init__(self,#pylint: disable=too-many-arguments, too-many-locals, too-many-branches, too-many-statements
                 n_iter=None,
                 learning_rate=None,
                 object_frequency=None,
                 n_components=None,
                 angle=None,
                 exaggeration=None,
                 thread_ratio=None,
                 random_state=None,
                 perplexity=None):
        super(TSNE, self).__init__()
        self.n_iter = self._arg('n_iter', n_iter, int)
        self.learning_rate = self._arg('learning_rate', learning_rate, float)
        self.object_frequency = self._arg('object_frequency', object_frequency, int)
        if self.object_frequency is not None:
            if self.n_iter is not None and self.n_iter < self.object_frequency:
                msg = ("'object_frequency' should not exceed the value of 'n_iter'")
                logger.error(msg)
                raise ValueError(msg)
        self.n_components = self._arg('n_components', n_components, int)
        if self.n_components is not None and self.n_components not in (2, 3):
            msg = ("'n_components' of the embedded space cannot have a value other than 2 or 3.")
            logger.error(msg)
            raise ValueError(msg)
        self.angle = self._arg('angle', angle, float)
        self.exaggeration = self._arg('exaggeration', exaggeration, float)
        self.thread_ratio = self._arg('thread_ratio', thread_ratio, float)
        self.random_state = self._arg('random_state', random_state, int)
        self.perplexity = self._arg('perplexity', perplexity, float)

    def fit_predict(self, data, key, features=None):
        """
        Fit the TSNE model with input data.
        Model parameters should be given by initializing the model first.

        Parameters
        ----------
        data : DataFrame
            Data to be fit.
        key : str, optional
            Name of the ID column.
        features : ListofStrings/str, optional
            Name of the features column.

            If not specifited, the feature columns should be all
            columns in the input dataframe except the key column.

        Returns
        -------
        DataFrame
            - Result table with coordinate value of different dimensions.
            - Table of statistical values.
            - Table of objective values of iterations.
        """
        conn = data.connection_context
        require_pal_usable(conn)
        key = self._arg('key', key, str)
        data_ = data
        if features is not None:
            if isinstance(features, str):
                features = [features]
            try:
                features = self._arg('features', features, ListOfStrings)#pylint: disable=undefined-variable
                data_ = data[[key] + features]
            except:
                msg = ("'features' must be list of string or string.")
                logger.error(msg)
                raise TypeError(msg)
        if self.perplexity is not None and not self.perplexity * 3 < int(data.count()):
            msg = ("'Perplexity' * 3 must be less than number of samples in input dataframe.")
            logger.error(msg)
            raise TypeError(msg)
        param_rows = [('THREAD_RATIO', None, self.thread_ratio, None),
                      ('SEED', self.random_state, None, None),
                      ('MAX_ITER', self.n_iter, None, None),
                      ('EXAGGERATION', None, self.exaggeration, None),
                      ('PERPLEXITY', None, self.perplexity, None),
                      ('THETA', None, self.angle, None),
                      ('NO_DIM', self.n_components, None, None),
                      ('OBJ_FREQ', self.object_frequency, None, None),
                      ('ETA', None, self.learning_rate, None)
                     ]

        unique_id = str(uuid.uuid1()).replace('-', '_').upper()
        tables = ['RESULT', 'STATISTICS', 'OBJECTIVES']
        tables = ["#PAL_TSNE_{}_TBL_{}_{}".format(tbl, self.id, unique_id) for tbl in tables]
        res_tbl, stats_tbl, obj_tbl = tables
        try:
            call_pal_auto(conn,
                          "PAL_TSNE",
                          data_,
                          ParameterTable().with_data(param_rows),
                          *tables)
        except dbapi.Error as db_err:
            logger.exception(str(db_err))
            try_drop(conn, tables)
            raise
        return conn.table(res_tbl), conn.table(stats_tbl), conn.table(obj_tbl)#pylint: disable=line-too-long
