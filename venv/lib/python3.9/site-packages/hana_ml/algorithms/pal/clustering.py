"""
This module contains Python wrappers for PAL clustering algorithms.

The following classes are available:

    * :class:`AffinityPropagation`
    * :class:`AgglomerateHierarchicalClustering`
    * :class:`DBSCAN`
    * :class:`GeometryDBSCAN`
    * :class:`KMeans`
    * :class:`KMedians`
    * :class:`KMedoids`
    * :func:`SlightSilhouette`
"""
#pylint: disable=too-many-lines, too-many-arguments, invalid-name, unused-variable, too-many-locals
#pylint: disable=line-too-long, bad-option-value, too-few-public-methods, useless-object-inheritance
#pylint: disable=relative-beyond-top-level, attribute-defined-outside-init, too-many-instance-attributes
import logging
import uuid

from hdbcli import dbapi
from hana_ml.algorithms.pal.sqlgen import trace_sql
from hana_ml.ml_exceptions import FitIncompleteError
from .pal_base import (
    PALBase,
    ParameterTable,
    arg,
    try_drop,
    pal_param_register,
    colspec_from_df,
    ListOfStrings,
    require_pal_usable,
    call_pal_auto
)

logger = logging.getLogger(__name__)

class _ClusterAssignmentMixin(object):

    @trace_sql
    def _predict(self, data, key, features=None):
        r"""
        Assign clusters to data based on a fitted model.

        The output structure of this method does not match that of
        fit_predict().

        Parameters
        ----------

        data : DataFrame

            Data points to match against computed clusters.

            This dataframe's column structure should match that
            of the data used for fit().

        key : str

            Name of the ID column.

        features : list of str, optional.

            Names of the feature columns.
            if ``features`` is not provided, it defaults to all
            the non-ID columns.

        Returns
        -------

        DataFrame
            Cluster assignment results, with 3 columns:

              - Data point ID, with name and type taken from the input
                ID column.
              - CLUSTER_ID, INTEGER type, representing the cluster the
                data point is assigned to.
              - DISTANCE, DOUBLE type, representing the distance between
                the data point and the cluster center (for K-means), the
                nearest core object (for DBSCAN), or the weight vector
                (for SOM).
        """
        conn = data.connection_context
        if self.model_ is None:
            raise FitIncompleteError('Model not initialized. Perform a fit first.')
        cols = data.columns
        cols.remove(key)
        if features is None:
            features = cols
        if key in features:
            message = "The key column can't also be a feature."
            logger.error(message)
            raise ValueError(message)
        adjusted_data = data[[key] + features]

        unique_id = str(uuid.uuid1()).replace('-', '_').upper()
        assignment_tbl = '#PAL_CLUSTER_ASSIGNMENT_ASSIGNMENT_TBL_{}_{}'.format(self.id, unique_id)

        try:
            call_pal_auto(conn,
                          'PAL_CLUSTER_ASSIGNMENT',
                          adjusted_data,
                          self.model_,
                          ParameterTable(),
                          assignment_tbl)
        except dbapi.Error as db_err:
            logger.exception(str(db_err))
            try_drop(conn, assignment_tbl)
            raise

        return conn.table(assignment_tbl)

def SlightSilhouette(data,
                     features=None,
                     label=None,
                     distance_level=None,
                     minkowski_power=None,
                     normalization=None,
                     thread_number=None,
                     categorical_variable=None,
                     category_weights=None):
    r"""
    Silhouette refers to a method used to validate the cluster of data.
    SAP HNAN PAL provides a light version of sihouette called slight sihouette. SlightSihouette
    is an wrapper for this light version sihouette method. \n
    Note that this function is a new function in SAP HANA SPS05 and Cloud.

    Parameters
    ----------

    data : DataFrame
        DataFrame containing the data.
    features : list of str, optional
        Names of the feature columns.

        If ``features`` is not provided, it defaults to all non-label columns.
    label: str, optional
         Name of the ID column.

         If ``label`` is not provided, it defaults to last column.
    distance_level : {'manhattan', 'euclidean', 'minkowski', 'chebyshev', 'cosine'} str, optional
        Ways to compute the distance between the item and the cluster center.
        'cosine' is only valid when ``accelerated`` is False.

        Defaults to 'euclidean'.
    minkowski_power : float, optional
        When Minkowski distance is used, this parameter controls the
        value of power.

        Only valid when ``distance_level`` is minkowski.

        Defaults to 3.0.
    normalization : {'no', 'l1_norm', 'min_max'}, optional
        Normalization type.

          - 'no': No normalization will be applied.
          - 'l1_norm': Yes, for each point X (x\ :sub:`1`\, x\ :sub:`2`\, ..., x\ :sub:`n`\), the normalized
            value will be X'(x\ :sub:`1` /S,x\ :sub:`2` /S,...,x\ :sub:`n` /S),
            where S = \|x\ :sub:`1`\ \|+\|x\ :sub:`2`\ \|+...\|x\ :sub:`n`\ \|.
          - 'min_max': Yes, for each column C, get the min and max value of C,
            and then C[i] = (C[i]-min)/(max-min).

        Defaults to 'no'.
    thread_number : int, optional
        Number of threads.

        Defaults to 1.
    categorical_variable : str or list of str, optional
        Indicates whether or not a column data is actually corresponding to a category variable even the data type of this column is INTEGER.

        By default, VARCHAR or NVARCHAR is category variable, and INTEGER or DOUBLE is continuous variable.

        Defaults to None.
    category_weights : float, optional
        Represents the weight of category attributes.

        Defaults to 0.707.

    Returns
    -------
    DataFrame
    Returns a DataFrame containing the validation value of Slight Silhouette.


    Examples
    --------

    Input dataframe df:

    >>> df.collect()
        V000 V001 V002 CLUSTER
     1   0.5    A  0.5       0
     2   1.5    A  0.5       0
     3   1.5    A  1.5       0
     4   0.5    A  1.5       0
     5   1.1    B  1.2       0
     6   0.5    B 15.5       1
     7   1.5    B 15.5       1
     8   1.5    B 16.5       1
     9   0.5    B 16.5       1
     10  1.2    C 16.1       1
     11 15.5    C 15.5       2
     12 16.5    C 15.5       2
     13 16.5    C 16.5       2
     14 15.5    C 16.5       2
     15 15.6    D 16.2       2
     16 15.5    D  0.5       3
     17 16.5    D  0.5       3
     18 16.5    D  1.5       3
     19 15.5    D  1.5       3
     20 15.7    A  1.6       3

    Call the function:

    >>> res = SlightSilhouette(df, label="CLUSTER")

    Result:

    >>> print(res.collect())
         VALIDATE_VALUE
    1    0.9385944
    """
    distance_map = {'manhattan':1, 'euclidean':2, 'minkowski':3, 'chebyshev':4}
    distance_level = arg('distance_level', distance_level, distance_map)
    if distance_level != 3 and minkowski_power is not None:
        msg = 'Minkowski_power will only be valid if distance_level is Minkowski.'
        logger.error(msg)
        raise ValueError(msg)
    minkowski_power = arg('minkowski_power', minkowski_power, int)
    normalization_map = {'no' : 0, 'l1.norm' : 1, 'min.max' : 2}
    normalization = arg('normalization', normalization, normalization_map)
    thread_number = arg('thread_number', thread_number, int)

    category_weights = arg('category_weights', category_weights, float)

    #handle CATEGORY_COL, transform from categorical_variable
    if categorical_variable is not None:
        column_choose = None
        if isinstance(categorical_variable, str):
            categorical_variable = [categorical_variable]
        categorical_variable = arg('categorical_variable', categorical_variable, ListOfStrings)
        try:
            column_choose = list()
            for var in categorical_variable:
                column_choose.append(data.columns.index(var))
        except:
            msg = ("Not all categorical_variable is in the features!")
            logger.error(msg)
            raise TypeError(msg)

    conn = data.connection_context
    require_pal_usable(conn)
    label = arg('label', label, str)
    features = arg('features', features, ListOfStrings)

    cols = data.columns
    if label is None:
        label = cols[-1]
    cols.remove(label)
    if features is None:
        features = cols

    data = data[features + [label]]
    unique_id = str(uuid.uuid1()).replace('-', '_').upper()
    result_tbl = '#PAL_SLIGHT_SIL_RESULT_TBL_{}_{}'.format(id, unique_id)

    param_rows = [('DISTANCE_LEVEL', distance_level, None, None),
                  ('MINKOWSKI_POWER', None, minkowski_power, None),
                  ('NORMALIZATION', normalization, None, None),
                  ('THREAD_NUMBER', thread_number, None, None),
                  ('CATEGORY_WEIGHTS', None, category_weights, None)]
    if categorical_variable is not None:
        param_rows.extend([('CATEGORY_COL', col, None, None)
                           for col in column_choose])

    try:
        call_pal_auto(conn,
                      'PAL_SLIGHT_SILHOUETTE',
                      data,
                      ParameterTable().with_data(param_rows),
                      result_tbl)

    except dbapi.Error as db_err:
        logger.exception(str(db_err))
        try_drop(conn, result_tbl)
        raise

    return conn.table(result_tbl)

class AffinityPropagation(PALBase):
    r"""
    Affinity Propagation is an algorithm that identifies exemplars among data points and forms clusters of
    data points around these exemplars. It operates by simultaneously considering all data point as
    potential exemplars and exchanging messages between data points until a good set of exemplars and clusters emerges.

    Parameters
    ----------

    affinity : {'manhattan', 'standardized_euclidean', 'euclidean', 'minkowski', 'chebyshev', 'cosine'}

        Ways to compute the distance between two points.

        No default value as it is mandatory.

    n_clusters :  int
        Number of clusters.

          - 0: does not adjust Affinity Propagation cluster result.
          - Non-zero int: If Affinity Propagation cluster number is bigger than ``n_clusters``,
            PAL will merge the result to make the cluster number be the value specified for ``n_clusters``.

        No default value as it is mandatory.

    max_iter : int, optional

        Maximum number of iterations.

        Defaults to 500.

    convergence_iter : int, optional

        When the clusters keep a steady one for the specified times, the algorithm ends.

        Defaults to 100.

    damping : float

        Controls the updating velocity. Value range: (0, 1).

        Defaults to 0.9.

    preference : float, optional
        Determines the preference. Value range: [0,1].

        Defaults to 0.5.

    seed_ratio : float, optional

        Select a portion of (seed_ratio * data_number) the input data as seed,
        where data_number is the row_size of the input data.

        Value range: (0,1].

        If ``seed_ratio`` is 1, all the input data will be the seed.

        Defaults to 1.

    times : int, optional

        The sampling times. Only valid when seed_ratio is less than 1.

        Defaults to 1.

    minkowski_power : int, optional

        The power of the Minkowski method. Only valid when affinity is 3.

        Defaults to 3.

    thread_ratio : float, optional

        Specifies the ratio of total number of threads that can be used by this function.

        The value range is from 0 to 1, where 0 means only using 1 thread, and 1 means using at most all the currently available threads.

        Values outside the range will be ignored and this function heuristically determines the number of threads to use.

        Defaults to 0.

    Attributes
    ----------

    labels_ : DataFrame

        Label assigned to each sample. structured as follows:

            - ID, record ID.
            - CLUSTER_ID, the range is from 0 to ``n_clusters`` - 1.

    Examples
    --------

    Input dataframe df for clustering:

    >>> df.collect()
        ID  ATTRIB1  ATTRIB2
    0    1   0.10     0.10
    1    2   0.11     0.10
    2    3   0.10     0.11
    3    4   0.11     0.11
    4    5   0.12     0.11
    5    6   0.11     0.12
    6    7   0.12     0.12
    7    8   0.12     0.13
    8    9   0.13     0.12
    9   10   0.13     0.13
    10  11   0.13     0.14
    11  12   0.14     0.13
    12  13  10.10    10.10
    13  14  10.11    10.10
    14  15  10.10    10.11
    15  16  10.11    10.11
    16  17  10.11    10.12
    17  18  10.12    10.11
    18  19  10.12    10.12
    19  20  10.12    10.13
    20  21  10.13    10.12
    21  22  10.13    10.13
    22  23  10.13    10.14
    23  24  10.14    10.13

    Create AffinityPropagation instance:

    >>> ap = AffinityPropagation(
                affinity='euclidean',
                n_clusters=0,
                max_iter=500,
                convergence_iter=100,
                damping=0.9,
                preference=0.5,
                seed_ratio=None,
                times=None,
                minkowski_power=None,
                thread_ratio=1)

    Perform fit on the given data:

    >>> ap.fit(data = df, key='ID')

    Expected output:

    >>> ap.labels_.collect()
        ID  CLUSTER_ID
    0    1           0
    1    2           0
    2    3           0
    3    4           0
    4    5           0
    5    6           0
    6    7           0
    7    8           0
    8    9           0
    9   10           0
    10  11           0
    11  12           0
    12  13           1
    13  14           1
    14  15           1
    15  16           1
    16  17           1
    17  18           1
    18  19           1
    19  20           1
    20  21           1
    21  22           1
    22  23           1
    23  24           1
    """
    affinity_map = {'manhattan':1, 'euclidean':2, 'minkowski':3,
                    'chebyshev':4, 'standardized_euclidean':5, 'cosine':6}

    def __init__(self,
                 affinity,
                 n_clusters,
                 max_iter=None,
                 convergence_iter=None,
                 damping=None,
                 preference=None,
                 seed_ratio=None,
                 times=None,
                 minkowski_power=None,
                 thread_ratio=None):

        super(AffinityPropagation, self).__init__()

        self.affinity = self._arg('affinity', affinity, self.affinity_map)
        self.n_clusters = self._arg('n_clusters', n_clusters, int)
        self.max_iter = self._arg('max_iter', max_iter, int)
        self.convergence_iter = self._arg('convergence_iter', convergence_iter, int)
        self.damping = self._arg('damping', damping, float)
        self.preference = self._arg('preference', preference, float)
        self.seed_ratio = self._arg('seed_ratio', seed_ratio, float)
        self.times = self._arg('times', times, float)
        self.minkowski_power = self._arg('minkowski_power', minkowski_power, int)
        self.thread_ratio = self._arg('thread_ratio', thread_ratio, int)

    def fit(self, data, key, features=None):
        """
        Fit the model when given the training dataset.

        Parameters
        ----------

        data : DataFrame

            DataFrame containing the data.

        key : str

            Name of the ID column.

        features : list of str, optional

            Names of the features columns.
            If ``features`` is not provided, it defaults to all the non-ID columns.
        """
        conn = data.connection_context
        require_pal_usable(conn)
        key = self._arg('key', key, str, True)
        features = self._arg('features', features, ListOfStrings)

        cols = data.columns
        cols.remove(key)
        if features is None:
            features = cols
        data_ = data[[key] + features]

        unique_id = str(uuid.uuid1()).replace('-', '_').upper()
        outputs = ['#AFFINITY_PROPAGATION_{}_TBL_{}_{}'.format(name, self.id, unique_id)
                   for name in ['RESULT', 'STAT']]
        result_tbl, stats_tbl = outputs
        seed_tbl = '#AFFINITY_PROPAGATION_SEED_TBL_{}_{}'.format(self.id, unique_id)

        param_rows = [('DISTANCE_METHOD', self.affinity, None, None),
                      ('CLUSTER_NUMBER', self.n_clusters, None, None),
                      ('MAX_ITERATION', self.max_iter, None, None),
                      ('CON_ITERATION', self.convergence_iter, None, None),
                      ('DAMP', None, self.damping, None),
                      ('PREFERENCE', None, self.preference, None),
                      ('SEED_RATIO', None, self.seed_ratio, None),
                      ('TIMES', None, self.times, None),
                      ('MINKOW_P', self.minkowski_power, None, None),
                      ('THREAD_RATIO', None, self.thread_ratio, None)]

        with conn.connection.cursor() as cur:
            cur.execute('CREATE LOCAL TEMPORARY COLUMN TABLE {} ("ID" INTEGER, "SEED_ID" INTEGER)'.format(seed_tbl))
            seed = conn.table(seed_tbl)

        try:
            call_pal_auto(conn,
                          'PAL_AFFINITY_PROPAGATION',
                          data_,
                          seed,
                          ParameterTable().with_data(param_rows),
                          *outputs)
        except dbapi.Error as db_err:
            logger.exception(str(db_err))
            try_drop(conn, outputs)
            raise

        self.labels_ = conn.table(result_tbl)

    def fit_predict(self, data, key, features=None):
        """
        Fit with the dataset and return the labels.

        Parameters
        ----------

        data : DataFrame

            DataFrame containing the data.

        key : str

            Name of the ID column.

        features : list of str, optional

            Names of the features columns.

            If ``features`` is not provided, it defaults to all the non-ID
            columns.

        Returns
        -------

        DataFrame

            Fit result, label of each points, structured as follows:

              - ID, record ID.
              - CLUSTER_ID, the range is from 0 to ``n_clusters`` - 1.
        """
        self.fit(data, key, features)
        return self.labels_

class AgglomerateHierarchicalClustering(PALBase):
    r"""
    This algorithm is a widely used clustering method \
    which can find natural groups within a set of data. The idea is to group the data into \
    a hierarchy or a binary tree of the subgroups. A hierarchical clustering can be either \
    agglomerate or divisive, depending on the method of hierarchical decomposition.
    The implementation in PAL follows the agglomerate approach, which merges the clusters \
    with a bottom-up strategy. Initially, each data point is considered as an own cluster.
    The algorithm iteratively merges two clusters based on the dissimilarity measure in \
    a greedy manner and forms a larger cluster.

    Parameters
    ----------

    n_clusters : int, optional

        Number of clusters after agglomerate hierarchical clustering algorithm.
        Value range: between 1 and the initial number of input data.

        Defaults to 1.

    affinity : {'manhattan', 'euclidean', 'minkowski', 'chebyshev', 'cosine', 'pearson correlation', 'squared euclidean', 'jaccard', 'gower'}, optional

        Ways to compute the distance between two points.

        .. Note ::

            - (1) For jaccard distance, non-zero input data will be treated as 1,
              and zero input data will be treated as 0.
              jaccard distance = (M01 + M10) / (M11 + M01 + M10)

            - (2) Only gower distance supports category attributes.
              When linkage is 'centroid clustering', 'median clustering', or 'ward',
              this parameter must be set to 'squared euclidean'.

        Defaults to 'squared euclidean'.

    linkage : { 'nearest neighbor', 'furthest neighbor', 'group average', 'weighted average', 'centroid clustering', 'median clustering', 'ward'}, optional

        Linkage type between two clusters.

            - 'nearest neighbor' : single linkage.
            - 'furthest neighbor' : complete linkage.
            - 'group average' : UPGMA.
            - 'weighted average' : WPGMA.
            - 'centroid clustering'.
            - 'median clustering'.
            - 'ward'.

        Defaults to centroid clustering.

            .. note::
                For linkage 'centroid clustering', 'median clustering', or 'ward',
                the corresponding affinity must be set to 'squared euclidean'.

    thread_ratio : float, optional

        Specifies the ratio of total number of threads that can be used by this function.

        The value range is from 0 to 1, where 0 means only using 1 thread, and 1 means using at most all the currently available threads.

        Values outside the range will be ignored and this function heuristically determines the number of threads to use.

        Defaults to 0.

    distance_dimension : float, optional

        Distance dimension can be set if affinity is set to 'minkowski'. The value should be no less than 1.

        Only valid when affinity is 'minkowski'.

        Defaults to 3.

    normalization : str, optional

        Specifies the type of normalization applied.

            - 'no': No normalization
            - 'z-score': Z-score standardization
            - 'zero-centred-min-max': Zero-centred min-max normalization, transforming to new range [-1, 1].
            - 'min-max': Standard min-max normalization, transforming to new range [0, 1].

        Defaults to 'no'.

    category_weights : float, optional

        Represents the weight of category columns.

        Defaults to 1.

    Attributes
    ----------

    combine_process_ : DataFrame
        Structured as follows:

              - 1st column: int, STAGE, cluster stage.
              - 2nd column: ID (in input table) data type, LEFT\_ + ID (in input table) column + name,
                One of the clusters that is to be combined in one combine stage, name as its row number in the input data table.
                After the combining, the new cluster is named after the left one.
              - 3rd column: ID (in input table) data type, RIGHT\_ + ID (in input table) column name,
                The other cluster to be combined in the same combine stage, named as its row number in the input data table.
              - 4th column: float, DISTANCE. Distance between the two combined clusters.

    labels_ : DataFrame
        Label assigned to each sample. structured as follows:

              - 1st column: ID, record ID.
              - 2nd column: CLUSTER_ID, cluster number after applying the hierarchical agglomerate algorithm.

    Examples
    --------

    Input dataframe df for clustering:

    >>> df.collect()
         POINT   X1    X2      X3
    0    0       0.5   0.5     1
    1    1       1.5   0.5     2
    2    2       1.5   1.5     2
    3    3       0.5   1.5     2
    4    4       1.1   1.2     2
    5    5       0.5   15.5    2
    6    6       1.5   15.5    3
    7    7       1.5   16.5    3
    8    8       0.5   16.5    3
    9    9       1.2   16.1    3
    10   10      15.5  15.5    3
    11   11      16.5  15.5    4
    12   12      16.5  16.5    4
    13   13      15.5  16.5    4
    14   14      15.6  16.2    4
    15   15      15.5  0.5     4
    16   16      16.5  0.5     1
    17   17      16.5  1.5     1
    18   18      15.5  1.5     1
    19   19      15.7  1.6     1

    Create an AgglomerateHierarchicalClustering instance:

    >>> hc = AgglomerateHierarchicalClustering(
                 n_clusters=4,
                 affinity='Gower',
                 linkage='weighted average',
                 thread_ratio=None,
                 distance_dimension=3,
                 normalization='no',
                 category_weights= 0.1)

    Perform fit on the given data:

    >>> hc.fit(data=df, key='POINT', categorical_variable=['X3'])

    Expected output:

    >>> hc.combine_process_.collect().head(3)
         STAGE    LEFT_POINT   RIGHT_POINT    DISTANCE
    0    1        18           19             0.0187
    1    2        13           14             0.0250
    2    3        7            9              0.0437

    >>> hc.labels_.collect().head(3)
               POINT    CLUSTER_ID
         0     0        1
         1     1        1
         2     2        1

    """

    affinity_map = {'manhattan':1, 'euclidean':2, 'minkowski':3, 'chebyshev':4,
                    'cosine':6, 'pearson correlation':7, 'squared euclidean':8,
                    'jaccard':9, 'gower':10}
    linkage_map = {'nearest neighbor':1, 'furthest neighbor':2, 'group average':3, 'weighted average':4,
                   'centroid clustering':5, 'median clustering':6, 'ward':7}
    normalization_map = {'no': 0, 'z-score': 1, 'zero-centred-min-max': 2, 'min-max': 3}

    def __init__(self,
                 n_clusters=None,
                 affinity=None,
                 linkage=None,
                 thread_ratio=None,
                 distance_dimension=None,
                 normalization=None,
                 category_weights=None):

        super(AgglomerateHierarchicalClustering, self).__init__()
        self.n_clusters = self._arg('n_clusters', n_clusters, int)
        self.affinity = self._arg('affinity', affinity, self.affinity_map)
        self.linkage = self._arg('linkage', linkage, self.linkage_map)
        linkage_range = [5, 6, 7]
        if self.linkage in linkage_range and self.affinity != 8:
            msg = ('For linkage is centroid clustering, median clustering or ward, ' +
                   'the corresponding affinity must be set to squared euclidean!')
            logger.error(msg)
            raise ValueError(msg)

        self.thread_ratio = self._arg('thread_ratio', thread_ratio, float)
        self.distance_dimension = self._arg('distance_dimension', distance_dimension, float)
        self.normalization = self._arg('normalization', normalization, (int, str))
        if isinstance(self.normalization, str):
            self.normalization = self._arg('normalization', normalization,
                                           self.normalization_map)
        self.category_weights = self._arg('category_weights', category_weights, float)

    @trace_sql
    def fit(self, data, key, features=None, categorical_variable=None):
        r"""
        Fit the model when given the training dataset.

        Parameters
        ----------

        data : DataFrame

            DataFrame containing the data.

        key : str

            Name of the ID column.

        features : list of str, optional

            Names of the features columns.
            If ``features`` is not provided, it defaults to all the non-ID columns.

        categorical_variable : str or list of str, optional

            Specifies INTEGER column(s) that should be treated as categorical.

            By default, 'VARCHAR' or 'NVARCHAR' is category variable, and 'INTEGER' or 'DOUBLE' is continuous variable.

            Defaults to None.

        """
        conn = data.connection_context
        require_pal_usable(conn)
        key = self._arg('key', key, str, True)
        features = self._arg('features', features, ListOfStrings)
        if isinstance(categorical_variable, str):
            categorical_variable = [categorical_variable]
        self.categorical_variable = self._arg('categorical_variable', categorical_variable, ListOfStrings)

        cols = data.columns
        cols.remove(key)
        if features is None:
            features = cols

        data_ = data[[key] + features]

        unique_id = str(uuid.uuid1()).replace('-', '_').upper()
        outputs = ['#HIERARCHICAL_CLUSTERING_{}_TBL_{}_{}'.format(name, self.id, unique_id) for
                   name in ['COMBINE', 'RESULT']]
        combine_process_tbl, result_tbl = outputs

        param_rows = [('CLUSTER_NUM', self.n_clusters, None, None),
                      ('DISTANCE_FUNC', self.affinity, None, None),
                      ('CLUSTER_METHOD', self.linkage, None, None),
                      ('THREAD_RATIO', None, self.thread_ratio, None),
                      ('DISTANCE_DIMENSION', None, self.distance_dimension, None),
                      ('NORMALIZE_TYPE', self.normalization, None, None),
                      ('CATEGORY_WEIGHTS', None, self.category_weights, None)]

        if self.categorical_variable is not None:
            param_rows.extend([('CATEGORICAL_VARIABLE', None, None, name)
                               for name in categorical_variable])

        try:
            call_pal_auto(conn,
                          'PAL_HIERARCHICAL_CLUSTERING',
                          data_,
                          ParameterTable().with_data(param_rows),
                          *outputs)
        except dbapi.Error as db_err:
            logger.exception(str(db_err))
            try_drop(conn, outputs)
            raise

        self.combine_process_ = conn.table(combine_process_tbl)
        self.labels_ = conn.table(result_tbl)

    def fit_predict(self, data, key, features=None, categorical_variable=None):
        r"""
        Fit with the dataset and return the labels.

        Parameters
        ----------

        data : DataFrame

            DataFrame containing the data.

        key : str

            Name of the ID column.

        features : list of str, optional

            Names of the features columns.

            If ``features`` is not provided, it defaults to all the non-ID columns.

        categorical_variable : str or list of str, optional

            Specifies INTEGER column(s) that should be treated as categorical.

            By default, 'VARCHAR' or 'NVARCHAR' is category variable, and 'INTEGER' or 'DOUBLE' is continuous variable.

            Defaults to None.

        Returns
        -------

        DataFrame

            Combine process, structured as follows:

              - 1st column: int, STAGE, cluster stage.
              - 2nd column: ID (in input table) data type, LEFT\_ + ID (in input table) column name,
                One of the clusters that is to be combined in one combine stage, name as its row number in the input data table.
                After the combining, the new cluster is named after the left one.
              - 3rd column: ID (in input table) data type, RIGHT\_ + ID (in input table) column name,
                The other cluster to be combined in the same combine stage, named as its row number in the input data table.
              - 4th column: float, DISTANCE. Distance between the two combined clusters.

            Label of each points, structured as follows:

              - 1st column: ID (in input table) data type, ID, record ID.
              - 2nd column: int, CLUSTER_ID, the range is from 0 to ``n_clusters`` - 1.
        """
        self.fit(data, key, features, categorical_variable)
        return self.labels_

class DBSCAN(PALBase, _ClusterAssignmentMixin):
    r"""
    DBSCAN (Density-Based Spatial Clustering of Applications with Noise) is
    a density-based data clustering algorithm that finds a number of clusters
    starting from the estimated density distribution of corresponding nodes.

    Parameters
    ----------

    minpts : int, optional

        The minimum number of points required to form a cluster.

            .. note ::

                ``minpts`` and ``eps`` need to be provided together by
                user or these two parameters are automatically determined.

    eps : float, optional

        The scan radius.

            .. note::

                ``minpts`` and ``eps`` need to be provided together
                by user or these two parameters are automatically determined.

    thread_ratio : float, optional

        Controls the proportion of available threads to use.

        The value range is from 0 to 1, where 0 indicates a single thread,
        and 1 indicates up to all available threads.

        Values between 0 and 1 will use that percentage of available threads.

        Values outside this range tell PAL to heuristically determine the number of threads to use.

        Defaults to heuristically determined.

    metric : {'manhattan', 'euclidean', 'minkowski', 'chebyshev', 'standardized_euclidean', 'cosine'}, optional

        Ways to compute the distance between two points.

        Defaults to 'euclidean'.

    minkowski_power : int, optional

        When minkowski is choosed for ``metric``, this parameter
        controls the value of power.
        Only applicable when ``metric`` is minkowski.

        Defaults to 3.

    categorical_variable : str or list of str, optional

        Specifies column(s) in the data that should be treated as categorical.

        Defaults to None.

    category_weights : float, optional

        Represents the weight of category attributes.

        Defaults to 0.707.

    algorithm : {'brute-force', 'kd-tree'}, optional

        Ways to search for neighbours.

        Defaults to 'kd-tree'.

    save_model : bool, optional

        If true, the generated model will be saved.

        ``save_model`` must be True in order to call predict().

        Defaults to True.

    Attributes
    ----------

    labels_ : DataFrame

        Label assigned to each sample.

    model_ : DataFrame

        Model content. Set to None if  ``save_model`` is False.

    Examples
    --------

    Input dataframe df for clustering:

    >>> df.collect()
        ID     V1     V2 V3
    0    1   0.10   0.10  B
    1    2   0.11   0.10  A
    2    3   0.10   0.11  C
    3    4   0.11   0.11  B
    4    5   0.12   0.11  A
    5    6   0.11   0.12  E
    6    7   0.12   0.12  A
    7    8   0.12   0.13  C
    8    9   0.13   0.12  D
    9   10   0.13   0.13  D
    10  11   0.13   0.14  A
    11  12   0.14   0.13  C
    12  13  10.10  10.10  A
    13  14  10.11  10.10  F
    14  15  10.10  10.11  E
    15  16  10.11  10.11  E
    16  17  10.11  10.12  A
    17  18  10.12  10.11  B
    18  19  10.12  10.12  B
    19  20  10.12  10.13  D
    20  21  10.13  10.12  F
    21  22  10.13  10.13  A
    22  23  10.13  10.14  A
    23  24  10.14  10.13  D
    24  25   4.10   4.10  A
    25  26   7.11   7.10  C
    26  27  -3.10  -3.11  C
    27  28  16.11  16.11  A
    28  29  20.11  20.12  C
    29  30  15.12  15.11  A

    Create DSBCAN instance:

    >>> dbscan = DBSCAN(thread_ratio=0.2, metric='manhattan')

    Perform fit on the given data:

    >>> dbscan.fit(data=df, key='ID')

    Expected output:

    >>> dbscan.labels_.collect()
        ID  CLUSTER_ID
    0    1           0
    1    2           0
    2    3           0
    3    4           0
    4    5           0
    5    6           0
    6    7           0
    7    8           0
    8    9           0
    9   10           0
    10  11           0
    11  12           0
    12  13           1
    13  14           1
    14  15           1
    15  16           1
    16  17           1
    17  18           1
    18  19           1
    19  20           1
    20  21           1
    21  22           1
    22  23           1
    23  24           1
    24  25          -1
    25  26          -1
    26  27          -1
    27  28          -1
    28  29          -1
    29  30          -1
    """
    metric_map = {'manhattan':1, 'euclidean':2, 'minkowski':3, 'chebyshev':4,
                  'standardized_euclidean':5, 'cosine':6}
    algorithm_map = {'brute-force':0, 'kd-tree':1}
    def __init__(self, minpts=None, eps=None, thread_ratio=None,
                 metric=None, minkowski_power=None, categorical_variable=None,
                 category_weights=None, algorithm=None, save_model=True):
        super(DBSCAN, self).__init__()
        setattr(self, 'hanaml_parameters', pal_param_register())
        self.auto_param = 'true' if any(param is None for param in (minpts, eps)) else 'false'
        if (minpts is not None and eps is None) or (minpts is None and eps is not None):
            msg = 'minpts and eps need to be provided together.'
            logger.error(msg)
            raise ValueError(msg)
        self.minpts = self._arg('minpts', minpts, int)
        self.eps = self._arg('eps', eps, float)
        self.thread_ratio = self._arg('thread_ratio', thread_ratio, float)
        self.metric = self._arg('metric', metric,
                                self.metric_map)
        if self.metric != 3 and minkowski_power is not None:
            msg = 'minkowski_power will only be applicable if metric is minkowski.'
            logger.error(msg)
            raise ValueError(msg)
        self.minkowski_power = self._arg('minkowski_power', minkowski_power, int)
        if isinstance(categorical_variable, str):
            categorical_variable = [categorical_variable]
        self.categorical_variable = self._arg('categorical_variable', categorical_variable,
                                              ListOfStrings)
        self.category_weights = self._arg('category_weights', category_weights, float)
        self.algorithm = self._arg('algorithm', algorithm, self.algorithm_map)
        self.save_model = self._arg('save_model', save_model, bool)
        self.string_variable = None
        self.variable_weight = None

    def _check_variable_weight(self, variable_weight):
        self.variable_weight = self._arg('variable_weight', variable_weight, dict)
        for key, value in  self.variable_weight.items():
            if not isinstance(key, str):
                msg = ("The key of variable_weight must be a string!")
                logger.error(msg)
                raise TypeError(msg)
            if not isinstance(value, (float, int)):
                msg = ("The value of variable_weight must be a float!")
                logger.error(msg)
                raise TypeError(msg)

    @trace_sql
    def fit(self, data, key, features=None, categorical_variable=None, string_variable=None, variable_weight=None):
        """
        Fit the DBSCAN model when given the training dataset.

        Parameters
        ----------

        data : DataFrame
            DataFrame containing the data.

        key : str
            Name of the ID column.

        features : list of str, optional
            Names of the features columns.

            If ``features`` is not provided, it defaults to all the non-ID
            columns.

        categorical_variable : str or list of str, optional
            Specifies INTEGER column(s) that should be treated as categorical.

            Other INTEGER columns will be treated as continuous.

            Defaults to None.
        string_variable : str or list of str, optional
            Indicates a string column storing not categorical data.

            Levenshtein distance is used to calculate similarity between two strings. Ignored if it is not a string column.

            Defaults to None.
        variable_weight : dict, optional
            Specifies the weight of a variable participating in distance calculation.

            The value must be greater or equal to 0. Defaults to 1 for variables not specified.

            Defaults to None.
        """
        conn = data.connection_context
        require_pal_usable(conn)
        key = self._arg('key', key, str, True)
        features = self._arg('features', features, ListOfStrings)
        if isinstance(categorical_variable, str):
            categorical_variable = [categorical_variable]
        categorical_variable = self._arg('categorical_variable', categorical_variable, ListOfStrings)
        if string_variable is not None:
            if isinstance(string_variable, str):
                string_variable = [string_variable]
            try:
                self.string_variable = self._arg('string_variable',
                                                 string_variable, ListOfStrings)
            except:
                msg = ("`string_variable` must be list of strings or string.")
                logger.error(msg)
                raise TypeError(msg)
        if variable_weight is not None:
            self._check_variable_weight(variable_weight)

        cols = data.columns
        cols.remove(key)
        if features is None:
            features = cols

        unique_id = str(uuid.uuid1()).replace('-', '_').upper()
        outputs = ['#DBSCAN_{}_TBL_{}_{}'.format(name, self.id, unique_id) for
                   name in ['RESULT', 'MODEL', 'STAT', 'PL']]
        result_tbl, model_tbl = outputs[:2]
        param_rows = [('AUTO_PARAM', None, None, self.auto_param),
                      ('MINPTS', self.minpts, None, None),
                      ('RADIUS', None, self.eps, None),
                      ('THREAD_RATIO', None, self.thread_ratio, None),
                      ('DISTANCE_METHOD', self.metric, None, None),
                      ('MINKOW_P', self.minkowski_power, None, None),
                      ('CATEGORY_WEIGHTS', None, self.category_weights, None),
                      ('METHOD', self.algorithm, None, None),
                      ('SAVE_MODEL', self.save_model, None, None)]
        if self.categorical_variable is not None:
            param_rows.extend([('CATEGORICAL_VARIABLE', None, None, name)
                               for name in self.categorical_variable])
        if self.string_variable is not None:
            param_rows.extend(('STRING_VARIABLE', None, None, variable)
                              for variable in self.string_variable)
        if self.variable_weight is not None:
            param_rows.extend(('VARIABLE_WEIGHT', None, value, key)
                              for key, value in self.variable_weight.items())

        try:
            call_pal_auto(conn,
                          'PAL_DBSCAN',
                          data[[key] + features],
                          ParameterTable().with_data(param_rows),
                          *outputs)
        except dbapi.Error as db_err:
            logger.exception(str(db_err))
            try_drop(conn, outputs)
            raise

        self.labels_ = conn.table(result_tbl)
        self.model_ = conn.table(model_tbl) if self.save_model else None

    def fit_predict(self, data, key, features=None, categorical_variable=None, string_variable=None, variable_weight=None):
        """
        Fit with the dataset and return the labels.

        Parameters
        ----------

        data : DataFrame
            DataFrame containing the data.

        key : str
            Name of the ID column.

        features : str or list of str, optional
            Names of the features columns.

            If ``features`` is not provided, it defaults to all the non-ID
            columns.

        categorical_variable : str or list of str, optional

            Specifies INTEGER column(s) that should be treated as categorical.

            Other INTEGER columns will be treated as continuous.

            Defaults to None.
        string_variable : str or list of str, optional
            Indicates a string column storing not categorical data.

            Levenshtein distance is used to calculate similarity between two strings. Ignored if it is not a string column.

            Defaults to None.
        variable_weight : dict, optional
            Specifies the weight of a variable participating in distance calculation.

            The value must be greater or equal to 0.

            Defaults to 1 for variables not specified.

            Defaults to None.

        Returns
        -------

        DataFrame

            Fit result, structured as follows:

              - ID column, with the same name and type as ``data`` 's ID column.
              - CLUSTER_ID, type INTEGER, cluster ID assigned to the data
                point. (Cluster IDs range from 0 to 1 less than the number of
                clusters. A cluster ID of -1 means the point is labeled
                as noise.)
        """
        self.fit(data, key, features, categorical_variable, string_variable, variable_weight)
        return self.labels_

    @trace_sql
    def predict(self, data, key, features=None):
        """
        Assign clusters to data based on a fitted model.
        The output structure of this method does not match that of fit_predict().

        Parameters
        ----------

        data : DataFrame

            Data points to match against computed clusters.

            This dataframe's column structure should match that
            of the data used for fit().

        key : str

            Name of the ID column.
        features : list of str, optional.

            Names of the feature columns.

            If ``features`` is not provided, it defaults to all
            the non-ID columns.

        Returns
        -------

        DataFrame

            Cluster assignment results, with 3 columns:

                - Data point ID, with name and type taken from the input
                  ID column.
                - CLUSTER_ID, type INTEGER, representing the cluster the
                  data point is assigned to.
                - DISTANCE, type DOUBLE, representing the distance between
                  the data point and the nearest core point.
        """
        return super(DBSCAN, self)._predict(data, key, features)

class GeometryDBSCAN(PALBase):
    r"""
    This function is a geometry version of DBSCAN, which only accepts geometry points as input data.
    Currently it only accepts 2-D points.

    Parameters
    ----------

    minpts : int, optional

        The minimum number of points required to form a cluster.

        .. note ::

            ``minpts`` and ``eps`` need to be provided together by user or
            these two parameters are automatically determined.

    eps : float, optional

        The scan radius.

        .. note ::

            ``minpts`` and ``eps`` need to be provided together by user or
            these two parameters are automatically determined.

    thread_ratio : float, optional

        Controls the proportion of available threads to use.

        The value range is from 0 to 1, where 0 indicates a single thread,
        and 1 indicates up to all available threads.

        Values between 0 and 1 will use that percentage of available threads.

        Values outside this range tell PAL to heuristically determine the number of threads to use.

        Defaults to -1.

    metric : {'manhattan', 'euclidean','minkowski',

              'chebyshev', 'standardized_euclidean', 'cosine'}, optional

        Ways to compute the distance between two points.

        Defaults to 'euclidean'.

    minkowski_power : int, optional

        When minkowski is choosed for ``metric``, this parameter
        controls the value of power.

        Only applicable when ``metric`` is 'minkowski'.

        Defaults to 3.

    algorithm : {'brute-force', 'kd-tree'}, optional

        Ways to search for neighbours.

        Defaults to 'kd-tree'.

    save_model : bool, optional

        If true, the generated model will be saved.

        ``save_model`` must be True in order to call predict().

        Defaults to True.

    Attributes
    ----------

    labels_ : DataFrame

        Label assigned to each sample.

    model_ : DataFrame

        Model content. Set to None if ``save_model`` is False.

    Examples
    --------

    In SAP HANA, the test table PAL_GEO_DBSCAN_DATA_TBL can be created by the following SQL::

      CREATE COLUMN TABLE PAL_GEO_DBSCAN_DATA_TBL (
        "ID" INTEGER,
        "POINT" ST_GEOMETRY);

    Then, input dataframe df for clustering:

    >>> df = conn.table("PAL_GEO_DBSCAN_DATA_TBL")

    Create DSBCAN instance:

    >>> geo_dbscan = GeometryDBSCAN(thread_ratio=0.2, metric='manhattan')

    Perform fit on the given data:

    >>> geo_dbscan.fit(data = df, key='ID')

    Expected output:

    >>> geo_dbscan.labels_.collect()
         ID   CLUSTER_ID
    0    1    0
    1    2    0
    2    3    0
    ......
    28   29  -1
    29   30  -1

    >>> geo_dbsan.model_.collect()
        ROW_INDEX    MODEL_CONTENT
    0      0         {"Algorithm":"DBSCAN","Cluster":[{"ClusterID":...


    Perform fit_predict on the given data:

    >>> result = geo_dbscan.fit_predict(df, key='ID')

    Expected output:

    >>> result.collect()
         ID   CLUSTER_ID
    0    1    0
    1    2    0
    2    3    0
    ......
    28    29  -1
    29    30  -1

    """
    metric_map = {'manhattan':1, 'euclidean':2, 'minkowski':3, 'chebyshev':4,
                  'standardized_euclidean':5, 'cosine':6}
    algorithm_map = {'brute-force':0, 'kd-tree':1}
    def __init__(self, minpts=None, eps=None, thread_ratio=None,
                 metric=None, minkowski_power=None, algorithm=None, save_model=True):
        super(GeometryDBSCAN, self).__init__()
        self.auto_param = 'true' if any(param is None for param in (minpts, eps)) else 'false'
        if (minpts is not None and eps is None) or (minpts is None and eps is not None):
            msg = 'minpts and eps need to be provided together.'
            logger.error(msg)
            raise ValueError(msg)
        self.minpts = self._arg('minpts', minpts, int)
        self.eps = self._arg('eps', eps, float)
        self.thread_ratio = self._arg('thread_ratio', thread_ratio, float)
        self.metric = self._arg('metric', metric,
                                self.metric_map)
        if self.metric != 3 and minkowski_power is not None:
            msg = 'minkowski_power will only be applicable if metric is minkowski.'
            logger.error(msg)
            raise ValueError(msg)
        self.minkowski_power = self._arg('minkowski_power', minkowski_power, int)

        self.algorithm = self._arg('algorithm', algorithm, self.algorithm_map)
        self.save_model = self._arg('save_model', save_model, bool)

    def fit(self, data, key, features=None):
        """
        Fit the Geometry DBSCAN model when given the training dataset.

        Parameters
        ----------

        data : DataFrame

            DataFrame containing the data for applying geometry DBSCAN.

            It must contain at least two columns: one ID column, and another for storing 2-D geometry points.

        key : str

            Name of the ID column.

        features : str, optional

            Name of the column for storing geometry points.

            If not provided, it defaults the first non-ID column.
        """
        conn = data.connection_context
        require_pal_usable(conn)
        key = self._arg('key', key, str, True)
        if isinstance(features, str):
            features = [features]
        features = self._arg('features', features, ListOfStrings)
        if features is not None and len(features) > 1:
            msg = "Only the column for storing 2D geometry points should be specified in features."#pylint:disable=line-too-long
            logger.error(msg)
            raise ValueError(msg)
        cols = data.columns
        cols.remove(key)
        if features is None:
            features = cols[0]

        unique_id = str(uuid.uuid1()).replace('-', '_').upper()
        outputs = ['#GEO_DBSCAN_{}_TBL_{}_{}'.format(name, self.id, unique_id) for
                   name in ['RESULT', 'MODEL', 'STAT', 'PL']]
        result_tbl, model_tbl = outputs[:2]
        param_rows = [('AUTO_PARAM', None, None, self.auto_param),
                      ('MINPTS', self.minpts, None, None),
                      ('RADIUS', None, self.eps, None),
                      ('THREAD_RATIO', None, self.thread_ratio, None),
                      ('DISTANCE_METHOD', self.metric, None, None),
                      ('MINKOW_P', self.minkowski_power, None, None),
                      ('METHOD', self.algorithm, None, None),
                      ('SAVE_MODEL', self.save_model, None, None)]

        try:
            call_pal_auto(conn,
                          'PAL_GEO_DBSCAN',
                          data[[key] + [features]],
                          ParameterTable().with_data(param_rows),
                          *outputs)
        except dbapi.Error as db_err:
            logger.exception(str(db_err))
            if 'AFL error: AFL DESCRIBE for nested call failed - invalid table(s) for ANY-procedure call (Input table 0: column 1' in str(db_err):
                msg = 'The 2nd column of data must be ST_GEOMETRY SQL type.'
                logger.error(msg)
                raise ValueError(msg)
            try_drop(conn, outputs)
            raise

        self.labels_ = conn.table(result_tbl)
        self.model_ = conn.table(model_tbl) if self.save_model else None

    def fit_predict(self, data, key, features=None):
        """
        Fit with the dataset and return the labels.

        Parameters
        ----------

        data : DataFrame

            DataFrame containing the data. The structure is as follows.

            It must contain at least two columns: one ID column, and another for storing 2-D geometry points.

        key : str

            Name of the ID column.

        features : str, optional

            Name of the column for storing 2-D geometry points.

            If not provided, it defaults to the first non-ID column.

        Returns
        -------

        DataFrame

            Fit result, structured as follows:

              - ID column, with the same name and type as ``data`` 's ID column.
              - CLUSTER_ID, type INTEGER, cluster ID assigned to the data
                point. (Cluster IDs range from 0 to 1 less than the number of
                clusters. A cluster ID of -1 means the point is labeled
                as noise.)
        """
        self.fit(data, key, features)
        return self.labels_

class KMeans(PALBase, _ClusterAssignmentMixin):
    r"""
    K-Means model that handles clustering problems.

    Parameters
    ----------

    n_clusters : int, optional

        Number of clusters. If this parameter is not specified, you must
        specify the minimum and maximum range parameters instead.

    n_clusters_min : int, optional

        Cluster range minimum.

    n_clusters_max : int, optional

        Cluster range maximum.

    init : {'first_k', 'replace', 'no_replace', 'patent'}, optional

        Controls how the initial centers are selected:

          - 'first_k': First k observations.
          - 'replace': Random with replacement.
          - 'no_replace': Random without replacement.
          - 'patent': Patent of selecting the init center (US 6,882,998 B1).

        Defaults to 'patent'.

    max_iter : int, optional

        Max iterations.

        Defaults to 100.

    thread_ratio : float, optional

        Specifies the ratio of total number of threads that can be used by this function.

        The value range is from 0 to 1, where 0 means only using 1 thread, and 1 means
        using at most all the currently available threads.

        Values outside the range will be ignored and this function heuristically determines the number of threads to use.

        Defaults to 0.

    distance_level : {'manhattan', 'euclidean', 'minkowski', 'chebyshev', 'cosine'}, optional

        Ways to compute the distance between the item and the cluster center.

        'cosine' is only valid when ``accelerated`` is False.

        Defaults to 'euclidean'.

    minkowski_power : float, optional

        When Minkowski distance is used, this parameter controls the
        value of power.

        Only valid when ``distance_level`` is 'minkowski'.

        Defaults to 3.0.

    category_weights : float, optional

        Represents the weight of category attributes.

        Defaults to 0.707.

    normalization : {'no', 'l1_norm', 'min_max'}, optional

        Normalization type.

          - 'no': No normalization will be applied.
          - 'l1_norm': Yes, for each point X (x\ :sub:`1`\, x\ :sub:`2`\, ..., x\ :sub:`n`\), the normalized
            value will be X'(x\ :sub:`1` /S,x\ :sub:`2` /S,...,x\ :sub:`n` /S),
            where S = \|x\ :sub:`1`\ \|+\|x\ :sub:`2`\ \|+...\|x\ :sub:`n`\ \|.
          - 'min_max': Yes, for each column C, get the min and max value of C,
            and then C[i] = (C[i]-min)/(max-min).

        Defaults to 'no'.

    categorical_variable : str or list of str, optional

        Specifies INTEGER column(s) in the data that should be treated as categorical.

        Defaults to None.

    tol : float, optional

        Convergence threshold for exiting iterations.

        Only valid when ``accelerated`` is False.

        Defaults to 1.0e-6.

    memory_mode : {'auto', 'optimize-speed', 'optimize-space'}, optional

        Indicates the memory mode that the algorithm uses.

          - 'auto': Chosen by algorithm.
          - 'optimize-speed': Prioritizes speed.
          - 'optimize-space': Prioritizes memory.

        Only valid when ``accelerated`` is True.

        Defaults to 'auto'.

    accelerated : bool, optional

        Indicates whether to use technology like cache to accelerate the
        calculation process:

          - If True, the calculation process will be accelerated.
          - If False, the calculation process will not be accelerated.

        Defaults to False.

    use_fast_library : bool, optional

       Use vectorized accelerated operation when it is set to 1. Not valid when accelerated is True.

       Defaults to False.

    use_float : bool, optional

       - False: double
       - True: float

       Only valid when use_fast_library is True. Not valid when accelerated is True.

       Defaults to True.

    Attributes
    ----------

    labels_ : DataFrame

        Label assigned to each sample.

    cluster_centers_ : DataFrame

        Coordinates of cluster centers.

    model_ : DataFrame

        Model content.

    statistics_ : DataFrame

        Statistic value.

    Examples
    --------

    Input dataframe df for K Means:

    >>> df.collect()
        ID  V000 V001  V002
    0    0   0.5    A   0.5
    1    1   1.5    A   0.5
    2    2   1.5    A   1.5
    3    3   0.5    A   1.5
    4    4   1.1    B   1.2
    5    5   0.5    B  15.5
    6    6   1.5    B  15.5
    7    7   1.5    B  16.5
    8    8   0.5    B  16.5
    9    9   1.2    C  16.1
    10  10  15.5    C  15.5
    11  11  16.5    C  15.5
    12  12  16.5    C  16.5
    13  13  15.5    C  16.5
    14  14  15.6    D  16.2
    15  15  15.5    D   0.5
    16  16  16.5    D   0.5
    17  17  16.5    D   1.5
    18  18  15.5    D   1.5
    19  19  15.7    A   1.6

    Create KMeans instance:

    >>> km = clustering.KMeans(n_clusters=4, init='first_k',
    ...                        max_iter=100, tol=1.0E-6, thread_ratio=0.2,
    ...                        distance_level='Euclidean',
    ...                        category_weights=0.5)

    Perform fit_predict:

    >>> labels = km.fit_predict(data=df, 'ID')
    >>> labels.collect()
        ID  CLUSTER_ID  DISTANCE  SLIGHT_SILHOUETTE
    0    0           0  0.891088           0.944370
    1    1           0  0.863917           0.942478
    2    2           0  0.806252           0.946288
    3    3           0  0.835684           0.944942
    4    4           0  0.744571           0.950234
    5    5           3  0.891088           0.940733
    6    6           3  0.835684           0.944412
    7    7           3  0.806252           0.946519
    8    8           3  0.863917           0.946121
    9    9           3  0.744571           0.949899
    10  10           2  0.825527           0.945092
    11  11           2  0.933886           0.937902
    12  12           2  0.881692           0.945008
    13  13           2  0.764318           0.949160
    14  14           2  0.923456           0.939283
    15  15           1  0.901684           0.940436
    16  16           1  0.976885           0.939386
    17  17           1  0.818178           0.945878
    18  18           1  0.722799           0.952170
    19  19           1  1.102342           0.925679

    Input dataframe df for Accelerated K-Means :

    >>> df = conn.table("PAL_ACCKMEANS_DATA_TBL")
    >>> df.collect()
        ID  V000 V001  V002
    0    0   0.5    A     0
    1    1   1.5    A     0
    2    2   1.5    A     1
    3    3   0.5    A     1
    4    4   1.1    B     1
    5    5   0.5    B    15
    6    6   1.5    B    15
    7    7   1.5    B    16
    8    8   0.5    B    16
    9    9   1.2    C    16
    10  10  15.5    C    15
    11  11  16.5    C    15
    12  12  16.5    C    16
    13  13  15.5    C    16
    14  14  15.6    D    16
    15  15  15.5    D     0
    16  16  16.5    D     0
    17  17  16.5    D     1
    18  18  15.5    D     1
    19  19  15.7    A     1

    Create Accelerated Kmeans instance:

    >>> akm = clustering.KMeans(init='first_k',
    ...                         thread_ratio=0.5, n_clusters=4,
    ...                         distance_level='euclidean',
    ...                         max_iter=100, category_weights=0.5,
    ...                         categorical_variable=['V002'],
    ...                         accelerated=True)

    Perform fit_predict:

    >>> labels = akm.fit_predict(df=data, key='ID')
    >>> labels.collect()
        ID  CLUSTER_ID  DISTANCE  SLIGHT_SILHOUETTE
    0    0           0  1.198938           0.006767
    1    1           0  1.123938           0.068899
    2    2           3  0.500000           0.572506
    3    3           3  0.500000           0.598267
    4    4           0  0.621517           0.229945
    5    5           0  1.037500           0.308333
    6    6           0  0.962500           0.358333
    7    7           0  0.895513           0.402992
    8    8           0  0.970513           0.352992
    9    9           0  0.823938           0.313385
    10  10           1  1.038276           0.931555
    11  11           1  1.178276           0.927130
    12  12           1  1.135685           0.929565
    13  13           1  0.995685           0.934165
    14  14           1  0.849615           0.944359
    15  15           1  0.995685           0.934548
    16  16           1  1.135685           0.929950
    17  17           1  1.089615           0.932769
    18  18           1  0.949615           0.937555
    19  19           1  0.915565           0.937717
    """

    distance_map_km = {'manhattan':1, 'euclidean':2, 'minkowski':3, 'chebyshev':4, 'cosine':6}
    distance_map_acc = {'manhattan':1, 'euclidean':2, 'minkowski':3, 'chebyshev':4}
    normalization_map = {'no':0, 'l1_norm':1, 'min_max':2}
    init_map = {"first_k":1, "replace":2, "no_replace":3, "patent":4}
    mem_mode_map = {'auto':0, 'optimize-speed':1, 'optimize-space':2}

    def __init__(self, n_clusters=None, n_clusters_min=None, n_clusters_max=None,
                 init=None, max_iter=None, thread_ratio=None, distance_level=None,
                 minkowski_power=None, category_weights=None, normalization=None,
                 categorical_variable=None, tol=None, memory_mode=None, accelerated=False,
                 use_fast_library=None, use_float=None):
        super(KMeans, self).__init__()
        setattr(self, 'hanaml_parameters', pal_param_register())
        if n_clusters is None:
            if n_clusters_max is None or n_clusters_min is None:
                msg = 'You must specify either an exact value or a range for number of clusters'
                logger.error(msg)
                raise ValueError(msg)
        else:
            if n_clusters_min is not None or n_clusters_max is not None:
                msg = ('Both exact value and range ending points' +
                       ' are provided for number of groups, please choose one or the other.')
                logger.error(msg)
                raise ValueError(msg)
        self.n_clusters = self._arg('n_clusters', n_clusters, int)
        self.n_clusters_min = self._arg('n_clusters_min', n_clusters_min, int)
        self.n_clusters_max = self._arg('n_clusters_max', n_clusters_max, int)
        self.init = self._arg('init', init, self.init_map)
        self.max_iter = self._arg('max_iter', max_iter, int)
        self.thread_ratio = self._arg('thread_ratio', thread_ratio, float)
        self.minkowski_power = self._arg('minkowski_power', minkowski_power, float)
        self.category_weights = self._arg('category_weights', category_weights, float)
        self.normalization = self._arg('normalization', normalization, self.normalization_map)
        if isinstance(categorical_variable, str):
            categorical_variable = [categorical_variable]
        self.categorical_variable = self._arg('categorical_variable', categorical_variable,
                                              ListOfStrings)
        if accelerated:
            self.mem_mode = self._arg('memory_mode', memory_mode, self.mem_mode_map)
            distance_map = self.distance_map_acc
            if tol is not None:
                msg = 'Tol is only valid when accelerated is false.'
                logger.error(msg)
                raise ValueError(msg)
        else:
            self.tol = self._arg('tol', tol, float)
            distance_map = self.distance_map_km
            if memory_mode is not None:
                msg = 'Memory_mode is only valid when accelerated is true.'
                logger.error(msg)
                raise ValueError(msg)
        self.distance_level = self._arg('distance_level', distance_level, distance_map)
        if self.distance_level != 3 and minkowski_power is not None:
            msg = 'Minkowski_power will only be valid if distance_level is Minkowski.'
            logger.error(msg)
            raise ValueError(msg)
        self.accelerated = self._arg('accelerated', accelerated, bool)
        self.use_fast_library = self._arg('use_fast_library', use_fast_library, bool)
        self.use_float = self._arg('use_float', use_float, bool)

    def _prep_param(self):
        param_rows = [('GROUP_NUMBER', self.n_clusters, None, None),
                      ('GROUP_NUMBER_MIN', self.n_clusters_min, None, None),
                      ('GROUP_NUMBER_MAX', self.n_clusters_max, None, None),
                      ('DISTANCE_LEVEL', self.distance_level, None, None),
                      ('MINKOWSKI_POWER', None, self.minkowski_power, None),
                      ('CATEGORY_WEIGHTS', None, self.category_weights, None),
                      ('MAX_ITERATION', self.max_iter, None, None),
                      ('INIT_TYPE', self.init, None, None),
                      ('NORMALIZATION', self.normalization, None, None),
                      ('THREAD_RATIO', None, self.thread_ratio, None),
                      ('USE_FAST_LIBRARY', self.use_fast_library, None, None),
                      ('USE_FLOAT', self.use_float, None, None)]
        #for categorical variable
        if self.categorical_variable is not None:
            param_rows.extend([('CATEGORICAL_VARIABLE', None, None, variable)
                               for variable in self.categorical_variable])
        if self.accelerated:
            proc_name = "PAL_ACCELERATED_KMEANS"
            param_rows.append(('MEMORY_MODE', self.mem_mode, None, None))
        else:
            proc_name = "PAL_KMEANS"
            param_rows.append(('EXIT_THRESHOLD', None, self.tol, None))
        return proc_name, param_rows

    @trace_sql
    def fit(self, data, key, features=None, categorical_variable=None):
        """
        Fit the model when given training dataset.

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

        categorical_variable : str or list of str, optional

            Specifies INTEGER column(s) that should be treated as categorical.

            Other INTEGER columns will be treated as continuous.

            Defaults to None.
        """
        #PAL input format ID, Features
        conn = data.connection_context
        require_pal_usable(conn)
        key = self._arg('key', key, str, True)
        features = self._arg('features', features, ListOfStrings)
        if isinstance(categorical_variable, str):
            categorical_variable = [categorical_variable]
        categorical_variable = self._arg('categorical_variable', categorical_variable, ListOfStrings)
        cols = data.columns
        cols.remove(key)
        if features is None:
            features = cols

        unique_id = str(uuid.uuid1()).replace('-', '_').upper()
        outputs = ['#PAL_KMEANS_{}_TBL_{}_{}'.format(name, self.id, unique_id)
                   for name in ['RESULT', 'CENTERS', 'MODEL', 'STATISTICS', 'PLACEHOLDER']]
        result_tbl, centers_tbl, model_tbl, statistics_tbl, placeholder_tbl = outputs

        proc_name, param_rows = self._prep_param()
        if categorical_variable is not None:
            param_rows.extend([('CATEGORICAL_VARIABLE', None, None, variable)
                               for variable in categorical_variable])
        try:
            call_pal_auto(conn,
                          proc_name,
                          data[[key] + features],
                          ParameterTable().with_data(param_rows),
                          *outputs)

        except dbapi.Error as db_err:
            logger.exception(str(db_err))
            try_drop(conn, outputs)
            raise

        self.cluster_centers_ = conn.table(centers_tbl)
        self.labels_ = conn.table(result_tbl)
        self.model_ = conn.table(model_tbl)
        self.statistics_ = conn.table(statistics_tbl)

    def fit_predict(self, data, key, features=None, categorical_variable=None):
        """
        Fit with the dataset and return the labels.

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

        categorical_variable : str or list of str, optional

            Specifies INTEGER column(s) that should be treated as categorical.

            Other INTEGER columns will be treated as continuous.

            Defaults to None.


        Returns
        -------

        DataFrame

            Fit result, structured as follows:

              - ID column, with the same name and type as ``data`` 's ID column.
              - CLUSTER_ID, type INTEGER, cluster ID assigned to the data
                point.
              - DISTANCE, type DOUBLE, the distance between the given
                point and the cluster center.
              - SLIGHT_SILHOUETTE, type DOUBLE, estimated
                value (slight silhouette).
        """
        self.fit(data, key, features, categorical_variable)
        return self.labels_

    @trace_sql
    def predict(self, data, key, features=None):
        """
        Assign clusters to data based on a fitted model.

        The output structure of this method does not match that of
        fit_predict().

        Parameters
        ----------

        data : DataFrame

            Data points to match against computed clusters.

            This dataframe's column structure should match that
            of the data used for fit().

        key : str

            Name of the ID column.

        features : list of str, optional.

            Names of the feature columns.

            If ``features`` is not provided, it defaults to all
            the non-ID columns.

        Returns
        -------

        DataFrame

            Cluster assignment results, with 3 columns:

              - Data point ID, with name and type taken from the input
                ID column.
              - CLUSTER_ID, INTEGER type, representing the cluster the
                data point is assigned to.
              - DISTANCE, DOUBLE type, representing the distance between
                the data point and the cluster center.
        """
        return super(KMeans, self)._predict(data, key, features)

class _KClusteringBase(PALBase):
    """Base class for K-Medians and K-Medoids clustering algorithms."""

    clustering_type_proc_map = {'KMedians' :'PAL_KMEDIANS', 'KMedoids':'PAL_KMEDOIDS'}
    #meant to be override
    distance_map = {}
    normalization_map = {'no':0, 'l1_norm':1, 'min_max':2}
    init_map = {"first_k":1, "replace":2, "no_replace":3, "patent":4}

    def __init__(self,
                 n_clusters,
                 init=None,
                 max_iter=None,
                 tol=None,
                 thread_ratio=None,
                 distance_level=None,
                 minkowski_power=None,
                 category_weights=None,
                 normalization=None,
                 categorical_variable=None
                ):
        super(_KClusteringBase, self).__init__()
        self.n_cluster = self._arg('n_clusters', n_clusters, int, True)
        self.init = self._arg('init', init, self.init_map)
        self.max_iter = self._arg('max_iter', max_iter, int)
        self.tol = self._arg('tol', tol, float)
        self.thread_ratio = self._arg('thread_ratio', thread_ratio, float)
        if not self.distance_map:
            raise NotImplementedError
        self.distance_level = self._arg('distance_level', distance_level, self.distance_map)
        if minkowski_power is not None and self.distance_level != 3:
            msg = ("Invalid minkowski_power, " +
                   "valid when distance_level is Minkowski distance")
            logger.error(msg)
            raise ValueError(msg)
        self.minkowski_power = self._arg('minkowski_power', minkowski_power, float)
        self.category_weights = self._arg('category_weights', category_weights, float)
        self.normalization = self._arg('normalization', normalization, self.normalization_map)
        if isinstance(categorical_variable, str):
            categorical_variable = [categorical_variable]
        self.categorical_variable = self._arg('categorical_variable', categorical_variable,
                                              ListOfStrings)
        self.labels_ = None
        self.cluster_centers_ = None
        self._proc_name = self.clustering_type_proc_map[self.__class__.__name__]

    def _prep_param(self):
        param_data = [
            ('GROUP_NUMBER', self.n_cluster, None, None),
            ('DISTANCE_LEVEL', self.distance_level, None, None),
            ('MINKOWSKI_POWER', None, self.minkowski_power, None),
            ('CATEGORY_WEIGHTS', None, self.category_weights, None),
            ('MAX_ITERATION', self.max_iter, None, None),
            ('INIT_TYPE', self.init, None, None),
            ('NORMALIZATION', self.normalization, None, None),
            ('THREAD_RATIO', None, self.thread_ratio, None),
            ('EXIT_THRESHOLD', None, self.tol, None)
            ]
        if self.categorical_variable is not None:
            param_data.extend(('CATEGORICAL_VARIABLE', None, None, variable)
                              for variable in self.categorical_variable)
        return param_data

    @trace_sql
    def fit(self, data, key, features=None, categorical_variable=None):
        """
        Perform clustering on input dataset.

        Parameters
        ----------

        data : DataFrame

            DataFrame contains input data.

        key : str

            Name of the ID column.

        features : list of str, optional

            Names of the feature columns.

            If ``features`` is not provided, it defaults to all
            the non-ID columns.

        categorical_variable : str or list of str, optional

            Specifies INTEGER columns that should be treated as categorical.

            Other INTEGER columns will be treated as continuous.

            Defaults to None.

        """
        conn = data.connection_context
        require_pal_usable(conn)
        key = self._arg('key', key, str, True)
        features = self._arg('features', features, ListOfStrings)
        if isinstance(categorical_variable, str):
            categorical_variable = [categorical_variable]
        categorical_variable = self._arg('categorical_variable', categorical_variable, ListOfStrings)
        cols = data.columns
        cols.remove(key)
        if features is None:
            features = cols
        data_ = data[[key] + features]
        if self.distance_level == 5:
            for col_name, col_type in colspec_from_df(data[data.columns[1:]]):
                if (col_type == "DOUBLE" or
                        col_type == "INT" and self.categorical_variable is None or
                        col_type == "INT" and col_name not in self.categorical_variable):
                    msg = "When jaccard distance is used, all columns must be categorical."
                    logger.error(msg)
                    raise ValueError(msg)

        param_data = self._prep_param()
        if categorical_variable is not None:
            param_data.extend(('CATEGORICAL_VARIABLE', None, None, variable)
                              for variable in categorical_variable)

        unique_id = str(uuid.uuid1()).replace('-', '_').upper()
        result_tbl = '#{}_RESULT_TBL_{}_{}'.format(self._proc_name, self.id, unique_id)
        centers_tbl = '#{}_CENTERS_TBL_{}_{}'.format(self._proc_name, self.id, unique_id)

        try:
            call_pal_auto(conn,
                          self._proc_name,
                          data_,
                          ParameterTable().with_data(param_data),
                          result_tbl,
                          centers_tbl)
        except dbapi.Error as db_err:
            logger.exception(str(db_err))
            try_drop(conn, result_tbl)
            try_drop(conn, centers_tbl)
            raise

        self.cluster_centers_ = conn.table(centers_tbl)
        self.labels_ = conn.table(result_tbl)

    def fit_predict(self, data, key, features=None, categorical_variable=None):
        """
        Perform clustering algorithm and return labels.

        Parameters
        ----------

        data : DataFrame

            DataFrame containing input data.

        key : str

            Name of the ID column.

        features : list of str, optional

            Names of the feature columns.

            If ``features`` is not provided, it defaults to all
            the non-ID columns.

        categorical_variable : str or list of str, optional

            Specifies INTEGER column(s) that should be treated as categorical.

            Other INTEGER columns will be treated as continuous.

            Defaults to None.

        Returns
        -------

        DataFrame

            Fit result, structured as follows:

              - ID column, with the same name and type as ``data`` 's ID column.
              - CLUSTER_ID, type INTEGER, cluster ID assigned to the data
                point.
              - DISTANCE, type DOUBLE, the distance between the given
                point and the cluster center.
        """
        self.fit(data, key, features, categorical_variable)
        return self.labels_

class KMedians(_KClusteringBase):
    r"""
    K-Medians clustering algorithm that partitions n observations into
    K clusters according to their nearest cluster center. It uses medians
    of each feature to calculate cluster centers.

    Parameters
    ----------

    n_clusters : int

        Number of groups.

    init : {'first_k', 'replace', 'no_replace', 'patent'}, optional
        Controls how the initial centers are selected:

          - 'first_k': First k observations.
          - 'replace': Random with replacement.
          - 'no_replace': Random without replacement.
          - 'patent': Patent of selecting the init center (US 6,882,998 B1).

        Defaults to 'patent'.

    max_iter : int, optional

        Max iterations.

        Defaults to 100.

    tol : float, optional

        Convergence threshold for exiting iterations.

        Defaults to 1.0e-6.

    thread_ratio : float, optional

        Specifies the ratio of total number of threads that can be used by this function.

        The value range is from 0 to 1, where 0 means only using 1 thread, and 1 means using at most all the currently available threads.

        Values outside the range will be ignored and this function heuristically determines the number of threads to use.

        Defaults to 0.

    distance_level : {'manhattan', 'euclidean', 'minkowski', 'chebyshev', 'cosine'} str, optional
        Ways to compute the distance between the item and the cluster center.

        Defaults to 'euclidean'.

    minkowski_power : float, optional

        When Minkowski distance is used, this parameter controls the value of
        power.

        Only valid when ``distance_level`` is 'minkowski'.

        Defaults to 3.0.

    category_weights : float, optional

        Represents the weight of category attributes.

        Defaults to 0.707.

    normalization : {'no', 'l1_norm', 'min_max'}, optional

        Normalization type.

          - 'no': No, normalization will not be applied.
          - 'l1_norm': Yes, for each point X (x\ :sub:`1`\, x\ :sub:`2`\, ..., x\ :sub:`n`\), the normalized
            value will be X'(x\ :sub:`1`\ /S,x\ :sub:`2`\ /S,...,x\ :sub:`n`\ /S),
            where S = \|x\ :sub:`1`\ \|+\|x\ :sub:`2`\ \|+...\|x\ :sub:`n`\ \|.
          - 'min_max': Yes, for each column C, get the min and max value of C,
            and then C[i] = (C[i]-min)/(max-min).

        Defaults to 'no'.

    categorical_variable : str or list of str, optional

        Specifies INTEGER column(s) in the data that should be treated as categorical.

        Defaults to None.

    Attributes
    ----------

    cluster_centers_ : DataFrame

        Coordinates of cluster centers.

    labels_ : DataFrame

        Cluster assignment and distance to cluster center for each point.

    Examples
    --------

    Input dataframe df1 for clustering:

    >>> df1.collect()
        ID  V000 V001  V002
    0    0   0.5    A   0.5
    1    1   1.5    A   0.5
    2    2   1.5    A   1.5
    3    3   0.5    A   1.5
    4    4   1.1    B   1.2
    5    5   0.5    B  15.5
    6    6   1.5    B  15.5
    7    7   1.5    B  16.5
    8    8   0.5    B  16.5
    9    9   1.2    C  16.1
    10  10  15.5    C  15.5
    11  11  16.5    C  15.5
    12  12  16.5    C  16.5
    13  13  15.5    C  16.5
    14  14  15.6    D  16.2
    15  15  15.5    D   0.5
    16  16  16.5    D   0.5
    17  17  16.5    D   1.5
    18  18  15.5    D   1.5
    19  19  15.7    A   1.6

    Creating KMedians instance:

    >>> kmedians = KMedians(n_clusters=4, init='first_k',
    ...                     max_iter=100, tol=1.0E-6,
    ...                     distance_level='Euclidean',
    ...                     thread_ratio=0.3, category_weights=0.5)

    Performing fit() on given dataframe:

    >>> kmedians.fit(data=df1, key='ID')
    >>> kmedians.cluster_centers_.collect()
       CLUSTER_ID  V000 V001  V002
    0           0   1.1    A   1.2
    1           1  15.7    D   1.5
    2           2  15.6    C  16.2
    3           3   1.2    B  16.1

    Performing fit_predict() on given dataframe:

    >>> kmedians.fit_predict(data=df1, key='ID').collect()
        ID  CLUSTER_ID  DISTANCE
    0    0           0  0.921954
    1    1           0  0.806226
    2    2           0  0.500000
    3    3           0  0.670820
    4    4           0  0.707107
    5    5           3  0.921954
    6    6           3  0.670820
    7    7           3  0.500000
    8    8           3  0.806226
    9    9           3  0.707107
    10  10           2  0.707107
    11  11           2  1.140175
    12  12           2  0.948683
    13  13           2  0.316228
    14  14           2  0.707107
    15  15           1  1.019804
    16  16           1  1.280625
    17  17           1  0.800000
    18  18           1  0.200000
    19  19           1  0.807107
    """
    distance_map = {'manhattan':1, 'euclidean':2,
                    'minkowski':3, 'chebyshev':4,
                    'cosine':6}

class KMedoids(_KClusteringBase):
    r"""
    K-Medoids clustering algorithm that partitions n observations into
    K clusters according to their nearest cluster center. It uses medoids
    to calculate cluster centers. K-Medoids is more robust
    to noise and outliers.

    Parameters
    ----------

    n_clusters : int

        Number of groups.

    init : {'first_k', 'replace', 'no_replace', 'patent'}, optional
        Controls how the initial centers are selected:

          - 'first_k': First k observations.
          - 'replace': Random with replacement.
          - 'no_replace': Random without replacement.
          - 'patent': Patent of selecting the init center (US 6,882,998 B1).

        Defaults to 'patent'.

    max_iter : int, optional

        Max iterations.

        Defaults to 100.

    tol : float, optional

        Convergence threshold for exiting iterations.

        Defaults to 1.0e-6.

    thread_ratio : float, optional

        Controls the proportion of available threads to use.

        The value range is from 0 to 1, where 0 indicates
        a single thread, and 1 indicates all available threads.

        Values between 0 and 1 will use up to that percentage of available
        threads.

        Values outside this range tell PAL to heuristically determine
        the number of threads to use.

        Defaults to 0.

    distance_level : {'manhattan', 'euclidean', 'minkowski', 'chebyshev', 'cosine'} str, optional
        Ways to compute the distance between the item and the cluster center.

        Defaults to 'euclidean'.

    minkowski_power : float, optional

        When Minkowski distance is used, this parameter controls the
        value of power.

        Only valid when ``distance_level`` is 'minkowski'.

        Defaults to 3.0.

    category_weights : float, optional

        Represents the weight of category attributes.

        Defaults to 0.707.

    normalization : {'no', 'l1_norm', 'min_max'}, optional

        Normalization type.

          - 'no': No, normalization will not be applied.
          - 'l1_norm': Yes, for each point X (x\ :sub:`1`\, x\ :sub:`2`\, ..., x\ :sub:`n`\), the normalized
            value will be X'(x\ :sub:`1`\ /S,x\ :sub:`2`\ /S,...,x\ :sub:`n`\ /S),
            where S = \|x\ :sub:`1`\ \|+\|x\ :sub:`2`\ \|+...\|x\ :sub:`n`\ \|.
          - 'min_max': Yes, for each column C, get the min and max value of C,
            and then C[i] = (C[i]-min)/(max-min).

        Defaults to 'no'.

    categorical_variable : str or list of str, optional

        Specifies INTEGER column(s) in the data that should be treated as categorical.

        Defaults to None.

    Attributes
    ----------

    cluster_centers_ : DataFrame

        Coordinates of cluster centers.

    labels_ : DataFrame

        Cluster assignment and distance to cluster center for each point.

    Examples
    --------

    Input dataframe df1 for clustering:

    >>> df1.collect()
        ID  V000 V001  V002
    0    0   0.5    A   0.5
    1    1   1.5    A   0.5
    2    2   1.5    A   1.5
    3    3   0.5    A   1.5
    4    4   1.1    B   1.2
    5    5   0.5    B  15.5
    6    6   1.5    B  15.5
    7    7   1.5    B  16.5
    8    8   0.5    B  16.5
    9    9   1.2    C  16.1
    10  10  15.5    C  15.5
    11  11  16.5    C  15.5
    12  12  16.5    C  16.5
    13  13  15.5    C  16.5
    14  14  15.6    D  16.2
    15  15  15.5    D   0.5
    16  16  16.5    D   0.5
    17  17  16.5    D   1.5
    18  18  15.5    D   1.5
    19  19  15.7    A   1.6

    Creating KMedoids instance:

    >>> kmedoids = KMedoids(n_clusters=4, init='first_K',
    ...                     max_iter=100, tol=1.0E-6,
    ...                     distance_level='Euclidean',
    ...                     thread_ratio=0.3, category_weights=0.5)

    Performing fit() on given dataframe:

    >>> kmedoids.fit(data=df1, key='ID')
    >>> kmedoids.cluster_centers_.collect()
       CLUSTER_ID  V000 V001  V002
    0           0   1.5    A   1.5
    1           1  15.5    D   1.5
    2           2  15.5    C  16.5
    3           3   1.5    B  16.5

    Performing fit_predict() on given dataframe:

    >>> kmedoids.fit_predict(data=df1, key='ID').collect()
        ID  CLUSTER_ID  DISTANCE
    0    0           0  1.414214
    1    1           0  1.000000
    2    2           0  0.000000
    3    3           0  1.000000
    4    4           0  1.207107
    5    5           3  1.414214
    6    6           3  1.000000
    7    7           3  0.000000
    8    8           3  1.000000
    9    9           3  1.207107
    10  10           2  1.000000
    11  11           2  1.414214
    12  12           2  1.000000
    13  13           2  0.000000
    14  14           2  1.023335
    15  15           1  1.000000
    16  16           1  1.414214
    17  17           1  1.000000
    18  18           1  0.000000
    19  19           1  0.930714
    """
    distance_map = {'manhattan':1, 'euclidean':2,
                    'minkowski':3, 'chebyshev':4,
                    'jaccard':5, 'cosine':6}
