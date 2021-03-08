"""
This module contains Python wrapper for SAP HANA PAL unified-clustering.

The following classes are available:
    * :class:`UnifiedClustering`
"""

#pylint: disable=too-many-lines, too-many-branches
#pylint: disable=line-too-long, too-many-statements
#pylint: disable=too-many-locals
#pylint: disable=too-many-arguments
#pylint: disable=ungrouped-imports
import logging
import uuid
import warnings
#import pandas as pd
from hdbcli import dbapi

from hana_ml.ml_base import try_drop
from hana_ml.ml_exceptions import FitIncompleteError
from .sqlgen import trace_sql
from .pal_base import (
    PALBase,
    ParameterTable,
    require_pal_usable,
    pal_param_register,
    call_pal_auto,
    ListOfStrings)

logger = logging.getLogger(__name__) #pylint: disable=invalid-name

class UnifiedClustering(PALBase):#pylint: disable=too-many-instance-attributes
    """
    The Python wrapper for SAP HANA PAL unified-clustering function.

    The clustering algorithms include:

    - 'AgglomerateHierarchicalClustering'
    - 'DBSCAN'
    - 'GaussianMixture'
    - 'AcceleratedKMeans'
    - 'KMeans'
    - 'KMedians'
    - 'KMedoids'
    - 'SOM'

    For GMM, you must configure the init_mode and n_components or init_centers parameters to define INITIALIZE_PARAMETER in SAP HANA PAL.

    Compared to the original K-Medians and K-Medoids, unified clustering creates models after a training and then performs cluster assignment through the model.


    Parameters
    ----------

    func : str

        The name of a specified regression algorithm.

        The following algorithms are supported:

        - 'AgglomerateHierarchicalClustering'
        - 'DBSCAN'
        - 'GaussianMixture'
        - 'AcceleratedKMeans'
        - 'KMeans'
        - 'KMedians'
        - 'KMedoids'
        - 'SOM'

    **kwargs : keyword arguments

        Arbitrary keyword arguments and please referred to the responding algorithm for the parameters' key-value pair.

        **Note that some parameters are disabled in the clustering algorithm!**

        - **'AgglomerateHierarchicalClustering'** : :class:`~hana_ml.algorithms.pal.clustering.AgglomerateHierarchicalClustering`
        Note that distance_level is supported which has the same options as affinity. If both parameters are entered, distance_level takes precedence over affinity.

        - **'DBSCAN'** : :class:`~hana_ml.algorithms.pal.clustering.DBSCAN`
        Note that distance_level is supported which has the same options as metric. If both parameters are entered, distance_level takes precedence over metric.

        - **'GMM'** : :class:`~hana_ml.algorithms.pal.mixture.GaussianMixture`

        - **'AcceleratedKMeans'** : :class:`~hana_ml.algorithms.pal.clustering.KMeans`  Note that parameter 'accelerated' is not valid in this function.

        - **'KMeans'** : :class:`~hana_ml.algorithms.pal.clustering.KMeans`

        - **'KMedians'** : :class:`~hana_ml.algorithms.pal.clustering.KMedians`

        - **'KMedoids'** : :class:`~hana_ml.algorithms.pal.clustering.KMedoids`

        - **'SOM'** : :class:`~hana_ml.algorithms.pal.som.SOM`

    Attributes
    ----------

    labels_ : DataFrame

        Label assigned to each sample. Also includes Distance between a given point and
        the cluster center (k-means), nearest core object (DBSCAN), weight vector (SOM)
        Or probability of a given point belonging to the corresponding cluster (GMM).

    centers_ : DataFrame

        Coordinates of cluster centers.

    model_ : DataFrame

        Model content.

    statistics_ : DataFrame

        Names and values of statistics.

    optim_param_ : DataFrame

        Provides optimal parameters selected.

        Available only when parameter selection is triggered.


    Examples
    --------
    Training data for clustering:

    >>> data_tbl.collect()
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

    Create a UnifiedClustering instance for linear regression problem:

    >>> kmeans_params = dict(n_clusters=4, init='first_k', max_iter=100,
                             tol=1.0E-6, thread_ratio=1.0, distance_level='Euclidean',
                             category_weights=0.5)

    >>> ukmeans = UnifiedClustering(func = 'Kmeans', **kmeans_params)

    Fit the UnifiedClustering instance with the aforementioned training data:

    >>> ukmeans.fit(self.data_kmeans, key='ID')

    Check the resulting statistics on testing data:

    >>> ukmeans.label_.collect()
        ID  CLUSTER_ID  DISTANCE  SLIGHT_SILHOUETE
    0    0           0  0.891088          0.944370
    1    1           0  0.863917          0.942478
    2    2           0  0.806252          0.946288
    3    3           0  0.835684          0.944942
    ......
    16  16           1  0.976885          0.939386
    17  17           1  0.818178          0.945878
    18  18           1  0.722799          0.952170
    19  19           1  1.102342          0.925679

    Data for prediction(cluster assignment):

    >>> data_pred.collect()
       ID  CLUSTER_ID  DISTANCE
    0  88           3  0.981659
    1  89           3  0.826454
    2  90           2  1.990205
    3  91           2  0.325812

    Perform prediction:

    >>> result = ukmeans.predict(self.data_kmeans_predict, key='ID')
    >>> result.collect()
       ID  CLUSTER_ID  DISTANCE
    0  88           3  0.981659
    1  89           3  0.826454
    2  90           2  1.990205
    3  91           2  0.325812

    """

    func_dict = {
        'agglomeratehierarchicalclustering' : 'AHC',
        'dbscan' : 'DBSCAN',
        'gaussianmixture' : 'GMM',
        'acceleratedkmeans' : 'AKMEANS',
        'kmeans' : 'KMEANS',
        'kmedians' : 'KMEDIANS',
        'kmedoids' : 'KMEDOIDS',
        'som': 'SOM'}

    map_dict = {
        'AHC' : {
            'n_clusters' : ('N_CLUSTERS', int),
            'affinity' : ('AFFINITY', int, {'manhattan':1, 'euclidean':2, 'minkowski':3, 'chebyshev':4,
                                            'cosine':6, 'pearson correlation':7, 'squared euclidean':8,
                                            'jaccard':9, 'gower':10}),
            'distance_level' : ('DISTANCE_LEVEL', int, {'manhattan':1, 'euclidean':2, 'minkowski':3, 'chebyshev':4,
                                                        'cosine':6, 'pearson correlation':7, 'squared euclidean':8,
                                                        'jaccard':9, 'gower':10}),
            'linkage' : ('LINKAGE', int, {'nearest neighbor':1, 'furthest neighbor':2, 'group average':3, 'weighted average':4,
                                          'centroid clustering':5, 'median clustering':6, 'ward':7}),
            'thread_ratio' : ('THREAD_RATIO', float),
            'distance_dimension' : ('DISTANCE_DIMENSION', float),
            'normalization' : ('NORMALIZATION', int, {'no': 0, 'z-score': 1, 'zero-centred-min-max': 2, 'min-max': 3}),
            'category_weights' : ('CATEGORY_WEIGHTS', float)},
        'DBSCAN' : {
            'minpts' : ('MINPTS', int),
            'eps' : ('EPS', float),
            'thread_ratio' : ('THREAD_RATIO', float),
            'metric' : ('METRIC', int, {'manhattan':1, 'euclidean':2, 'minkowski':3, 'chebyshev':4,
                                        'standardized_euclidean':5, 'cosine':6}),
            'distance_level' : ('DISTANCE_LEVEL', int, {'manhattan':1, 'euclidean':2, 'minkowski':3, 'chebyshev':4,
                                                        'standardized_euclidean':5, 'cosine':6}),
            'minkowski_power' : ('MINKOWSKI_POWER', int),
            'category_weights' : ('CATEGORY_WEIGHTS', float),
            'algorithm' : ('ALGORITHM', int, {'brute-force':0, 'kd-tree':1}),
            'save_model' : ('SAVE_MODEL', bool)},
        'GMM' : {
            'init_param' : ('INIT_MODE', int, {'farthest_first_traversal': 0, 'manual': 1,
                                               'random_means': 2, 'k_means++': 3}),
            'n_components' : ('INITIALIZE_PARAMETER', int),
            'init_centers' : ('INITIALIZE_PARAMETER', ListOfStrings),
            'covariance_type' : ('COVARIANCE_TYPE', int, {'full': 0, 'diag': 1, 'tied_diag': 2}),
            'shared_covariance' : ('SHARED_COVARIANCE', bool),
            'category_weight' : ('CATEGORY_WEIGHT', float),
            'max_iter' : ('MAX_ITER', int),
            'thread_ratio' : ('THREAD_RATIO', float),
            'error_tol' : ('ERROR_TOL', float),
            'regularization' : ('REGULARIZATION', float),
            'random_seed' : ('SEED', int)},
        'AKMEANS' : {
            'n_clusters' : ('N_CLUSTERS', int),
            'n_clusters_min' : ('N_CLUSTERS_MIN', int),
            'n_clusters_max' : ('N_CLUSTERS_MAX', int),
            'distance_level' : ('DISTANCE_LEVEL', int, {'manhattan':1, 'euclidean':2,
                                                        'minkowski':3, 'chebyshev':4}),
            'minkowski_power' : ('MINKOWSKI_POWER', float),
            'category_weights' : ('CATEGORY_WEIGHTS', float),
            'max_iter' : ('MAX_ITER', int),
            'init' : ('INIT', int, {'first_k':1, 'no_replace':2, 'replace':3, 'patent':4}),
            'normalization' : ('NORMALIZATION', int, {'no':0, 'l1_norm':1, 'min_max':2}),
            'thread_ratio' : ('THREAD_RATIO', float),
            'memory_mode' : ('MEMORY_MODE', int, {'auto':0, 'optimize-speed':1, 'optimize-space':2}),
            'tol' : ('TOL', float)},
        'KMEANS' : {
            'n_clusters' : ('N_CLUSTERS', int),
            'n_clusters_min' : ('N_CLUSTERS_MIN', int),
            'n_clusters_max' : ('N_CLUSTERS_MAX', int),
            'distance_level' : ('DISTANCE_LEVEL', int, {'manhattan':1, 'euclidean':2,
                                                        'minkowski':3, 'chebyshev':4,
                                                        'cosine':6}),
            'minkowski_power' : ('MINKOWSKI_POWER', float),
            'category_weights' : ('CATEGORY_WEIGHTS', float),
            'max_iter' : ('MAX_ITER', int),
            'init' : ('INIT', int, {'first_k':1, 'replace':2, 'no_replace':3, 'patent':4}),
            'normalization' : ('NORMALIZATION', int, {'no':0, 'l1_norm':1, 'min_max':2}),
            'thread_ratio' : ('THREAD_RATIO', float),
            'tol' : ('TOL', float)},
        'KMEDIANS' : {
            'n_clusters' : ('N_CLUSTERS', int),
            'init' : ('INIT', int, {'first_k':1, 'replace':2, 'no_replace':3, 'patent':4}),
            'max_iter' : ('MAX_ITER', int),
            'tol' : ('TOL', float),
            'thread_ratio' : ('THREAD_RATIO', float),
            'distance_level' : ('DISTANCE_LEVEL', int, {'manhattan':1, 'euclidean':2,
                                                        'minkowski':3, 'chebyshev':4,
                                                        'cosine':6}),
            'minkowski_power' : ('MINKOWSKI_POWER', float),
            'category_weights' : ('CATEGORY_WEIGHTS', float),
            'normalization' : ('NORMALIZATION', int, {'no':0, 'l1_norm':1, 'min_max':2})},
        'KMEDOIDS' : {
            'n_clusters' : ('N_CLUSTERS', int),
            'init' : ('INIT', int, {'first_k':1, 'replace':2, 'no_replace':3, 'patent':4}),
            'max_iter' : ('MAX_ITER', int),
            'tol' : ('TOL', float),
            'thread_ratio' : ('THREAD_RATIO', float),
            'distance_level' : ('DISTANCE_LEVEL', int, {'manhattan':1, 'euclidean':2,
                                                        'minkowski':3, 'chebyshev':4,
                                                        'jaccard':5, 'cosine':6}),
            'minkowski_power' : ('MINKOWSKI_POWER', float),
            'category_weights' : ('CATEGORY_WEIGHTS', float),
            'normalization' : ('NORMALIZATION', int, {'no':0, 'l1_norm':1, 'min_max':2})},
        'SOM' : {
            'covergence_criterion' : ('COVERGENCE_CRITERION', float),
            'normalization' : ('NORMALIZATION', int, {'no': 0, 'min-max': 1, 'z-score': 2}),
            'n_clusters' : ('N_CLUSTERS', int),
            'random_seed' : ('RANDOM_SEED', int),
            'height_of_map' : ('HEIGHT_OF_MAP', int),
            'width_of_map' : ('WIDTH_OF_MAP', int),
            'kernel_function' : ('KERNEL_FUNCTION', int, {'gaussian':1, 'flat':2}),
            'alpha' : ('ALPHA', float),
            'learning_rate' : ('LEARNING_RATE', int, {'exponential':1, 'linear':2}),
            'shape_of_grid' : ('SHAPE_OF_GRID', int, {'rectangle':1, 'hexagon':2}),
            'radius' : ('RADIUS', float),
            'batch_som' : ('BATCH_SOM', int, {'classical':0, 'batch':1}),
            'max_iter' : ('MAX_ITER', int)}
    }

    def __init__(self,
                 func,
                 **kwargs):
        super(UnifiedClustering, self).__init__()
        setattr(self, 'hanaml_parameters', pal_param_register())
        func = func.lower()
        self.func = self._arg('Function name', func, self.func_dict)

        self.params = dict(**kwargs)
        self.__pal_params = {}
        func_map = self.map_dict[self.func]

        if self.func == 'GMM':
            if self.params.get('init_param') is None:
                err_msg = "'init_param' is a madantory paramter for gaussianmixture!"
                logger.error(err_msg)
                raise ValueError(err_msg)

            method = self.params['init_param']
            if method == 'manual' and self.params.get('init_centers') is None:
                err_msg = "'init_centers' is a madantory paramter when init_param is 'manual' of gaussianmixture!"
                logger.error(err_msg)
                raise ValueError(err_msg)
            if method != 'manual' and self.params.get('n_components') is None:
                err_msg = "'n_components' is a madantory paramter when init_param is not 'manual' of gaussianmixture!"
                logger.error(err_msg)
                raise ValueError(err_msg)

        if self.func == 'DBSCAN':
            if ((self.params.get('metric') is not None) and (self.params.get('distance_level') is not None) and (self.params['metric'] != self.params['distance_level'])):
                self.params['metric'] = None
                warn_msg = "when metric and distance_level are both entered in DBSCAN, distance_level takes precedence over metric!"
                warnings.warn(message=warn_msg)

        if self.func == 'AHC':
            if ((self.params.get('affinity') is not None) and (self.params.get('distance_level') is not None) and (self.params['affinity'] != self.params['distance_level'])):
                self.params['affinity'] = None
                warn_msg = "when affinity and distance_level are both entered in AgglomerateHierarchicalClustering, distance_level takes precedence over affinity!"
                warnings.warn(message=warn_msg)

        for parm in self.params:
            if parm in func_map.keys():
                if parm == 'n_components':
                    self.params['n_components'] = self._arg('n_components', self.params['n_components'], int)
                elif parm == 'init_centers':
                    self.params['init_centers'] = self._arg('init_centers', self.params['init_centers'], list)
                else:
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

        self.labels_ = None
        self.centers_ = None
        self.model_ = None
        self.statistics_ = None
        self.optimal_param_ = None

    def __map_param(self, name, value, typ):#pylint:disable=no-self-use
        tpl = ()
        if typ in [int, bool]:
            tpl = (name, value, None, None)
        elif typ == float:
            tpl = (name, None, value, None)
        elif typ in [str, ListOfStrings]:
            tpl = (name, None, None, value)
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
            key,
            features=None,
            categorical_variable=None,
            string_variable=None,
            variable_weight=None):
        """
        Fit function for unified clustering.

        Parameters
        ----------

        data : DataFrame
            Training data.

        key : str
            Name of the ID column.

        features : list of str, optional
            Names of the feature columns.

            If ``features`` is not provided, it defaults to all non-ID,
            non-label columns.

        categorical_variable : str or list of str, optional
            Specifies INTEGER column(s) that should be treated as categorical.

            Other INTEGER columns will be treated as continuous.

            No default value.

        string_variable : str or list of str, optional
            Indicates a string column storing not categorical data.

            Levenshtein distance is used to calculate similarity between two strings.

            Ignored if it is not a string column.

            Only valid for DBSCAN clustering.

            Defaults to None.
        variable_weight : dict, optional
            Specifies the weight of a variable participating in distance calculation.

            The value must be greater or equal to 0.

            Defaults to 1 for variables not specified.

            Only valid for DBSCAN clustering.

            Defaults to None.
        """
        conn = data.connection_context
        require_pal_usable(conn)
        key = self._arg('key', key, str)
        if isinstance(features, str):
            features = [features]
        features = self._arg('features', features, ListOfStrings)
        cols = data.columns
        id_col = [key]
        cols.remove(key)

        if features is None:
            features = cols
        data_ = data[id_col + features]
        unique_id = str(uuid.uuid1()).replace('-', '_').upper()
        param_rows = [('FUNCTION', None, None, self.func)]
        for name in self.__pal_params:
            value, typ = self.__pal_params[name]
            tpl = [self.__map_param(name, value, typ)]
            param_rows.extend(tpl)

        if isinstance(categorical_variable, str):
            categorical_variable = [categorical_variable]
        categorical_variable = self._arg('categorical_variable', categorical_variable, ListOfStrings)
        if categorical_variable is not None:
            param_rows.extend([('CATEGORICAL_VARIABLE', None, None, var) for var in categorical_variable])

        if self.func != 'DBSCAN' and ((string_variable is not None) or (variable_weight is not None)):
            err_msg = "'string_variable' and 'variable_weight' is only valid when func is DBSCAN!"
            logger.error(err_msg)
            raise ValueError(err_msg)

        # DBSCAN auto_param handling
        if self.func == 'DBSCAN':
            auto_param = [('AUTO_PARAM', None, None, 'true')]
            if (self.params.get('minpts')) and (self.params.get('eps')):
                if (self.params['minpts']) and (self.params['eps']):
                    auto_param = [('AUTO_PARAM', None, None, 'false')]
            param_rows.extend(auto_param)
            # string_variable handling
            if string_variable is not None:
                if isinstance(string_variable, str):
                    string_variable = [string_variable]
                try:
                    string_variable = self._arg('string_variable', string_variable, ListOfStrings)
                except:
                    msg = ("`string_variable` must be list of strings or string.")
                    logger.error(msg)
                    raise TypeError(msg)
                param_rows.extend(('STRING_VARIABLE', None, None, variable)
                                  for variable in string_variable)
            # variable_weight handling
            if variable_weight is not None:
                variable_weight = self._arg('variable_weight', variable_weight, dict)
                for k, value in variable_weight.items():
                    if not isinstance(k, str):
                        msg = ("The key of variable_weight must be a string!")
                        logger.error(msg)
                        raise TypeError(msg)
                if not isinstance(value, (float, int)):
                    msg = ("The value of variable_weight must be a float!")
                    logger.error(msg)
                    raise TypeError(msg)
                param_rows.extend(('VARIABLE_WEIGHT', None, value, k) for k, value in variable_weight.items())

        # GMM INITIALIZE_PARAMETER handling
        if self.func == 'GMM':
            if self.params['init_param'] == 'manual':
                init_centers = self.params['init_centers']
                param_rows.extend([('INITIALIZE_PARAMETER', None, None, str(var)) for var in init_centers])
            else:
                param_rows.extend([('INITIALIZE_PARAMETER', None, None, str(self.params['n_components']))])

        outputs = ['RESULT', 'CENTERS', 'MODEL', 'STATS', 'OPT_PARAM', 'PLACE_HOLDER1', 'PLACE_HOLDER2']
        outputs = ['#PAL_UNIFIED_CLUSTERING_{}_{}_{}'.format(tbl, self.id, unique_id)
                   for tbl in outputs]
        labels_tbl, centers_tbl, model_tbl, stats_tbl, opt_param_tbl, _, _ = outputs
        try:
            call_pal_auto(conn,
                          'PAL_UNIFIED_CLUSTERING',
                          data_,
                          ParameterTable().with_data(param_rows),
                          *outputs)
        except dbapi.Error as db_err:
            logger.error(str(db_err))
            try_drop(conn, outputs)
            raise
        #pylint: disable=attribute-defined-outside-init
        self.labels_ = conn.table(labels_tbl)
        self.centers_ = conn.table(centers_tbl)
        self.model_ = conn.table(model_tbl)
        self.statistics_ = conn.table(stats_tbl)
        self.optimal_param_ = conn.table(opt_param_tbl)

    @trace_sql
    def predict(self, data, key, features=None, model=None):
        r"""
        Predict with the clustering model.

        Cluster assignment is a unified interface to call a cluster assignment algorithm
        to assign data to clusters that are previously generated by some clustering methods,
        including K-Means, Accelerated K-Means, K-Medians, K- Medoids, DBSCAN, SOM, and GMM.

        AgglomerateHierarchicalClustering does not provide predict function!

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
            Fitted clustering model.

            Defaults to self.model\_.

        Returns
        -------
            DataFrame
                Cluster assignment result, structured as follows:

                -  1st column : Data ID
                -  2nd column : Assigned cluster ID
                -  3rd column : Distance metric between a given point and the assigned cluster. For different functions, this could be:

                    - Distance between a given point and the cluster center(k-means, k-medians, k-medoids)
                    - Distance between a given point and the nearest core object(DBSCAN)
                    - Distance between a given point and the weight vector(SOM)
                    - Probability of a given point belonging to the corresponding cluster(GMM)

        """

        if self.func == 'AHC':
            err_msg = "AgglomerateHierarchicalClustering does not provide predict function!"
            logger.error(err_msg)
            raise ValueError(err_msg)
        if model is None and getattr(self, 'model_') is None:
            msg = "Model not initialized. Perform a fit first."
            logger.error(msg)
            raise FitIncompleteError(msg)
        conn = data.connection_context
        require_pal_usable(conn)
        cols = data.columns
        key = self._arg('key', key, str, required=True)
        id_col = [key]
        cols.remove(key)
        if isinstance(features, str):
            features = [features]
        features = self._arg('features', features, ListOfStrings)
        if features is None:
            features = cols
        data_ = data[id_col + features]

        if model is None:
            model = self.model_

        unique_id = str(uuid.uuid1()).replace('-', '_').upper()
        outputs = ['ASSIGNMENT', 'PLACE_HOLDER1']
        outputs = ['#PAL_UNIFIED_CLUSTERING_PREDICT_{}_{}_{}'.format(tbl, self.id, unique_id)
                   for tbl in outputs]
        assignment_tbl, _ = outputs

        try:
            call_pal_auto(conn,
                          'PAL_UNIFIED_CLUSTERING_ASSIGNMENT',
                          data_,
                          model,
                          ParameterTable(),
                          *outputs)
        except dbapi.Error as db_err:
            logger.exception(str(db_err))
            try_drop(conn, outputs)
            raise
        return conn.table(assignment_tbl)
