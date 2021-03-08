"""
This module contains Python wrappers for Gaussian mixture model algorithm.

The following class is available:

    * :class:`GaussianMixture`
"""

#pylint: disable=too-many-locals,unused-variable, line-too-long, relative-beyond-top-level
import logging
import uuid
from hdbcli import dbapi
from hana_ml.algorithms.pal.sqlgen import trace_sql
from hana_ml.algorithms.pal.clustering import _ClusterAssignmentMixin
from hana_ml.ml_base import try_drop
from .pal_base import (
    PALBase,
    ParameterTable,
    ListOfStrings,
    pal_param_register,
    require_pal_usable,
    call_pal_auto
)

logger = logging.getLogger(__name__) #pylint: disable=invalid-name

class GaussianMixture(PALBase, _ClusterAssignmentMixin):#pylint: disable=too-many-instance-attributes, too-few-public-methods
    """
    Representation of a Gaussian mixture model probability distribution.

    Parameters
    ----------

    init_param : {'farthest_first_traversal','manual','random_means','kmeans++'}
        Specifies the initialization mode.

          - farthest_first_traversal: The initial centers are given by
            the farthest-first traversal algorithm.
          - manual: The initial centers are the init_centers given by
            user.
          - random_means: The initial centers are the means of all the data
            that are randomly weighted.
          - kmeans++: The initial centers are given using the k-means++ approach.

    n_components : int
        Specifies the number of Gaussian distributions.

        Mandatory when ``init_param`` is not 'manual'.

    init_centers : list of integers/strings
        Specifies the rows of ``data`` to be used as initial centers by provides their IDs in ``data``.

        Mandatory when ``init_param`` is 'manual'.

    covariance_type : {'full', 'diag', 'tied_diag'}, optional
        Specifies the type of covariance matrices in the model.

          - full: use full covariance matrices.
          - diag: use diagonal covariance matrices.
          - tied_diag: use diagonal covariance matrices with all equal
            diagonal entries.

        Defaults to 'full'.

    shared_covariance : bool, optional
        All clusters share the same covariance matrix if True.

        Defaults to False.

    thread_ratio : float, optional
        Controls the proportion of available threads that can be used.

        The value range is from 0 to 1, where 0 indicates a single thread,
        and 1 indicates up to all available threads.

        Values between 0 and 1 will use that percentage of available threads.

        Values outside this range tell PAL to heuristically determine the number of threads to use.

        Defaults to 0.

    max_iter : int, optional
        Specifies the maximum number of iterations for the EM algorithm.

        Defaults value: 100.

    categorical_variable : str or list of str, optional
        Specifies INTEGER column(s) that should be be treated as categorical.

        Other INTEGER columns will be treated as continuous.

    category_weight : float, optional
        Represents the weight of category attributes.

        Defaults to 0.707.

    error_tol : float, optional
        Specifies the error tolerance, which is the stop condition.

        Defaults to 1e-5.

    regularization : float, optional
        Regularization to be added to the diagonal of covariance matrices
        to ensure positive-definite.

        Defaults to 1e-6.

    random_seed : int, optional
        Indicates the seed used to initialize the random number generator:

          - 0: Uses the system time.
          - Not 0: Uses the provided value.

        Defaults to 0.

    Attributes
    ----------
    model_ : DataFrame
        Trained model content.
    labels_ : DataFrame
        Cluster membership probabilties for each data point.
    stats_ : DataFrame
        Statistics.

    Examples
    --------
    Input dataframe df1 for training:

    >>> df1.collect()
        ID     X1     X2  X3
    0    0   0.10   0.10   1
    1    1   0.11   0.10   1
    2    2   0.10   0.11   1
    3    3   0.11   0.11   1
    4    4   0.12   0.11   1
    5    5   0.11   0.12   1
    6    6   0.12   0.12   1
    7    7   0.12   0.13   1
    8    8   0.13   0.12   2
    9    9   0.13   0.13   2
    10  10   0.13   0.14   2
    11  11   0.14   0.13   2
    12  12  10.10  10.10   1
    13  13  10.11  10.10   1
    14  14  10.10  10.11   1
    15  15  10.11  10.11   1
    16  16  10.11  10.12   2
    17  17  10.12  10.11   2
    18  18  10.12  10.12   2
    19  19  10.12  10.13   2
    20  20  10.13  10.12   2
    21  21  10.13  10.13   2
    22  22  10.13  10.14   2
    23  23  10.14  10.13   2

    Creating the GMM instance:

    >>> gmm = GaussianMixture(init_param='farthest_first_traversal',
    ...                       n_components=2, covariance_type='full',
    ...                       shared_covariance=False, max_iter=500,
    ...                       error_tol=0.001, thread_ratio=0.5,
    ...                       categorical_variable=['X3'], random_seed=1)

    Performing fit() on the given dataframe:

    >>> gmm.fit(data=df1, key='ID')

    Expected output:

    >>> gmm.labels_.head(14).collect()
        ID  CLUSTER_ID     PROBABILITY
    0    0           0          0.0
    1    1           0          0.0
    2    2           0          0.0
    3    4           0          0.0
    4    5           0          0.0
    5    6           0          0.0
    6    7           0          0.0
    7    8           0          0.0
    8    9           0          0.0
    9    10          0          1.0
    10   11          0          1.0
    11   12          0          1.0
    12   13          0          1.0
    13   14          0          0.0

    >>> gmm.stats_.collect()
           STAT_NAME       STAT_VALUE
    1     log-likelihood     11.7199
    2         aic          -504.5536
    3         bic          -480.3900

    >>> gmm.model_collect()
           ROW_INDEX    CLUSTER_ID         MODEL_CONTENT
    1        0            -1           {"Algorithm":"GMM","Metadata":{"DataP...
    2        1             0           {"GuassModel":{"covariance":[22.18895...
    3        2             1           {"GuassModel":{"covariance":[22.19450...
    """
    init_param_map = {'farthest_first_traversal': 0,
                      'manual': 1,
                      'random_means': 2,
                      'k_means++': 3}
    covariance_type_map = {'full': 0, 'diag': 1, 'tied_diag': 2}

    def __init__(self, #pylint: disable=too-many-arguments
                 init_param,
                 n_components=None,
                 init_centers=None,
                 covariance_type=None,
                 shared_covariance=False,
                 thread_ratio=None,
                 max_iter=None,
                 categorical_variable=None,
                 category_weight=None,
                 error_tol=None,
                 regularization=None,
                 random_seed=None):
        super(GaussianMixture, self).__init__()
        setattr(self, 'hanaml_parameters', pal_param_register())
        self.init_param = self._arg('init_param',
                                    init_param,
                                    self.init_param_map,
                                    required=True)
        self.n_components = self._arg('n_components', n_components, int)
        self.init_centers = self._arg('init_centers', init_centers, list)
        if init_param == 'manual':
            if init_centers is None:
                msg = ("Parameter init_centers is required when init_param is manual.")
                logger.error(msg)
                raise ValueError(msg)
            if n_components is not None:
                msg = ("Parameter n_components is not applicable when " +
                       "init_param is manual.")
                logger.error(msg)
                raise ValueError(msg)
        else:
            if n_components is None:
                msg = ("Parameter n_components is required when init_param is " +
                       "farthest_first_traversal, random_means and k_means++.")
                logger.error(msg)
                raise ValueError(msg)
            if init_centers is not None:
                msg = ("Parameter init_centers is not applicable when init_param is " +
                       "farthest_first_traversal, random_means and k_means++.")
                logger.error(msg)
                raise ValueError(msg)
        self.covariance_type = self._arg('covariance_type',
                                         covariance_type,
                                         self.covariance_type_map)
        self.shared_covariance = self._arg('shared_covariance', shared_covariance, bool)
        self.thread_ratio = self._arg('thread_ratio', thread_ratio, float)
        self.max_iter = self._arg('max_iter', max_iter, int)
        if isinstance(categorical_variable, str):
            categorical_variable = [categorical_variable]
        self.categorical_variable = self._arg('categorical_variable',
                                              categorical_variable, ListOfStrings)
        self.category_weight = self._arg('category_weight',
                                         category_weight, float)
        self.error_tol = self._arg('error_tol', error_tol, float)
        self.regularization = self._arg('regularization', regularization, float)
        self.random_seed = self._arg('random_seed', random_seed, int)

    @trace_sql
    def fit(self, data, key, features=None, categorical_variable=None):#pylint: disable=invalid-name, too-many-locals
        """
        Perform GMM clustering on input dataset.

        Parameters
        ----------
        data : DataFrame
            Data to be clustered.
        key : str
            Name of the ID column.
        features : list of str, optional
            List of strings specifying feature columns.

            If a list of features is not given, all the columns except the ID column
            are taken as features.
        categorical_variable : str or list of str, optional
            Specifies INTEGER column(s) that should be treated as categorical.

            Other INTEGER columns will be treated as continuous.
        """
        conn = data.connection_context
        require_pal_usable(conn)
        key = self._arg('key', key, str, required=True)
        features = self._arg('features', features, ListOfStrings)
        if isinstance(categorical_variable, str):
            categorical_variable = [categorical_variable]
        categorical_variable = self._arg('categorical_variable', categorical_variable, ListOfStrings)
        cols = data.columns
        cols.remove(key)
        if features is None:
            features = cols

        data_ = data[[key] + features]
        unique_id = str(uuid.uuid1()).replace('-', '_').upper()
        outputs = ['RESULT', 'MODEL', 'STATISTICS', 'PLACEHOLDER']
        outputs = ['#PAL_GMM_{}_TBL_{}_{}'.format(name, self.id, unique_id)
                   for name in outputs]
        result_tbl, model_tbl, statistics_tbl, placeholder_tbl = outputs
        init_param_data = self._prep_init_param(conn, data, key)

        param_rows = [
            ("INIT_MODE", self.init_param, None, None),
            ("COVARIANCE_TYPE", self.covariance_type, None, None),
            ("SHARED_COVARIANCE", self.shared_covariance, None, None),
            ("CATEGORY_WEIGHT", None, self.category_weight, None),
            ("MAX_ITERATION", self.max_iter, None, None),
            ("THREAD_RATIO", None, self.thread_ratio, None),
            ("ERROR_TOL", None, self.error_tol, None),
            ("REGULARIZATION", None, self.regularization, None),
            ("SEED", self.random_seed, None, None)
            ]
        if self.categorical_variable is not None:
            param_rows.extend(("CATEGORICAL_VARIABLE", None, None, variable)
                              for variable in self.categorical_variable)

        if categorical_variable is not None:
            param_rows.extend(("CATEGORICAL_VARIABLE", None, None, variable)
                              for variable in categorical_variable)

        try:
            call_pal_auto(conn,
                          'PAL_GMM',
                          data_,
                          init_param_data,
                          ParameterTable().with_data(param_rows),
                          *outputs)
        except dbapi.Error as db_err:
            logger.exception(str(db_err))
            try_drop(conn, outputs)
            raise

        # pylint: disable=attribute-defined-outside-init
        self.model_ = conn.table(model_tbl)
        self.labels_ = conn.table(result_tbl)
        self.stats_ = conn.table(statistics_tbl)

    def fit_predict(self, data, key, features=None, categorical_variable=None):#pylint: disable=invalid-name, too-many-locals
        """
        Perform GMM clustering on input dataset and return cluster membership
        probabilties for each data point.

        Parameters
        ----------
        data : DataFrame
            Data to be clustered.
        key : str
            Name of the ID column.
        features : list of str, optional
            List of strings specifying feature columns.

            If a list of features is not given, all the columns except the ID column
            are taken as features.
        categorical_variable : str or list of str, optional
            Specifies INTEGER column(s) specified that should be treated as categorical.

            Other INTEGER columns will be treated as continuous.

        Returns
        -------
        DataFrame
            Cluster membership probabilities.
        """
        self.fit(data, key, features, categorical_variable)
        return self.labels_

    def _prep_init_param(self, conn, data, key):

        unique_id = str(uuid.uuid1()).replace('-', '_').upper()
        init_param_tbl = "#PAL_GMM_INITIALIZE_PARAMETER_TBL_{}".format(unique_id)

        if self.n_components is not None:
            with conn.connection.cursor() as cur:
                cur.execute('CREATE LOCAL TEMPORARY COLUMN TABLE {} ("ID" INTEGER, "CLUSTER_NUMBER" INTEGER);'.format(init_param_tbl))
                cur.execute('INSERT INTO {} VALUES (0, {});'.format(init_param_tbl, self.n_components))
                init_param_data = conn.table(init_param_tbl)
        elif self.init_centers is not None:
            id_type = data.dtypes([key])[0][1]
            if id_type in ['VARCHAR', 'NVARCHAR']:
                id_type = '{}({})'.format(id_type, data.dtypes([key])[0][2])
            with conn.connection.cursor() as cur:
                cur.execute('CREATE LOCAL TEMPORARY COLUMN TABLE {} ("ID" INTEGER, "SEEDS" {});'.format(init_param_tbl, id_type))
                for idx, val in enumerate(self.init_centers):
                    cur.execute('INSERT INTO {} VALUES ( {}, {} );'.format(init_param_tbl, idx, val))
                init_param_data = conn.table(init_param_tbl)
        return init_param_data

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
        return super(GaussianMixture, self)._predict(data, key, features)
