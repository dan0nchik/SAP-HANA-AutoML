"""This module contains PAL wrapper for discriminant analysis algorithm.
The following class is available:

    * :class:`LinearDiscriminantAnalysis`
"""

#pylint: disable=too-many-locals, line-too-long, too-many-arguments, too-many-lines, relative-beyond-top-level
import logging
import uuid
from hdbcli import dbapi
from hana_ml.ml_exceptions import FitIncompleteError
from hana_ml.ml_base import try_drop
from .pal_base import (
    PALBase,
    ParameterTable,
    ListOfStrings,
    require_pal_usable,
    call_pal_auto
    )

logger = logging.getLogger(__name__)#pylint:disable=invalid-name

#pylint:disable=too-many-instance-attributes
class LinearDiscriminantAnalysis(PALBase):
    r"""
    Linear discriminant analysis for classification and data reduction.

    Parameters
    ----------

    regularization_type : {'mixing', 'diag', 'pseudo'}, optional

        The strategy for hanlding ill-conditioning or rank-deficiency
        of the empirical covariance matrix.

        Defaults to 'mixing'.

    regularization_amount : float, optional

        The convex mixing weight assigned to the diagonal matrix
        obtained from diagonal of the empirical covriance matrix.

        Valid range for this parameter is [0,1].

        Valid only when ``regularization_type`` is 'mixing'.

        Defaults to the smallest number in [0,1] that makes the
        regularized empircal covariance matrix invertible.

    projection : bool, optional

        Whether or not to compute the projection model.

        Defaults to True.

    Attributes
    ----------

    basic_info_ : DataFrame

        Basic information of the training data for linear discriminant analysis.

    priors_ : DataFrame

        The empirical pirors for each class in the training data.
    coef_ : DataFrame

        Coefficients (inclusive of intercepts) of each class' linear score function
        for the training data.

    proj_info : DataFrame

        Projection related info, such as standar deviations of the discriminants,
        variance proportaion to the total variance explained by each discriminant, etc.

    proj_model : DataFrame

         The projection matrix and overall means for features.

    Examples
    --------

    The training data for linear discriminant analysis:

    >>> df.collect()
         X1   X2   X3   X4            CLASS
    0   5.1  3.5  1.4  0.2      Iris-setosa
    1   4.9  3.0  1.4  0.2      Iris-setosa
    2   4.7  3.2  1.3  0.2      Iris-setosa
    3   4.6  3.1  1.5  0.2      Iris-setosa
    4   5.0  3.6  1.4  0.2      Iris-setosa
    5   5.4  3.9  1.7  0.4      Iris-setosa
    6   4.6  3.4  1.4  0.3      Iris-setosa
    7   5.0  3.4  1.5  0.2      Iris-setosa
    8   4.4  2.9  1.4  0.2      Iris-setosa
    9   4.9  3.1  1.5  0.1      Iris-setosa
    10  7.0  3.2  4.7  1.4  Iris-versicolor
    11  6.4  3.2  4.5  1.5  Iris-versicolor
    12  6.9  3.1  4.9  1.5  Iris-versicolor
    13  5.5  2.3  4.0  1.3  Iris-versicolor
    14  6.5  2.8  4.6  1.5  Iris-versicolor
    15  5.7  2.8  4.5  1.3  Iris-versicolor
    16  6.3  3.3  4.7  1.6  Iris-versicolor
    17  4.9  2.4  3.3  1.0  Iris-versicolor
    18  6.6  2.9  4.6  1.3  Iris-versicolor
    19  5.2  2.7  3.9  1.4  Iris-versicolor
    20  6.3  3.3  6.0  2.5   Iris-virginica
    21  5.8  2.7  5.1  1.9   Iris-virginica
    22  7.1  3.0  5.9  2.1   Iris-virginica
    23  6.3  2.9  5.6  1.8   Iris-virginica
    24  6.5  3.0  5.8  2.2   Iris-virginica
    25  7.6  3.0  6.6  2.1   Iris-virginica
    26  4.9  2.5  4.5  1.7   Iris-virginica
    27  7.3  2.9  6.3  1.8   Iris-virginica
    28  6.7  2.5  5.8  1.8   Iris-virginica
    29  7.2  3.6  6.1  2.5   Iris-virginica

    Set up an instance of LinearDiscriminantAnalysis model and train it:

    >>> lda = LinearDiscriminantAnalysis(regularization_type='mixing', projection=True)
    >>> lda.fit(data=df, features=['X1', 'X2', 'X3', 'X4'], label='CLASS')

    Check the coefficients of obtained linear discriminators and the projection model

    >>> lda.coef_.collect()
                 CLASS   COEFF_X1   COEFF_X2   COEFF_X3   COEFF_X4   INTERCEPT
    0      Iris-setosa  23.907391  51.754001 -34.641902 -49.063407 -113.235478
    1  Iris-versicolor   0.511034  15.652078  15.209568  -4.861018  -53.898190
    2   Iris-virginica -14.729636   4.981955  42.511486  12.315007  -94.143564
    >>> lda.proj_model_.collect()
             NAME        X1        X2        X3        X4
    0  DISCRIMINANT_1  1.907978  2.399516 -3.846154 -3.112216
    1  DISCRIMINANT_2  3.046794 -4.575496 -2.757271  2.633037
    2    OVERALL_MEAN  5.843333  3.040000  3.863333  1.213333

    Data to predict the class labels:

    >>> df_pred.collect()
         ID   X1   X2   X3   X4
    0    1  5.1  3.5  1.4  0.2
    1    2  4.9  3.0  1.4  0.2
    2    3  4.7  3.2  1.3  0.2
    3    4  4.6  3.1  1.5  0.2
    4    5  5.0  3.6  1.4  0.2
    5    6  5.4  3.9  1.7  0.4
    6    7  4.6  3.4  1.4  0.3
    7    8  5.0  3.4  1.5  0.2
    8    9  4.4  2.9  1.4  0.2
    9   10  4.9  3.1  1.5  0.1
    10  11  7.0  3.2  4.7  1.4
    11  12  6.4  3.2  4.5  1.5
    12  13  6.9  3.1  4.9  1.5
    13  14  5.5  2.3  4.0  1.3
    14  15  6.5  2.8  4.6  1.5
    15  16  5.7  2.8  4.5  1.3
    16  17  6.3  3.3  4.7  1.6
    17  18  4.9  2.4  3.3  1.0
    18  19  6.6  2.9  4.6  1.3
    19  20  5.2  2.7  3.9  1.4
    20  21  6.3  3.3  6.0  2.5
    21  22  5.8  2.7  5.1  1.9
    22  23  7.1  3.0  5.9  2.1
    23  24  6.3  2.9  5.6  1.8
    24  25  6.5  3.0  5.8  2.2
    25  26  7.6  3.0  6.6  2.1
    26  27  4.9  2.5  4.5  1.7
    27  28  7.3  2.9  6.3  1.8
    28  29  6.7  2.5  5.8  1.8
    29  30  7.2  3.6  6.1  2.5

    Perform predict() and check the result:

    >>> res_pred = lda.predict(data=df_pred,
    ...                        key='ID',
    ...                        features=['X1', 'X2', 'X3', 'X4'],
    ...                        verbose=False)
    >>> res_pred.collect()
        ID            CLASS       SCORE
    0    1      Iris-setosa  130.421263
    1    2      Iris-setosa   99.762784
    2    3      Iris-setosa  108.796296
    3    4      Iris-setosa   94.301777
    4    5      Iris-setosa  133.205924
    5    6      Iris-setosa  138.089829
    6    7      Iris-setosa  108.385827
    7    8      Iris-setosa  119.390933
    8    9      Iris-setosa   82.633689
    9   10      Iris-setosa  106.380335
    10  11  Iris-versicolor   63.346631
    11  12  Iris-versicolor   59.511996
    12  13  Iris-versicolor   64.286132
    13  14  Iris-versicolor   38.332614
    14  15  Iris-versicolor   54.823224
    15  16  Iris-versicolor   53.865644
    16  17  Iris-versicolor   63.581912
    17  18  Iris-versicolor   30.402809
    18  19  Iris-versicolor   57.411739
    19  20  Iris-versicolor   42.433076
    20  21   Iris-virginica  114.258002
    21  22   Iris-virginica   72.984306
    22  23   Iris-virginica   91.802556
    23  24   Iris-virginica   86.640121
    24  25   Iris-virginica   97.620689
    25  26   Iris-virginica  114.195778
    26  27   Iris-virginica   57.274694
    27  28   Iris-virginica  101.668525
    28  29   Iris-virginica   87.257782
    29  30   Iris-virginica  106.747065

    Data to project:

    >>> df_proj.collect()
        ID   X1   X2   X3   X4
    0    1  5.1  3.5  1.4  0.2
    1    2  4.9  3.0  1.4  0.2
    2    3  4.7  3.2  1.3  0.2
    3    4  4.6  3.1  1.5  0.2
    4    5  5.0  3.6  1.4  0.2
    5    6  5.4  3.9  1.7  0.4
    6    7  4.6  3.4  1.4  0.3
    7    8  5.0  3.4  1.5  0.2
    8    9  4.4  2.9  1.4  0.2
    9   10  4.9  3.1  1.5  0.1
    10  11  7.0  3.2  4.7  1.4
    11  12  6.4  3.2  4.5  1.5
    12  13  6.9  3.1  4.9  1.5
    13  14  5.5  2.3  4.0  1.3
    14  15  6.5  2.8  4.6  1.5
    15  16  5.7  2.8  4.5  1.3
    16  17  6.3  3.3  4.7  1.6
    17  18  4.9  2.4  3.3  1.0
    18  19  6.6  2.9  4.6  1.3
    19  20  5.2  2.7  3.9  1.4
    20  21  6.3  3.3  6.0  2.5
    21  22  5.8  2.7  5.1  1.9
    22  23  7.1  3.0  5.9  2.1
    23  24  6.3  2.9  5.6  1.8
    24  25  6.5  3.0  5.8  2.2
    25  26  7.6  3.0  6.6  2.1
    26  27  4.9  2.5  4.5  1.7
    27  28  7.3  2.9  6.3  1.8
    28  29  6.7  2.5  5.8  1.8
    29  30  7.2  3.6  6.1  2.5

    Do project and check the result:

    >>> res_proj = lda.project(data=df_proj,
    ...                        key='ID',
    ...                        features=['X1','X2','X3','X4'],
    ...                        proj_dim=2)
    >>> res_proj.collect()
        ID  DISCRIMINANT_1  DISCRIMINANT_2 DISCRIMINANT_3 DISCRIMINANT_4
    0    1       12.313584       -0.245578           None           None
    1    2       10.732231        1.432811           None           None
    2    3       11.215154        0.184080           None           None
    3    4       10.015174       -0.214504           None           None
    4    5       12.362738       -1.007807           None           None
    5    6       12.069495       -1.462312           None           None
    6    7       10.808422       -1.048122           None           None
    7    8       11.498220       -0.368435           None           None
    8    9        9.538291        0.366963           None           None
    9   10       10.898789        0.436231           None           None
    10  11       -1.208079        0.976629           None           None
    11  12       -1.894856       -0.036689           None           None
    12  13       -2.719280        0.841349           None           None
    13  14       -3.226081        2.191170           None           None
    14  15       -3.048480        1.822461           None           None
    15  16       -3.567804       -0.865854           None           None
    16  17       -2.926155       -1.087069           None           None
    17  18       -0.504943        1.045723           None           None
    18  19       -1.995288        1.142984           None           None
    19  20       -2.765274       -0.014035           None           None
    20  21      -10.727149       -2.301788           None           None
    21  22       -7.791979       -0.178166           None           None
    22  23       -8.291120        0.730808           None           None
    23  24       -7.969943       -1.211807           None           None
    24  25       -9.362513       -0.558237           None           None
    25  26      -10.029438        0.324116           None           None
    26  27       -7.058927       -0.877426           None           None
    27  28       -8.754272       -0.095103           None           None
    28  29       -8.935789        1.285655           None           None
    29  30       -8.674729       -1.208049           None           None

    """
    regularization_map = {'diag':1, 'pseudo':2, 'mixing':0}
    def __init__(self,
                 regularization_type=None,
                 regularization_amount=None,
                 projection=None):
        super(LinearDiscriminantAnalysis, self).__init__()
        self.regularization_type = self._arg('regularization_type',
                                             regularization_type,
                                             self.regularization_map)
        self.regularization_amount = self._arg('regularization_amount',
                                               regularization_amount,
                                               float)
        self.projection = self._arg('projection', projection, bool)
        self.basic_info_ = None
        self.priors_ = None
        self.coef_ = None
        self.proj_info_ = None
        self.proj_model_ = None
        self.model_ = None

    #pylint:disable=too-many-locals
    def fit(self, data, key=None, features=None, label=None):
        r"""
        Calculate linear discriminators from training data.

        Parameters
        ----------

        data : DataFrame

            Training data.

        key : str, optional

            Name of the ID colum. If not provided, it is assumed that

            the input data has no ID column.

        features : list of str, optional

            Names of the feature columns.

            If not provided, its defaults to all non-ID, non-label columns.

        label : str, optional

            Name of the class label.

            if not provided, it defaults to the final column.
        """
        #self.basic_info_ = None
        #self.priors_ = None
        #self.coef_ = None
        #self.proj_info = None
        #self.proj_model_ = None
        conn = data.connection_context
        require_pal_usable(conn)
        key = self._arg('key', key, str)
        features = self._arg('features', features, ListOfStrings)
        label = self._arg('label', label, str)
        cols = data.columns
        if key is not None:
            #id_col = [key]
            cols.remove(key)
        if label is None:
            label = cols[-1]
        cols.remove(label)
        if features is None:
            features = cols
        data_ = data[features + [label]]
        unique_id = str(uuid.uuid1()).replace('-', '_').upper()
        tables = ['BASIC_INFO', 'PRIORS', 'COEF', 'PROJ_INFO', 'PROJ_MODEL']
        tables = ['#PAL_LINEAR_DISCRIMINANT_ANALYSIS_{}_TBL_{}_{}'.format(name, self.id, unique_id)
                  for name in tables]
        basic_info_tbl, priors_tbl, coef_tbl, proj_info_tbl, proj_model_tbl = tables
        param_rows = [
            ('REGULARIZATION_TYPE', self.regularization_type, None, None),
            ('REGULARIZATION_AMOUNT', self. regularization_amount, None, None),
            ('DO_PROJECTION', self.projection, None, None)
        ]
        try:
            call_pal_auto(conn,
                          'PAL_LINEAR_DISCRIMINANT_ANALYSIS',
                          data_,
                          ParameterTable().with_data(param_rows),
                          basic_info_tbl,
                          priors_tbl,
                          coef_tbl,
                          proj_info_tbl,
                          proj_model_tbl)
        except dbapi.Error as db_err:
            #msg = ("HANA error while attempting to fit linear discriminat analysis model.")
            logger.exception(str(db_err))
            try_drop(conn, tables)
            raise
        self.basic_info_ = conn.table(basic_info_tbl)
        self.priors_ = conn.table(priors_tbl)
        self.coef_ = conn.table(coef_tbl)
        self.model_ = [self.priors_, self.coef_]
        if self.projection is not False:
            self.proj_info_ = conn.table(proj_info_tbl)
            self.proj_model_ = conn.table(proj_model_tbl)

    def predict(self, data, key, features=None, verbose=None):
        r"""
        Predict class labels using fitted linear discriminators.

        Parameters
        ----------

        data : DataFrame

            Data for predicting the class labels.

        key : str

            Name of the ID column.

        features : list of str, optional

            Name of the feature columns.
            If not provided, defaults to all non-ID columns.

        verbose : bool, optional

            Whether or not outputs scores of all classes.

            If False, only score of the predicted class will be outputed.

            Defaults to False.

        Returns
        -------

        DataFrame

            Predicted class labels and the corresponding scores, structured as follows:

              - ID: with the same name and data type as ``data``'s ID column.
              - CLASS: with the same name and data type as training data's label column
              - SCORE: type double, socre of the predicted class.

        """
        conn = data.connection_context
        require_pal_usable(conn)
        if getattr(self, 'model_') is None:
            raise FitIncompleteError("Model not initialized. Perform a fit first.")
        key = self._arg('key', key, str, required=True)
        features = self._arg('features', features, ListOfStrings)
        verbose = self._arg('verbose', verbose, bool)
        cols = data.columns
        cols.remove(key)
        if features is None:
            features = cols
        data_ = data[[key] + features]
        unique_id = str(uuid.uuid1()).replace('-', '_').upper()

        result_tbl = '#PAL_LINEAR_DISCRIMINANT_ANALYSIS_RESULT_TBL_{}_{}'.format(self.id, unique_id)

        param_rows = [
            ('VERBOSE_OUTPUT', verbose, None, None)
        ]
        try:
            call_pal_auto(conn,
                          'PAL_LINEAR_DISCRIMINANT_ANALYSIS_CLASSIFY',
                          data_,
                          self.model_[0],
                          self.model_[1],
                          ParameterTable().with_data(param_rows),
                          result_tbl)
        except dbapi.Error as db_err:
            #msg = ("HANA error during linear discriminant analysis prediction.")
            logger.exception(str(db_err))
            try_drop(conn, result_tbl)
            raise
        return conn.table(result_tbl)

    def project(self, data, key, features=None, proj_dim=None):
        r"""
        Project `data` into lower dimensional spaces using fitted LDA projection model.

        Parameters
        ----------

        data : DataFrame

            Data for linear discriminant projection.

        key : str

            Name of the ID column.

        features : list of str, optional

            Name of the feature columns.

            If not provided, defaults to all non-ID columns.

        proj_dim : int, optional

            Dimension of the projected space, equivalent to the number
            of discriminant used for projection.

            Defaults to the number of obtained discriminants.

        Returns
        -------

        DataFrame

            Projected data, structured as follows:
                - 1st column: ID, with the same name and data type as ``data`` for projection.
                - other columns with name DISCRIMINANT_i, where i iterates from 1 to the number
                  of elements in ``features``, data type DOUBLE.

        """
        conn = data.connection_context
        require_pal_usable(conn)
        if getattr(self, 'proj_model_', None) is not None:
            proj_model = self.proj_model_
        else:
            msg = ("Projection model not initialized. "+
                   "Set `projection` to True and perform a fit first.")
            raise FitIncompleteError(msg)
        key = self._arg('key', key, str, required=True)
        features = self._arg('features', features, ListOfStrings)
        proj_dim = self._arg('proj_dim', proj_dim, int)

        cols = data.columns
        cols.remove(key)
        if features is None:
            features = cols
        data_ = data[[key] + features]

        unique_id = str(uuid.uuid1()).replace('-', '_').upper()

        projected_tbl = ('#PAL_LINEAR_DISCRIMINANT_'+
                         'ANALYSIS_PROJECTION_TBL_{}_{}'.format(self.id, unique_id))
        param_rows = [
            ('DISCRIMINANT_NUMBER', proj_dim, None, None)
        ]

        try:
            call_pal_auto(conn,
                          'PAL_LINEAR_DISCRIMINANT_ANALYSIS_PROJECT',
                          data_,
                          proj_model,
                          ParameterTable().with_data(param_rows),
                          projected_tbl)
        except dbapi.Error as db_err:
            #msg = 'HANA error during linear discriminant analysis projection.'
            logger.exception(str(db_err))
            try_drop(conn, projected_tbl)
            raise

        return conn.table(projected_tbl)
