"""
This module contains wrappers for PAL regression algorithms.

The following classes are available:

    * :class:`PolynomialRegression`
    * :class:`GLM`
    * :class:`ExponentialRegression`
    * :class:`BiVariateGeometricRegression`
    * :class:`BiVariateNaturalLogarithmicRegression`
    * :class:`CoxProportionalHazardModel`
"""

# pylint: disable=too-many-lines, line-too-long, too-many-arguments, too-many-branches, too-many-instance-attributes
import logging
import sys
import uuid
from hdbcli import dbapi
from hana_ml.ml_exceptions import FitIncompleteError
from .sqlgen import trace_sql
from .pal_base import (
    PALBase,
    ParameterTable,
    ListOfStrings,
    ListOfTuples,
    pal_param_register,
    try_drop,
    require_pal_usable,
    call_pal_auto
)
from . import metrics
logger = logging.getLogger(__name__) #pylint: disable=invalid-name
if sys.version_info.major == 2:
    #pylint: disable=undefined-variable
    _INTEGER_TYPES = (int, long)
    _STRING_TYPES = (str, unicode)
else:
    _INTEGER_TYPES = (int,)
    _STRING_TYPES = (str,)

class PolynomialRegression(PALBase):
    r"""
    Polynomial regression is an approach to model the relationship between a scalar variable y and a variable denoted X. In polynomial regression,
    data is modeled using polynomial functions, and unknown model parameters are estimated from the data. Such models are called polynomial models.

    Parameters
    ----------

    degree : int
        Degree of the polynomial model.
    decomposition : {'LU', 'QR', 'SVD', 'Cholesky'}, optional
        Matrix factorization type to use. Case-insensitive.

          - 'LU': LU decomposition.
          - 'QR': QR decomposition.
          - 'SVD': singular value decomposition.
          - 'Cholesky': Cholesky(LDLT) decomposition.

        Defaults to QR decomposition.
    adjusted_r2 : bool, optional
        If true, include the adjusted R2 value in the statistics table.

        Defaults to False.
    pmml_export : {'no', 'single-row', 'multi-row'}, optional
        Controls whether to output a PMML representation of the model,
        and how to format the PMML. Case-insensitive.

            - 'no' or not provided: No PMML model.
            - 'single-row': Exports a PMML model in a maximum of
              one row. Fails if the model doesn't fit in one row.
            - 'multi-row': Exports a PMML model, splitting it
              across multiple rows if it doesn't fit in one.

        Prediction does not require a PMML model.
    thread_ratio : float, optional
        Controls the proportion of available threads to use for fitting.

        The value range is from 0 to 1, where 0 indicates a single thread,
        and 1 indicates up to all available threads. Values between 0 and 1
        will use that percentage of available threads.

        Values outside this range tell PAL to heuristically determine the number of threads to use.

        Defaults to 0.
    resampling_method : {'cv', 'bootstrap'}, optional
        Specifies the resampling method for model evaluation/parameter selection.

        If no value is specified for this parameter, neither model evaluation
        nor parameter selection is activated.

        Must be set together with ``evaluation_metric``.

        No default value.
    evaluation_metric : {'rmse'}, optional
        Specifies the evaluation metric for model evaluation or parameter selection.

        Must be set together with ``resampling_method``.

        No default value.
    fold_num : int, optional
        Specifies the fold number for the cross validation method.
        Mandatory and valid only when ``resampling_method`` is set to 'cv'.

        No default value.
    repeat_times : int, optional
        Specifies the number of repeat times for resampling.

        Defaults to 1.
    search_strategy : {'grid', 'random'}, optional
        Specifies the method to activate parameter selection.

        No default value.
    random_search_times : int, optional
        Specifies the number of times to randomly select candidate parameters for selection.

        Mandatory and valid when ``search_strategy`` is set to 'random'.

        No default value.
    random_state : int, optional
        Specifies the seed for random generation. Use system time when 0 is specified.

        Defaults to 0.
    timeout : int, optional
        Specifies maximum running time for model evaluation or parameter
        selection, in seconds.

        No timeout when 0 is specified.

        Defaults to 0.
    progress_indicator_id : str, optional
        Sets an ID of progress indicator for model evaluation or parameter selection.

        No progress indicator is active if no value is provided.

        No default value.
    degree_values : list of int, optional
        Specifies values of ``degree`` to be selected.

        Only valid when ``search_strategy`` is specified.

        No default value.
    degree_range : list of int, optional
        Specifies range of ``degree`` to be selected.

        Only valid when ``search_strategy`` is specified.

        No default value.

    Attributes
    ----------
    coefficients_ : DataFrame
        Fitted regression coefficients.
    pmml_ : DataFrame
        PMML model. Set to None if no PMML model was requested.
    fitted_ : DataFrame
        Predicted dependent variable values for training data.
        Set to None if the training data has no row IDs.
    statistics_ : DataFrame
        Regression-related statistics, such as mean squared error.
    optim_param_ : DataFrame
        If cross validation is enabled, the optimal parameters will be selected.

    Examples
    --------

    Training data (based on :math:`y = x^3 - 2x^2 + 3x + 5`, with noise):

    >>> df.collect()
       ID    X       Y
    0   1  0.0   5.048
    1   2  1.0   7.045
    2   3  2.0  11.003
    3   4  3.0  23.072
    4   5  4.0  49.041

    Training the model:

    >>> pr = PolynomialRegression(degree=3)
    >>> pr.fit(data=df, key='ID')

    Prediction:

    >>> df2.collect()
       ID    X
    0   1  0.5
    1   2  1.5
    2   3  2.5
    3   4  3.5
    >>> pr.predict(data=df2, key='ID').collect()
       ID      VALUE
    0   1   6.157063
    1   2   8.401269
    2   3  15.668581
    3   4  33.928501

    Ideal output:

    >>> df2.select('ID', ('POWER(X, 3)-2*POWER(X, 2)+3*x+5', 'Y')).collect()
       ID       Y
    0   1   6.125
    1   2   8.375
    2   3  15.625
    3   4  33.875
    """
    # pylint: disable=too-many-arguments,too-many-instance-attributes, too-many-locals, too-many-statements
    decomposition_map = {'lu': 0, 'qr': 1, 'svd': 2, 'cholesky': 5}
    pmml_export_map = {'no': 0, 'single-row': 1, 'multi-row': 2}
    def __init__(self,
                 degree=None,#Polynomial_num
                 decomposition=None,#alg
                 adjusted_r2=None,
                 pmml_export=None,
                 thread_ratio=0.0,
                 resampling_method=None,
                 evaluation_metric=None,
                 fold_num=None,
                 repeat_times=None,
                 search_strategy=None,
                 random_search_times=None,
                 random_state=None,
                 timeout=None,
                 progress_indicator_id=None,
                 degree_values=None,
                 degree_range=None):
        super(PolynomialRegression, self).__init__()
        setattr(self, 'hanaml_parameters', pal_param_register())
        self.decomposition = self._arg('decomposition', decomposition, self.decomposition_map)
        self.adjusted_r2 = self._arg('adjusted_r2', adjusted_r2, bool)
        self.pmml_export = self._arg('pmml_export', pmml_export, self.pmml_export_map)
        self.thread_ratio = self._arg('thread_ratio', thread_ratio, float)
        self.resampling_method_map = {'cv': 'cv', 'bootstrap': 'bootstrap'}
        self.evaluation_metric_map = {'rmse': 'RMSE'}
        self.search_strategy_map = {'grid': 'grid', 'random': 'random'}
        self.resampling_method = self._arg('resampling_method', resampling_method,
                                           self.resampling_method_map)
        self.evaluation_metric = self._arg('evaluation_metric', evaluation_metric,
                                           self.evaluation_metric_map)
        self.fold_num = self._arg('fold_num', fold_num, int)
        self.repeat_times = self._arg('repeat_times', repeat_times, int)
        self.search_strategy = self._arg('search_strategy', search_strategy,
                                         self.search_strategy_map)
        self.random_search_times = self._arg('random_search_times', random_search_times, int)
        self.random_state = self._arg('random_state', random_state, int)
        self.timeout = self._arg('timeout', timeout, int)
        self.progress_indicator_id = self._arg('progress_indicator_id',
                                               progress_indicator_id, str)
        self.degree_values = self._arg('degree_values', degree_values, list)
        self.degree_range = self._arg('degree_range', degree_range, list)
        self.degree = self._arg('degree', degree, int, required=self.resampling_method is None)
        search_param_count = 0
        for param in (self.resampling_method, self.evaluation_metric):
            if param is not None:
                search_param_count += 1
        if search_param_count not in (0, 2):
            msg = ("`resampling_method`, and `evaluation_metric` must be set together.")
            logger.error(msg)
            raise ValueError(msg)
        if self.search_strategy is not None and self.resampling_method is None:
            msg = ("`search_strategy` cannot be set if `resampling_method` is not specified.")
            logger.error(msg)
            raise ValueError(msg)
        if self.resampling_method == 'cv' and self.fold_num is None:
            msg = ("`fold_num` must be set when "+
                   "`resampling_method` is set as 'cv'.")
            logger.error(msg)
            raise ValueError(msg)
        if self.resampling_method != 'cv' and self.fold_num is not None:
            msg = ("`fold_num` is not valid when parameter selection is not" +
                   " enabled, or `resampling_method` is not set as 'cv'.")
            logger.error(msg)
            raise ValueError(msg)
        if self.search_strategy == 'random' and self.random_search_times is None:
            msg = ("`random_search_times` must be set when "+
                   "`search_strategy` is set as 'random'.")
            logger.error(msg)
            raise ValueError(msg)
        if self.search_strategy in (None, 'grid') and self.random_search_times is not None:
            msg = ("`random_search_times` is not valid " +
                   "when parameter selection is not enabled"+
                   ", or `search_strategy` is not set as 'random'.")
            logger.error(msg)
            raise ValueError(msg)
        if self.search_strategy is None:
            if self.degree_values is not None:
                msg = ("`degree_values` can only be specified "+
                       "when parameter selection is enabled.")
                logger.error(msg)
                raise ValueError(msg)
            if self.degree_range is not None:
                msg = ("`degree_range` can only be specified "+
                       "when parameter selection is enabled.")
                logger.error(msg)
                raise ValueError(msg)
        if self.search_strategy is not None:
            degree_set_count = 0
            for degree_set in (self.degree, self.degree_values, self.degree_range):
                if degree_set is not None:
                    degree_set_count += 1
            if degree_set_count > 1:
                msg = ("The following paramters cannot be specified together:" +
                       "`degree`, `degree_values`, `degree_range`.")
                logger.error(msg)
                raise ValueError(msg)
            if degree_set_count == 0:
                msg = ("One of the following paramters must be set: " +
                       "`degree`, `degree_values`, `degree_range`.")
                logger.error(msg)
                raise ValueError(msg)
            if self.degree_values is not None:
                if not all(isinstance(t, int) for t in self.degree_values):
                    msg = "Valid values of `degree_values` must be a list of integer."
                    logger.error(msg)
                    raise TypeError(msg)
            if self.degree_range is not None:
                rsz = [3] if self.search_strategy == 'grid'else [2, 3]
                if not len(self.degree_range) in rsz or not all(isinstance(t, int) for t in self.degree_range):#pylint:disable=line-too-long
                    msg = ("The provided `degree_range` is either not "+
                           "a list of integer, or it contains wrong number of values.")
                    logger.error(msg)
                    raise TypeError(msg)

    @trace_sql
    def fit(self, data, key=None, features=None, label=None):
        r"""
        Fit regression model based on training data.

        Parameters
        ----------
        data : DataFrame
            Training data.
        key : str, optional
            Name of the ID column. If ``key`` is not provided, it is assumed
            that the input has no ID column.
        features : list of str, optional
            Names of the feature columns.

            Since the underlying PAL procedure for polynomial regression algorithm only supports one feature,
            this list can only contain one element.

            If ``features`` is not provided, ``data`` must have exactly 1
            non-ID, non-label column, and ``features`` defaults to that
            column.
        label : str, optional
            Name of the dependent variable.

            Defaults to the last column. (This is not the PAL default.)

        """
        # pylint: disable=too-many-locals,too-many-statements
        conn = data.connection_context
        require_pal_usable(conn)
        conn.sql_tracer.set_sql_trace(self, self.__class__.__name__, 'Fit')
        key = self._arg('key', key, str)
        features = self._arg('features', features, ListOfStrings)
        label = self._arg('label', label, str)
        cols = data.columns
        if key is not None:
            maybe_id = [key]
            cols.remove(key)
        else:
            maybe_id = []
        if label is None:
            label = cols[-1]
        cols.remove(label)
        if features is None:
            features = cols
        if len(features) != 1:
            msg = ('PAL polynomial regression requires exactly one' +
                   ' feature column.')
            logger.error(msg)
            raise TypeError(msg)
        data_ = data[maybe_id + [label] + features]
        unique_id = str(uuid.uuid1()).replace('-', '_').upper()
        outputs = ['COEF', 'PMML', 'FITTED', 'STATS', 'OPTIMAL_PARAM']
        outputs = ['#PAL_POLYNOMIAL_REGRESSION_{}_TBL_{}_{}'.format(name, self.id, unique_id)
                   for name in outputs]
        coef_tbl, pmml_tbl, fitted_tbl, stats_tbl, opt_param_tbl = outputs
        param_rows = [('ALG', self.decomposition, None, None),
                      ('ADJUSTED_R2', self.adjusted_r2, None, None),
                      ('PMML_EXPORT', self.pmml_export, None, None),
                      ('HAS_ID', key is not None, None, None),
                      ('RESAMPLING_METHOD', None, None, self.resampling_method),
                      ('EVALUATION_METRIC', None, None, self.evaluation_metric),
                      ('SEED', self.random_state, None, None),
                      ('REPEAT_TIMES', self.repeat_times, None, None),
                      ('PARAM_SEARCH_STRATEGY', None, None, self.search_strategy),
                      ('FOLD_NUM', self.fold_num, None, None),
                      ('RANDOM_SEARCH_TIMES', self.random_search_times, None, None),
                      ('TIMEOUT', self.timeout, None, None),
                      ('PROGRESS_INDICATOR_ID', None, None, self.progress_indicator_id),
                      ('THREAD_RATIO', None, self.thread_ratio, None)]
        if self.degree is not None:
            param_rows.extend([('POLYNOMIAL_NUM', self.degree, None, None)])
        if self.degree_range is not None:
            val = str(self.degree_range)
            param_rows.extend([('POLYNOMIAL_NUM_RANGE', None, None, val)])
        if self.degree_values is not None:
            val = str(self.degree_values).replace('[', '{').replace(']', '}')
            param_rows.extend([('POLYNOMIAL_NUM_VALUES', None, None, val)])
        try:
            call_pal_auto(conn,
                          'PAL_POLYNOMIAL_REGRESSION',
                          data_,
                          ParameterTable().with_data(param_rows),
                          *outputs)
        except dbapi.Error as db_err:
            logger.exception(str(db_err))
            try_drop(conn, outputs)
            raise
        # pylint: disable=attribute-defined-outside-init
        self.coefficients_ = conn.table(coef_tbl)
        self.pmml_ = conn.table(pmml_tbl) if self.pmml_export else None
        self.fitted_ = conn.table(fitted_tbl) if key is not None else None
        self.statistics_ = conn.table(stats_tbl)
        self.optim_param_ = conn.table(opt_param_tbl)
        self.model_ = self.coefficients_

    @trace_sql
    def predict(self, data, key, features=None, model_format=None, thread_ratio=0.0):
        r"""
        Predict dependent variable values based on fitted model.

        Parameters
        ----------
        data : DataFrame
            Independent variable values used for prediction.

        key : str
            Name of the ID column.

        features : list of str, optional
            Names of the feature columns.

            Since the underlying PAL procedure for polynomial regression only supports one feature,
            this list can only contain one element.

            If ``features`` is not provided, ``data`` must have exactly 1
            non-ID column, and ``features`` defaults to that column.

        model_format : int, optional

            - 0: coefficient
            - 1: pmml

        thread_ratio : float, optional
            Controls the proportion of available threads to use for prediction.

            The value range is from 0 to 1, where 0 indicates a single thread,
            and 1 indicates up to all available threads.

            Values between 0 and 1 will use that percentage of available threads.

            Values outside this range tell PAL to heuristically determine the number of threads to use.

            Defaults to 0.

        Returns
        -------
        DataFrame

            Predicted values, structured as follows:

                - ID column, with same name and type as ``data``'s ID column.
                - VALUE, type DOUBLE, representing predicted values.

        .. note::

            predict() will pass the ``pmml_`` table to PAL as the model
            representation if there is a ``pmml_`` table, or the ``coefficients_``
            table otherwise.
        """
        # pylint: disable=too-many-locals

        # SQLTRACE
        conn = data.connection_context
        require_pal_usable(conn)
        conn.sql_tracer.set_sql_trace(self, self.__class__.__name__, 'Predict')
        key = self._arg('key', key, str, required=True)
        features = self._arg('features', features, ListOfStrings)
        thread_ratio = self._arg('thread_ratio', thread_ratio, float)
        if getattr(self, 'pmml_', None) is not None:
            model = self.pmml_
            model_format = 1
        elif hasattr(self, 'model_'):
            model = self.model_
        else:
            raise FitIncompleteError("Model not initialized. Perform a fit first.")
        if features is None:
            cols = data.columns
            cols.remove(key)
            features = cols
        if len(features) != 1:
            msg = ('PAL polynomial regression requires exactly one ' +
                   'feature column.')
            logger.error(msg)
            raise TypeError(msg)

        data_ = data[[key] + features]
        unique_id = str(uuid.uuid1()).replace('-', '_').upper()
        fitted_tbl = '#PAL_POLYNOMIAL_REGRESSION_FITTED_TBL_{}_{}'.format(self.id, unique_id)
        param_rows = [('THREAD_RATIO', None, thread_ratio, None),
                      ('MODEL_FORMAT', model_format, None, None)]
        try:
            call_pal_auto(conn,
                          'PAL_POLYNOMIAL_REGRESSION_PREDICT',
                          data_,
                          model,
                          ParameterTable().with_data(param_rows),
                          fitted_tbl)
        except dbapi.Error as db_err:
            logger.exception(str(db_err))
            try_drop(conn, fitted_tbl)
            raise
        return conn.table(fitted_tbl)

    def score(self, data, key, features=None, label=None):
        r"""
        Returns the coefficient of determination R2 of the prediction.

        Parameters
        ----------
        data : DataFrame
            Data on which to assess model performance.
        key : str
            Name of the ID column.
        features : list of str, optional
            Names of the feature columns.

            Since the underlying PAL procedure for polynomial regression prediction only supports one feature,
            this list can only contain one element.

            If ``features`` is not provided, ``data`` must have exactly 1
            non-ID, non-label column, and ``features`` defaults to that
            column.
        label : str, optional
            Name of the dependent variable.

            Defaults to the last column(this is not the PAL default.)

        Returns
        -------
        float

            The coefficient of determination R2 of the prediction on the
            given data.
        """

        # SQLTRACE
        conn = data.connection_context
        require_pal_usable(conn)
        conn.sql_tracer.set_sql_trace(self, self.__class__.__name__, 'Score')
        key = self._arg('key', key, str, required=True)
        features = self._arg('features', features, ListOfStrings)
        label = self._arg('label', label, str)
        cols = data.columns
        cols.remove(key)
        if label is None:
            label = cols[-1]
        cols.remove(label)
        if features is None:
            features = cols
        if len(features) != 1:
            msg = ('PAL polynomial regression requires exactly one ' +
                   'feature column.')
            logger.error(msg)
            raise TypeError(msg)
        prediction = self.predict(data, key=key, features=features)
        prediction = prediction.rename_columns(['ID_P', 'PREDICTION'])
        actual = data.select(key, label).rename_columns(['ID_A', 'ACTUAL'])
        joined = actual.join(prediction, 'ID_P=ID_A').select('ACTUAL', 'PREDICTION')
        return metrics.r2_score(joined,
                                label_true='ACTUAL',
                                label_pred='PREDICTION')

class GLM(PALBase):
    r"""
    Regression by a generalized linear model, based on PAL_GLM. Also supports
    ordinal regression.

    Parameters
    ----------

    family : {'gaussian', 'normal', 'poisson', 'binomial', 'gamma', 'inversegaussian', 'negativebinomial', 'ordinal'}, optional
        The kind of distribution the dependent variable outcomes are
        assumed to be drawn from.

        Defaults to 'gaussian'.
    link : str, optional
        GLM link function. Determines the relationship between the linear
        predictor and the predicted response.

        Default and allowed values depend on ``family``. 'inverse' is accepted as a synonym of
        'reciprocal'.

        ================ ============= ========================================
        family           default link  allowed values of link
        ================ ============= ========================================
        gaussian         identity      identity, log, reciprocal
        poisson          log           identity, log
        binomial         logit         logit, probit, comploglog, log
        gamma            reciprocal    identity, reciprocal, log
        inversegaussian  inversesquare inversesquare, identity, reciprocal, log
        negativebinomial log           identity, log, sqrt
        ordinal          logit         logit, probit, comploglog
        ================ ============= ========================================

    solver : {'irls', 'nr', 'cd'}, optional
        Optimization algorithm to use.

            - 'irls': Iteratively re-weighted least squares.
            - 'nr': Newton-Raphson.
            - 'cd': Coordinate descent. (Picking coordinate descent activates
              elastic net regularization.)

        Defaults to 'irls', except when ``family`` is 'ordinal'.

        Ordinal regression requires (and defaults to) 'nr', and Newton-Raphson
        is not supported for other values of ``family``.
    handle_missing_fit : {'skip', 'abort', 'fill_zero'}, optional
        How to handle data rows with missing independent variable values
        during fitting.

            - 'skip': Don't use those rows for fitting.
            - 'abort': Throw an error if missing independent variable values
              are found.
            - 'fill_zero': Replace missing values with 0.

        Defaults to 'skip'.
    quasilikelihood : bool, optional
        If True, enables the use of quasi-likelihood to estimate overdispersion.

        Defaults to False.
    max_iter : int, optional
        Maximum number of optimization iterations.

        Defaults to 100 for IRLS
        and Newton-Raphson.

        Defaults to 100000 for coordinate descent.
    tol : float, optional
        Stopping condition for optimization.

        Defaults to 1e-8 for IRLS,
        1e-6 for Newton-Raphson, and 1e-7 for coordinate descent.
    significance_level : float, optional
        Significance level for confidence intervals and prediction intervals.

        Defaults to 0.05.
    output_fitted : bool, optional
        If True, create the ``fitted_`` DataFrame of fitted response values
        for training data in fit.

        Defaults to False.
    alpha : float, optional
        Elastic net mixing parameter. Only accepted when using coordinate
        descent. Should be between 0 and 1 inclusive.

        Defaults to 1.0.

    lamb : float, optional
        Coefficient(lambda) value for elastic-net regularization.

        Valid only when ``solver`` is 'cd'.

        No default value.
    num_lambda : int, optional
        The number of lambda values. Only accepted when using coordinate
        descent.

        Defaults to 100.
    lambda_min_ratio : float, optional
        The smallest value of lambda, as a fraction of the maximum lambda,
        where lambda_max is the smallest value for which all coefficients
        are zero. Only accepted when using coordinate descent.

        Defaults to 0.01 when the number of observations is smaller than the number
        of covariates, and 0.0001 otherwise.
    categorical_variable : list of str, optional
        INTEGER columns specified in this list will be treated as categorical
        data. Other INTEGER columns will be treated as continuous.
    ordering : list of str or list of int, optional
        Specifies the order of categories for ordinal regression.

        The default is numeric order for ints and alphabetical order for
        strings.
    thread_ratio : float, optional
        Controls the proportion of available threads to use for fitting.

        The value range is from 0 to 1, where 0 indicates a single thread,
        and 1 indicates up to all available threads.

        Values between 0 and 1 will use that percentage of available threads.

        Values outside this range tell PAL to heuristically determine the number of threads to use.

        Defaults to 0.
    resampling_method : {'cv', 'bootstrap'}, optional

        Specifies the resampling method for model evaluation/parameter selection.

        If no value is specified for this parameter, neither model evaluation
        nor parameter selection is activated.

        Must be set together with ``evaluation_metric``.

        No default value.

    evaluation_metric : {'rmse', 'mae', 'error_rate'}, optional

        Specifies the evaluation metric for model evaluation or parameter selection.

        Must be set together with ``resampling_method``.

        'error_rate' applies only for ordinal regression.

        No default value.

    fold_num : int, optional

        Specifies the fold number for the cross validation method.

        Mandatory and valid only when `resampling_method` is set to 'cv'.

        No default value.

    repeat_times : int, optional

        Specifies the number of repeat times for resampling.

        Defaults to 1.

    search_strategy : {'grid', 'random'}, optional

        Specifies the method to activate parameter selection.

        No default value.

    random_search_times : int, optional

        Specifies the number of times to randomly select candidate parameters for selection.

        Mandatory and valid when `search_strategy` is set to 'random'.

        No default value.

    random_state : int, optional

        Specifies the seed for random generation. Use system time when 0 is specified.

        Defaults to 0.

    timeout : int, optional

        Specifies maximum running time for model evaluation or parameter

        selection, in seconds.

        No timeout when 0 is specified.

        Defaults to 0.

    progress_indicator_id : str, optional

        Sets an ID of progress indicator for model evaluation or parameter selection.

        No progress indicator is active if no value is provided.

        No default value.

    param_values : dict or list of tuples, optional

        Specifies values of specific parameters to be selected.

        Valid only when parameter selection is activated.

        Specified parameters could be ``link``, ``lamb`` and ``alpha``.

        No default value.

    param_range : dict or list of tuples, optional

        Specifies range of specific parameters to be selected.

        Valid only when parameter selection is activated.

        Specified parameters could be ``lamb``, ``alpha``.

        No default value.

    Attributes
    ----------

    statistics_ : DataFrame
        Training statistics and model information other than the
        coefficients and covariance matrix.
    coef_ : DataFrame
        Model coefficients.
    covmat_ : DataFrame
        Covariance matrix. Set to None for coordinate descent.
    fitted_ : DataFrame
        Predicted values for the training data. Set to None if
        ``output_fitted`` is False.

    Examples
    --------
    Training data:

    >>> df.collect()
       ID  Y  X
    0   1  0 -1
    1   2  0 -1
    2   3  1  0
    3   4  1  0
    4   5  1  0
    5   6  1  0
    6   7  2  1
    7   8  2  1
    8   9  2  1

    Fitting a GLM on that data:

    >>> glm = GLM(solver='irls', family='poisson', link='log')
    >>> glm.fit(data=df, key='ID', label='Y')

    Performing prediction:

    >>> df2.collect()
       ID  X
    0   1 -1
    1   2  0
    2   3  1
    3   4  2
    >>> glm.predict(data=df2, key='ID')[['ID', 'PREDICTION']].collect()
       ID           PREDICTION
    0   1  0.25543735346197155
    1   2    0.744562646538029
    2   3   2.1702915689746476
    3   4     6.32608352871737
    """
    family_link_values = {
        'gaussian': ['identity', 'log', 'reciprocal', 'inverse'],
        'normal': ['identity', 'log', 'reciprocal', 'inverse'],
        'poisson': ['identity', 'log'],
        'binomial': ['logit', 'probit', 'comploglog', 'log'],
        'gamma': ['identity', 'reciprocal', 'inverse', 'log'],
        'inversegaussian': ['inversesquare', 'identity', 'reciprocal',
                            'inverse', 'log'],
        'negativebinomial': ['identity', 'log', 'sqrt'],
        'ordinal': ['logit', 'probit', 'comploglog']
    }
    solvers = ['irls', 'nr', 'cd']
    handle_missing_fit_map = {
        'abort': 0,
        'skip': 1,
        'fill_zero': 2
    }
    handle_missing_predict_map = {
        'skip': 1,
        'fill_zero': 2
    }
    resampling_methods = ['cv', 'bootstrap']
    evaluation_metrics = ['rmse', 'mae', 'error_rate']
    search_strategies = ['grid', 'random']
    __cv_params = ['link', 'alpha', 'lamb']
    __cv_param_map = {'link' : 'LINK', 'alpha' : 'ENET_ALPHA',
                      'lamb' : 'ENET_LAMBDA'}
    def __init__(self,#pylint:disable=too-many-statements
                 family=None,
                 link=None,
                 solver=None,
                 handle_missing_fit=None,
                 quasilikelihood=None,
                 max_iter=None,
                 tol=None,
                 significance_level=None,
                 output_fitted=None,
                 alpha=None,
                 lamb=None,
                 num_lambda=None,
                 lambda_min_ratio=None,
                 categorical_variable=None,
                 ordering=None,
                 thread_ratio=0.0,
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
                 param_range=None):
        # pylint:disable=too-many-arguments
        # pylint:disable=too-many-locals
        # pylint:disable=too-many-branches
        super(GLM, self).__init__()
        setattr(self, 'hanaml_parameters', pal_param_register())
        self.family = self._arg('family',
                                family,
                                {x:x for x in self.family_link_values})
        links = self.family_link_values['gaussian' if self.family is None
                                        else self.family]
        self.link = self._arg('link',
                              link,
                              {x:x for x in links})
        self.solver = self._arg('solver',
                                solver,
                                {x:x for x in self.solvers})
        if self.family == 'ordinal':
            self.solver = 'nr'
        elif self.solver == 'nr':
            msg = "Newton-Raphson method is only for ordinal regression."
            logger.error(msg)
            raise ValueError(msg)
        self.handle_missing_fit = self._arg('handle_missing_fit',
                                            handle_missing_fit,
                                            self.handle_missing_fit_map)
        self.quasilikelihood = self._arg('quasilikelihood',
                                         quasilikelihood,
                                         bool)
        self.max_iter = self._arg('max_iter', max_iter, int)
        self.tol = self._arg('tol', tol, float)
        self.significance_level = self._arg('significance_level',
                                            significance_level,
                                            float)
        self.output_fitted = self._arg('output_fitted',
                                       output_fitted,
                                       bool)
        bad = False
        if self.solver != 'cd':
            temp_dict = dict(alpha=alpha, lamb=lamb, num_lambda=num_lambda,
                             lambda_min_ratio=lambda_min_ratio)
            for dic_itm in temp_dict:
                if temp_dict[dic_itm] is not None:
                    bad = dic_itm
                    break
        if bad is not False:
            msg = ("Parameter {} should not be provided when solver " +
                   "is not 'cd'.").format(bad)
            logger.error(msg)
            raise ValueError(msg)
        self.alpha = self._arg('alpha', alpha, float)
        self.lamb = self._arg('lamb', lamb, float)
        self.num_lambda = self._arg('num_lambda', num_lambda, int)
        self.lambda_min_ratio = self._arg('lambda_min_ratio',
                                          lambda_min_ratio,
                                          float)
        self.categorical_variable = categorical_variable
        if ordering is None:
            self.ordering = ordering
        elif not ordering:
            msg = 'ordering should be nonempty.'
            logger.error(msg)
            raise ValueError(msg)
        elif all(isinstance(val, _INTEGER_TYPES) for val in ordering):
            self.ordering = ', '.join(map(str, ordering))
        elif all(isinstance(val, _STRING_TYPES) for val in ordering):
            for value in ordering:
                if ',' in value or value.strip() != value:
                    # I don't know whether this check is enough, but it
                    # covers the cases I've found to be problematic in
                    # testing.
                    # The PAL docs don't say anything about escaping.
                    msg = ("Can't have commas or leading/trailing spaces"
                           + " in the elements of the ordering list.")
                    logger.error(msg)
                    raise ValueError(msg)
            self.ordering = ', '.join(ordering)
        else:
            msg = 'ordering should be a list of ints or a list of strings.'
            logger.error(msg)
            raise ValueError(msg)
        self.thread_ratio = self._arg('thread_ratio', thread_ratio, float)
        self.resampling_method = self._arg('resampling_method', resampling_method,
                                           {x:x for x in self.resampling_methods})
        self.evaluation_metric = self._arg('evaluation_metric', evaluation_metric,
                                           {x:x.upper() for x in self.evaluation_metrics})
        self.fold_num = self._arg('fold_num', fold_num, int,
                                  required=self.resampling_method == 'cv')
        self.repeat_times = self._arg('repeat_times', repeat_times, int)
        self.search_strategy = self._arg('search_strategy', search_strategy,
                                         {x:x for x in self.search_strategies})
        self.random_search_times = self._arg('random_search_times', random_search_times, int,
                                             required=self.search_strategy == 'random')
        self.random_state = self._arg('random_state', random_state, int)
        self.timeout = self._arg('timeout', timeout, int)
        self.progress_indicator_id = self._arg('progress_indicator_id',
                                               progress_indicator_id, str)
        if isinstance(param_range, dict):
            param_range = [(x, param_range[x]) for x in param_range]
        self.param_range = self._arg('param_range', param_range, ListOfTuples)
        if isinstance(param_values, dict):
            param_values = [(x, param_values[x]) for x in param_values]
        self.param_values = self._arg('param_values', param_values, ListOfTuples)
        search_param_count = 0
        for param in (self.resampling_method, self.evaluation_metric):
            if param is not None:
                search_param_count += 1
        if search_param_count not in (0, 2):
            msg = ("`resampling_method`, and `evaluation_metric` must be set together.")
            logger.error(msg)
            raise ValueError(msg)
        if self.search_strategy is not None and self.resampling_method is None:
            msg = ("`search_strategy` cannot be set if `resampling_method` is not specified.")
            logger.error(msg)
            raise ValueError(msg)
        if self.search_strategy is None:
            if self.param_values is not None:
                msg = ("`param_values` can only be specified "+
                       "when `search_strategy` is enabled.")
                logger.error(msg)
                raise ValueError(msg)
            if self.param_range is not None:
                msg = ("`param_range` can only be specified "+
                       "when `search_strategy` is enabled.")
                logger.error(msg)
                raise ValueError(msg)
        else:
            set_param_list = list()
            if self.lamb is not None:
                set_param_list.append("lamb")
            if self.alpha is not None:
                set_param_list.append("alpha")
            if self.link is not None:
                set_param_list.append("link")
            if self.param_values is not None:
                for x in self.param_values:#pylint:disable=invalid-name
                    if len(x) != 2:#pylint:disable=bad-option-value
                        msg = ("Each tuple that specifies the values of a parameter should"+
                               " contain exactly 2 elements: 1st is parameter name,"+
                               " 2nd is a list of valid values.")
                        logger.error(msg)
                        raise ValueError(msg)
                    if x[0] not in self.__cv_params:
                        msg = ("Specifying the values of `{}` for ".format(x[0])+
                               "parameter selection is invalid.")
                        logger.error(msg)
                        raise ValueError(msg)
                    if x[0] in set_param_list:
                        msg = ("Parameter `{}` is invalid for ".format(x[0])+
                               "re-specification in parameter selection.")
                        logger.error(msg)
                        raise ValueError(msg)
                    if x[0] in ['lamb', 'alpha'] and self.solver != 'cd':
                        msg = ("Parameter `{}` is invalid when ".format(x[0])+
                               "`solver` is not set as 'cd'.")
                        logger.error(msg)
                        raise ValueError(msg)
                    set_param_list.append(x[0])
            if self.param_range is not None:
                rsz = [3] if self.search_strategy == 'grid' else [2, 3]
                for x in self.param_range:#pylint:disable=invalid-name
                    if len(x) != 2:#pylint:disable=bad-option-value
                        msg = ("Each tuple that specifies the range of a parameter should contain"+
                               " exactly 2 elements: 1st is parameter name, 2nd is value range.")
                        logger.error(msg)
                        raise ValueError(msg)
                    if x[0] not in self.__cv_params[1:]:
                        msg = ("Specifying the values of `{}` for ".format(x[0])+
                               "parameter selection is invalid.")
                        logger.error(msg)
                        raise ValueError(msg)
                    if x[0] in set_param_list:
                        msg = ("Parameter `{}` is invalid for ".format(x[0])+
                               "re-specification in parameter selection.")
                        logger.error(msg)
                        raise ValueError(msg)
                    if not (isinstance(x[1], list) and len(x[1]) in rsz and all(isinstance(t, (int, float)) for t in x[1])):#pylint:disable=line-too-long
                        msg = ("The provided `{}` is either not ".format(x[0])+
                               "a list of numericals, or it contains wrong number of values.")
                        logger.error(msg)
                        raise TypeError(msg)
                    set_param_list.append(x[0])
            if any([x in set_param_list] for x in ['lamb', 'alpha']):
                self.solver = 'cd'

    @trace_sql
    def fit(self,# pylint: disable=too-many-arguments
            data,
            key=None,
            features=None,
            label=None,
            categorical_variable=None,
            dependent_variable=None,
            excluded_feature=None):
        r"""
        Fit a generalized linear model based on training data.

        Parameters
        ----------

        data : DataFrame
            Training data.
        key : str, optional
            Name of the ID column. If ``key`` is not provided, it is assumed
            that the input has no ID column.

            Required when ``output_fitted`` is True.
        features : list of str, optional
            Names of the feature columns.

            Defaults to all non-ID, non-label
            columns.
        label : str or list of str, optional
            Name of the dependent variable.

            Defaults to the last column(this is not the PAL default.)

            When ``family`` is 'binomial', ``label`` may be either a single
            column name or a list of two column names.
        categorical_variable : list of str, optional
            INTEGER columns specified in this list will be treated as categorical
            data. Other INTEGER columns will be treated as continuous.
        dependent_variable : str, optional(deprecated and ineffective)
            Only used when you need to indicate the dependence.

            Please use ``label`` instead.
        excluded_feature : list of str, optional(deprecated and ineffective)
            Excludes the indicated feature column.

            If necessary, please use ``features`` instead.

            Defaults to None.

        """

        # pylint:disable=too-many-locals
        # pylint:disable=too-many-statements

        # SQLTRACE
        conn = data.connection_context
        require_pal_usable(conn)
        conn.sql_tracer.set_sql_trace(self, self.__class__.__name__, 'Fit')

        key = self._arg('key', key, str)
        features = self._arg('features', features, ListOfStrings)
        # label requires more complex check.
        categorical_variable = self._arg('categorical_variable', categorical_variable, ListOfStrings)
        dependent_variable = self._arg('dependent_variable', dependent_variable, str)
        excluded_feature = self._arg('excluded_feature', excluded_feature, ListOfStrings)
        if label is not None and not isinstance(label, _STRING_TYPES):
            if self.family != 'binomial':
                msg = ("When family is not 'binomial', "
                       + "label must be a single string.")
                logger.error(msg)
                raise TypeError(msg)
            if (not isinstance(label, list)
                    or len(label) != 2
                    or not all(isinstance(elem, _STRING_TYPES) for elem in label)):
                msg = "A non-string label must be a list of two strings."
                logger.error(msg)
                raise TypeError(msg)

        if key is None and self.output_fitted:
            msg = 'A key column is required when output_fitted is True.'
            logger.error(msg)
            raise TypeError(msg)

        cols_left = data.columns
        if key is None:
            maybe_id = []
        else:
            maybe_id = [key]
            cols_left.remove(key)
        if label is None:
            label = cols_left[-1]
        if isinstance(label, _STRING_TYPES):
            label = [label]
        for column in label:
            cols_left.remove(column)
        if features is None:
            features = cols_left

        data_ = data[maybe_id + label + features]

        unique_id = str(uuid.uuid1()).replace('-', '_').upper()
        outputs = stat_tbl, coef_tbl, covmat_tbl, fit_tbl = [
            '#GLM_{}_TBL_{}_{}'.format(tab, self.id, unique_id)
            for tab in ['STAT', 'COEF', 'COVMAT', 'FIT']]
        param_rows = [
            ('SOLVER', None, None, self.solver),
            ('FAMILY', None, None, self.family),
            ('LINK', None, None, self.link),
            ('HANDLE_MISSING', self.handle_missing_fit, None, None),
            ('QUASI', self.quasilikelihood, None, None),
            ('GROUP_RESPONSE', len(label) == 2, None, None),
            ('MAX_ITERATION', self.max_iter, None, None),
            ('CONVERGENCE_CRITERION', None, self.tol, None),
            ('SIGNIFICANCE_LEVEL', None, self.significance_level, None),
            ('OUTPUT_FITTED', self.output_fitted, None, None),
            ('ENET_ALPHA', None, self.alpha, None),
            ('ENET_LAMBDA', None, self.lamb, None),
            ('ENET_NUM_LAMBDA', self.num_lambda, None, None),
            ('LAMBDA_MIN_RATIO', None, self.lambda_min_ratio, None),
            ('HAS_ID', key is not None, None, None),
            ('ORDERING', None, None, self.ordering),
            ('THREAD_RATIO', None, self.thread_ratio, None),
            ('RESAMPLING_METHOD', None, None, self.resampling_method),
            ('EVALUATION_METRIC', None, None, self.evaluation_metric),
            ('SEED', self.random_state, None, None),
            ('REPEAT_TIMES', self.repeat_times, None, None),
            ('PARAM_SEARCH_STRATEGY', None, None, self.search_strategy),
            ('FOLD_NUM', self.fold_num, None, None),
            ('RANDOM_SEARCH_TIMES', self.random_search_times, None, None),
            ('TIMEOUT', self.timeout, None, None),
            ('PROGRESS_INDICATOR_ID', None, None, self.progress_indicator_id)
        ]
        if self.categorical_variable is not None:
            param_rows.extend(('CATEGORICAL_VARIABLE', None, None, variable)
                              for variable in self.categorical_variable)
        if categorical_variable is not None:
            param_rows.extend(('CATEGORICAL_VARIABLE', None, None, variable)
                              for variable in categorical_variable)
        if self.param_values is not None:
            for x in self.param_values:#pylint:disable=invalid-name
                values = str(x[1]).replace('[', '{').replace(']', '}')
                values = values.replace("'", "\"") if x[0] == 'link' else values
                param_rows.extend([(self.__cv_param_map[x[0]]+"_VALUES",
                                    None, None, values)])
        if self.param_range is not None:
            for x in self.param_range:#pylint:disable=invalid-name
                range_ = str(x[1])
                if len(x[1]) == 2 and self.search_strategy == 'random':
                    range_ = range_.replace(',', ',,')
                param_rows.extend([(self.__cv_param_map[x[0]]+"_RANGE",
                                    None, None, range_)])
        try:
            call_pal_auto(conn,
                          'PAL_GLM',
                          data_,
                          ParameterTable().with_data(param_rows),
                          *outputs)
        except dbapi.Error as db_err:
            logger.exception(str(db_err))
            try_drop(conn, outputs)
            raise

        # pylint: disable=attribute-defined-outside-init
        self.statistics_ = conn.table(stat_tbl)
        self.coef_ = conn.table(coef_tbl)

        # For coordinate descent, this table is empty, but PAL_GLM_PREDICT
        # still wants it.
        self._covariance = conn.table(covmat_tbl)
        self.covariance_ = self._covariance if self.solver != 'cd' else None

        self.fitted_ = conn.table(fit_tbl) if self.output_fitted else None
        self.model_ = [self.statistics_, self.coef_, self._covariance]

    @trace_sql
    def predict(self, # pylint: disable=too-many-arguments
                data,
                key,
                features=None,
                prediction_type=None,
                significance_level=None,
                handle_missing=None,
                thread_ratio=0.0):
        r"""
        Predict dependent variable values based on fitted model.

        Parameters
        ----------

        data : DataFrame
            Independent variable values to predict for.
        key : str
            Name of the ID column.
        features : list of str, optional
            Names of the feature columns.

            Defaults to all non-ID columns.
        prediction_type : {'response', 'link'}, optional
            Specifies whether to output predicted values of the
            response or the link function.

            Defaults to 'response'.
        significance_level : float, optional
            Significance level for confidence intervals and prediction
            intervals. If specified, overrides the value passed to the
            GLM constructor.
        handle_missing : {'skip', 'fill_zero'}, optional
            How to handle data rows with missing independent variable values.

                - 'skip': Don't perform prediction for those rows.
                - 'fill_zero': Replace missing values with 0.

            Defaults to 'skip'.
        thread_ratio : float, optional
            Controls the proportion of available threads to use for prediction.

            The value range is from 0 to 1, where 0 indicates a single thread,
            and 1 indicates up to all available threads.

            Values between 0 and 1 will use that percentage of available threads.

            Values outside this range tell PAL to heuristically determine the number of threads to use.

        Returns
        -------

        DataFrame

            Predicted values, structured as follows. The following two
            columns are always populated:

                - ID column, with same name and type as ``data``'s ID column.
                - PREDICTION, type NVARCHAR(100), representing predicted values.

            The following five columns are only populated for IRLS:

                - SE, type DOUBLE. Standard error, or for ordinal regression, \
                  the probability that the data point belongs to the predicted \
                  category.
                - CI_LOWER, type DOUBLE. Lower bound of the confidence interval.
                - CI_UPPER, type DOUBLE. Upper bound of the confidence interval.
                - PI_LOWER, type DOUBLE. Lower bound of the prediction interval.
                - PI_UPPER, type DOUBLE. Upper bound of the prediction interval.
        """
        # pylint:disable=too-many-locals
        conn = data.connection_context
        require_pal_usable(conn)
        if not hasattr(self, 'model_'):
            raise FitIncompleteError("Model not initialized. Perform a fit first.")

        # SQLTRACE
        conn.sql_tracer.set_sql_trace(self, self.__class__.__name__, 'Predict')

        key = self._arg('key', key, str, required=True)
        features = self._arg('features', features, ListOfStrings)
        prediction_type = self._arg('prediction_type', prediction_type,
                                    {'response': 'response', 'link': 'link'})
        significance_level = self._arg('significance_level', significance_level, float)
        if significance_level is None:
            significance_level = self.significance_level
        handle_missing = self._arg('handle_missing',
                                   handle_missing,
                                   self.handle_missing_predict_map)
        thread_ratio = self._arg('thread_ratio', thread_ratio, float)
        if features is None:
            features = data.columns
            features.remove(key)
        data_ = data[[key] + features]

        unique_id = str(uuid.uuid1()).replace('-', '_').upper()

        result_tbl = '#PAL_GLM_RESULT_TBL_{}_{}'.format(self.id, unique_id)

        param_rows = [
            ('TYPE', None, None, prediction_type),
            ('SIGNIFICANCE_LEVEL', None, significance_level, None),
            ('HANDLE_MISSING', handle_missing, None, None),
            ('THREAD_RATIO', None, thread_ratio, None)
        ]

        try:
            call_pal_auto(conn,
                          'PAL_GLM_PREDICT',
                          data_,
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

    def score(self, # pylint: disable=too-many-arguments
              data,
              key,
              features=None,
              label=None,
              prediction_type=None,
              handle_missing=None):
        r"""
        Returns the coefficient of determination R2 of the prediction.

        Not applicable for ordinal regression.

        Parameters
        ----------

        data : DataFrame
            Data on which to assess model performance.
        key : str
            Name of the ID column.
        features : list of str, optional
            Names of the feature columns.

            Defaults to all non-ID, non-label
            columns.
        label : str, optional
            Name of the dependent variable.

            Defaults to the last column(this is not the PAL default).

            Cannot be two columns, even when ``family`` is 'binomial' when initializing the GLM class instance.
        prediction_type : {'response', 'link'}, optional
            Specifies whether to predict the value of the
            response or the link function.

            The contents of the ``label`` column should match this choice.

            Defaults to 'response'.
        handle_missing : {'skip', 'fill_zero'}, optional
            How to handle data rows with missing independent variable values.

              - 'skip': Don't perform prediction for those rows. Those rows
                will be left out of the R2 computation.
              - 'fill_zero': Replace missing values with 0.

            Defaults to 'skip'.

        Returns
        -------

        float

            The coefficient of determination R2  of the prediction on the
            given data.
        """
        conn = data.connection_context
        require_pal_usable(conn)
        if self.family == 'ordinal':
            msg = "Can't compute R^2 for ordinal regression."
            logger.error(msg)
            raise TypeError(msg)

        # SQLTRACE
        conn.sql_tracer.set_sql_trace(self, self.__class__.__name__, 'Score')

        key = self._arg('key', key, str, required=True)
        features = self._arg('features', features, ListOfStrings)
        label = self._arg('label', label, str)
        # leaving prediction_type and handle_missing to predict()

        cols = data.columns
        cols.remove(key)
        if label is None:
            label = cols[-1]
        cols.remove(label)
        if features is None:
            features = cols

        prediction = self.predict(data, key=key, features=features,
                                  prediction_type=prediction_type,
                                  handle_missing=handle_missing)
        prediction = prediction.select(key, 'PREDICTION')
        prediction = prediction.cast('PREDICTION', 'DOUBLE')
        prediction = prediction.rename_columns(['ID_P', 'PREDICTION'])

        actual = data.select(key, label).rename_columns(['ID_A', 'ACTUAL'])

        joined = actual.join(prediction, 'ID_P=ID_A').select(
            'ID_A', 'ACTUAL', 'PREDICTION')
        return metrics.r2_score(joined,
                                label_true='ACTUAL',
                                label_pred='PREDICTION')

class ExponentialRegression(PALBase):
    r"""
    Exponential regression is an approach to modeling the relationship between a scalar variable y and one or more variables denoted X. In exponential regression,
    data is modeled using exponential functions, and unknown model parameters are estimated from the data. Such models are called exponential models.

    Parameters
    ----------
    decomposition : {'LU', 'QR', 'SVD', 'Cholesky'}, optional
        Matrix factorization type to use. Case-insensitive.

          - 'LU': LU decomposition.
          - 'QR': QR decomposition.
          - 'SVD': singular value decomposition.
          - 'Cholesky': Cholesky(LDLT) decomposition.

        Defaults to QR decomposition.
    adjusted_r2 : bool, optional
        If true, include the adjusted R2 value in the statistics table.

        Defaults to False.
    pmml_export : {'no', 'single-row', 'multi-row'}, optional
        Controls whether to output a PMML representation of the model,
        and how to format the PMML. Case-insensitive.

          - 'no' or not provided: No PMML model.
          - 'single-row': Exports a PMML model in a maximum of
            one row. Fails if the model doesn't fit in one row.
          - 'multi-row': Exports a PMML model, splitting it
            across multiple rows if it doesn't fit in one.

        Prediction does not require a PMML model.
    thread_ratio : float, optional
        Controls the proportion of available threads to use for fitting.

        The value range is from 0 to 1, where 0 indicates a single thread,
        and 1 indicates up to all available threads.

        Values between 0 and 1 will use that percentage of available threads.

        Values outside this range tell PAL to heuristically determine the number of threads to use.

        Defaults to 0.

    Attributes
    ----------

    coefficients_ : DataFrame
        Fitted regression coefficients.
    pmml_ : DataFrame
        PMML model. Set to None if no PMML model was requested.
    fitted_ : DataFrame
        Predicted dependent variable values for training data.
        Set to None if the training data has no row IDs.
    statistics_ : DataFrame
        Regression-related statistics, such as mean squared error.

    Examples
    --------

    >>> df.collect()
       ID    Y       X1      X2
       0    0.5     0.13    0.33
       1    0.15    0.14    0.34
       2    0.25    0.15    0.36
       3    0.35    0.16    0.35
       4    0.45    0.17    0.37

    Training the model:

    >>> er = ExponentialRegression(pmml_export = 'multi-row')
    >>> er.fit(data=df, key='ID')

    Prediction:

    >>> df2.collect()
       ID    X1       X2
       0    0.5      0.3
       1    4        0.4
       2    0        1.6
       3    0.3      0.45
       4    0.4      1.7

    >>> er.predict(data=df2, key='ID').collect()
       ID      VALUE
       0      0.6900598931338715
       1      1.2341502316656843
       2      0.006630664136180741
       3      0.3887970208571841
       4      0.0052106543571450266
    """

    # pylint: disable=too-many-arguments,too-many-instance-attributes
    decomposition_map = {'lu': 0, 'qr': 1, 'svd': 2, 'cholesky': 5}
    pmml_export_map = {'no': 0, 'single-row': 1, 'multi-row': 2}
    def __init__(self,
                 decomposition=None,
                 adjusted_r2=None,
                 pmml_export=None,
                 thread_ratio=0.0):
        super(ExponentialRegression, self).__init__()
        setattr(self, 'hanaml_parameters', pal_param_register())
        self.decomposition = self._arg('decomposition', decomposition, self.decomposition_map)
        self.adjusted_r2 = self._arg('adjusted_r2', adjusted_r2, bool)
        self.pmml_export = self._arg('pmml_export', pmml_export, self.pmml_export_map)
        self.thread_ratio = self._arg('thread_ratio', thread_ratio, float)

    @trace_sql
    def fit(self, data, key=None, features=None, label=None):
        r"""
        Fit regression model based on training data.

        Parameters
        ----------
        data : DataFrame
            Training data.
        key : str, optional
            Name of the ID column.

            If ``key`` is not provided, it is assumed that the input has no ID column.
        features : list of str, optional
            Names of the feature columns.
        label : str, optional
            Name of the dependent variable.

            Defaults to the last column(this is not the PAL default).
        """
        # pylint: disable=too-many-locals

        # SQLTRACE
        conn = data.connection_context
        require_pal_usable(conn)
        conn.sql_tracer.set_sql_trace(self, self.__class__.__name__, 'Fit')

        key = self._arg('key', key, str)
        features = self._arg('features', features, ListOfStrings)
        label = self._arg('label', label, str)
        cols = data.columns
        if key is not None:
            maybe_id = [key]
            cols.remove(key)
        else:
            maybe_id = []
        if label is None:
            label = cols[-1]
        cols.remove(label)
        if features is None:
            features = cols
        str1 = ','.join(str(e) for e in features)
        data_ = data[maybe_id + [label] + features]
        unique_id = str(uuid.uuid1()).replace('-', '_').upper()
        tables = ['PARAM', 'COEF', 'PMML', 'FITTED', 'STATS']
        tables = ['#PAL_EXPONENTIAL_REGRESSION_{}_TBL_{}_{}'.format(name, self.id, unique_id)
                  for name in tables]
        param_tbl, coef_tbl, pmml_tbl, fitted_tbl, stats_tbl = tables

        param_rows = [
            ('ALG', self.decomposition, None, None),
            ('ADJUSTED_R2', self.adjusted_r2, None, None),
            ('PMML_EXPORT', self.pmml_export, None, None),
            ('SELECTED_FEATURES', None, None, str1),
            ('DEPENDENT_VARIABLE', None, None, label),
            ('HAS_ID', key is not None, None, None),
            ('THREAD_RATIO', None, self.thread_ratio, None)
        ]
        try:
            call_pal_auto(conn,
                          'PAL_EXPONENTIAL_REGRESSION',
                          data_,
                          ParameterTable(param_tbl).with_data(param_rows),
                          coef_tbl,
                          fitted_tbl,
                          stats_tbl,
                          pmml_tbl)
        except dbapi.Error as db_err:
            #msg = 'HANA error while attempting to fit exponential regression model.'
            logger.exception(str(db_err))
            try_drop(conn, tables)
            raise

        # pylint: disable=attribute-defined-outside-init
        self.coefficients_ = conn.table(coef_tbl)
        self.pmml_ = conn.table(pmml_tbl) if self.pmml_export else None
        self.fitted_ = conn.table(fitted_tbl) if key is not None else None
        self.statistics_ = conn.table(stats_tbl)
        self.model_ = self.coefficients_

    @trace_sql
    def predict(self, data, key, features=None, model_format=None, thread_ratio=0.0):
        r"""
        Predict dependent variable values based on fitted model.

        Parameters
        ----------

        data : DataFrame
            Independent variable values used for prediction.
        key : str
            Name of the ID column.
        features : list of str, optional
            Names of the feature columns.
        model_format : int, optional

            - 0: coefficient
            - 1: pmml
        thread_ratio : float, optional
            Controls the proportion of available threads to use for prediction.

            The value range is from 0 to 1, where 0 indicates a single thread,
            and 1 indicates up to all available threads.

            Values between 0 and 1 will use that percentage of available threads.

            Values outside this range tell PAL to heuristically determine the number of threads to use.

        Returns
        -------

        DataFrame

            Predicted values, structured as follows:

                - ID column, with same name and type as ``data`` 's ID column.
                - VALUE, type DOUBLE, representing predicted values.

        .. note::

            predict() will pass the ``pmml_`` table to PAL as the model
            representation if there is a ``pmml_`` table, or the ``coefficients_``
            table otherwise.
        """
        # pylint: disable=too-many-locals

        # SQLTRACE
        conn = data.connection_context
        require_pal_usable(conn)
        conn.sql_tracer.set_sql_trace(self, self.__class__.__name__, 'Predict')

        key = self._arg('key', key, str, required=True)
        features = self._arg('features', features, ListOfStrings)
        thread_ratio = self._arg('thread_ratio', thread_ratio, float)
        if getattr(self, 'pmml_', None) is not None:
            model = self.pmml_
            model_format = 1
        elif hasattr(self, 'model_'):
            model = self.model_
        else:
            raise FitIncompleteError("Model not initialized. Perform a fit first.")

        if features is None:
            cols = data.columns
            cols.remove(key)
            features = cols

        data_ = data[[key] + features]

        unique_id = str(uuid.uuid1()).replace('-', '_').upper()

        tables = param_tbl, fitted_tbl = [
            '#PAL_EXPONENTIAL_REGRESSION_{}_TBL_{}_{}'.format(name, self.id, unique_id)
            for name in ['PARAM', 'FITTED']]

        param_rows = [
            ('THREAD_RATIO', None, thread_ratio, None),
            ('MODEL_FORMAT', model_format, None, None),
        ]

        try:
            call_pal_auto(conn,
                          'PAL_EXPONENTIAL_REGRESSION_PREDICT',
                          data_,
                          model,
                          ParameterTable(param_tbl).with_data(param_rows),
                          fitted_tbl)
        except dbapi.Error as db_err:
            logger.exception(str(db_err))
            try_drop(conn, tables)
            raise

        return conn.table(fitted_tbl)

    def score(self, data, key, features=None, label=None):
        r"""
        Returns the coefficient of determination R2 of the prediction.

        Parameters
        ----------

        data : DataFrame

            Data on which to assess model performance.

        key : str

            Name of the ID column.

        features : list of str, optional

            Names of the feature columns.

        label : str, optional

            Name of the dependent variable.

            Defaults to the last column(this is not the PAL default).

        Returns
        -------
        float
            The coefficient of determination R2 of the prediction on the
            given data.
        """

        # SQLTRACE
        conn = data.connection_context
        require_pal_usable(conn)
        conn.sql_tracer.set_sql_trace(self, self.__class__.__name__, 'Score')

        key = self._arg('key', key, str, required=True)
        features = self._arg('features', features, ListOfStrings)
        label = self._arg('label', label, str)

        cols = data.columns
        cols.remove(key)
        if label is None:
            label = cols[-1]
        cols.remove(label)
        if features is None:
            features = cols

        prediction = self.predict(data, key=key, features=features)
        prediction = prediction.rename_columns(['ID_P', 'PREDICTION'])

        actual = data.select(key, label).rename_columns(['ID_A', 'ACTUAL'])

        joined = actual.join(prediction, 'ID_P=ID_A').select(
            'ACTUAL', 'PREDICTION')
        return metrics.r2_score(joined,
                                label_true='ACTUAL',
                                label_pred='PREDICTION')

class BiVariateGeometricRegression(PALBase):
    r"""
    Geometric regression is an approach used to model the relationship between a scalar variable y and a variable denoted X. In geometric regression,
    data is modeled using geometric functions, and unknown model parameters are estimated from the data. Such models are called geometric models.

    Parameters
    ----------
    decomposition : {'LU', 'QR', 'SVD', 'Cholesky'}, optional
        Matrix factorization type to use. Case-insensitive.

          - 'LU': LU decomposition.
          - 'QR': QR decomposition.
          - 'SVD': singular value decomposition.
          - 'Cholesky': Cholesky(LDLT) decomposition.

        Defaults to QR decomposition.
    adjusted_r2 : bool, optional
        If true, include the adjusted R2 value in the statistics table.

        Defaults to False.
    pmml_export : {'no', 'single-row', 'multi-row'}, optional
        Controls whether to output a PMML representation of the model,
        and how to format the PMML. Case-insensitive.

          - 'no' or not provided: No PMML model.

          - 'single-row': Exports a PMML model in a maximum of
            one row. Fails if the model doesn't fit in one row.

          - 'multi-row': Exports a PMML model, splitting it
            across multiple rows if it doesn't fit in one.

        Prediction does not require a PMML model.
    thread_ratio : float, optional
        Controls the proportion of available threads to use for fitting.

        The value range is from 0 to 1, where 0 indicates a single thread,
        and 1 indicates up to all available threads.

        Values between 0 and 1 will use that percentage of available threads.

        Values outside this range tell PAL to heuristically determine the number of threads to use.

        Defaults to 0.

    Attributes
    ----------

    coefficients_ : DataFrame
        Fitted regression coefficients.
    pmml_ : DataFrame
        PMML model. Set to None if no PMML model was requested.
    fitted_ : DataFrame
        Predicted dependent variable values for training data.
        Set to None if the training data has no row IDs.
    statistics_ : DataFrame
        Regression-related statistics, such as mean squared error.

    Examples
    --------

    >>> df.collect()
    ID    Y       X1
    0    1.1      1
    1    4.2      2
    2    8.9      3
    3    16.3     4
    4    24       5


    Training the model:

    >>> gr = BiVariateGeometricRegression(pmml_export='multi-row')
    >>> gr.fit(data=df, key='ID')

    Prediction:

    >>> df2.collect()
    ID    X1
    0     1
    1     2
    2     3
    3     4
    4     5

    >>> er.predict(data=df2, key='ID').collect()
    ID      VALUE
    0        1
    1       3.9723699817481437
    2       8.901666037549536
    3       15.779723271893747
    4       24.60086108408644
    """

    # pylint: disable=too-many-arguments,too-many-instance-attributes

    decomposition_map = {'lu': 0, 'qr': 1, 'svd': 2, 'cholesky': 5}
    pmml_export_map = {'no': 0, 'single-row': 1, 'multi-row': 2}
    def __init__(self,
                 decomposition=None,
                 adjusted_r2=None,
                 pmml_export=None,
                 thread_ratio=0.0):
        super(BiVariateGeometricRegression, self).__init__()
        setattr(self, 'hanaml_parameters', pal_param_register())
        self.decomposition = self._arg('decomposition', decomposition, self.decomposition_map)
        self.adjusted_r2 = self._arg('adjusted_r2', adjusted_r2, bool)
        self.pmml_export = self._arg('pmml_export', pmml_export, self.pmml_export_map)
        self.thread_ratio = self._arg('thread_ratio', thread_ratio, float)

    @trace_sql
    def fit(self, data, key=None, features=None, label=None):
        r"""
        Fit regression model based on training data.

        Parameters
        ----------

        data : DataFrame
            Training data.
        key : str, optional
            Name of the ID column.

            If ``key`` is not provided, it is assumed that the input has no ID column.
        features : list of str, optional
            Names of the feature columns.
        label : str, optional
            Name of the dependent variable.

            Defaults to the last column(this is not the PAL default).
        """
        # pylint: disable=too-many-locals

        # SQLTRACE
        conn = data.connection_context
        require_pal_usable(conn)
        conn.sql_tracer.set_sql_trace(self, self.__class__.__name__, 'Fit')

        key = self._arg('key', key, str)
        features = self._arg('features', features, ListOfStrings)
        label = self._arg('label', label, str)
        cols = data.columns
        if key is not None:
            maybe_id = [key]
            cols.remove(key)
        else:
            maybe_id = []
        if label is None:
            label = cols[-1]
        cols.remove(label)
        if features is None:
            features = cols
        str1 = ','.join(str(e) for e in features)
        data_ = data[maybe_id + [label] + features]
        unique_id = str(uuid.uuid1()).replace('-', '_').upper()
        tables = ['PARAM', 'COEF', 'PMML', 'FITTED', 'STATS']
        tables = ['#PAL_GEOMETRIC_REGRESSION_{}_TBL_{}_{}'.format(name, self.id, unique_id)
                  for name in tables]
        param_tbl, coef_tbl, pmml_tbl, fitted_tbl, stats_tbl = tables

        param_rows = [
            ('ALG', self.decomposition, None, None),
            ('ADJUSTED_R2', self.adjusted_r2, None, None),
            ('PMML_EXPORT', self.pmml_export, None, None),
            ('SELECTED_FEATURES', None, None, str1),
            ('DEPENDENT_VARIABLE', None, None, label),
            ('HAS_ID', key is not None, None, None),
            ('THREAD_RATIO', None, self.thread_ratio, None)
        ]
        try:
            call_pal_auto(conn,
                          'PAL_GEOMETRIC_REGRESSION',
                          data_,
                          ParameterTable(param_tbl).with_data(param_rows),
                          coef_tbl,
                          fitted_tbl,
                          stats_tbl,
                          pmml_tbl)
        except dbapi.Error as db_err:
            logger.exception(str(db_err))
            try_drop(conn, tables)
            raise

        # pylint: disable=attribute-defined-outside-init
        self.coefficients_ = conn.table(coef_tbl)
        self.pmml_ = conn.table(pmml_tbl) if self.pmml_export else None
        self.fitted_ = conn.table(fitted_tbl) if key is not None else None
        self.statistics_ = conn.table(stats_tbl)
        self.model_ = self.coefficients_

    @trace_sql
    def predict(self, data, key, features=None, model_format=None, thread_ratio=0.0):
        r"""
        Predict dependent variable values based on fitted model.

        Parameters
        ----------

        data : DataFrame
            Independent variable values used for prediction.
        key : str
            Name of the ID column.
        features : list of str, optional
            Names of the feature columns.
        model_format : int, optional

            - 0: coefficient
            - 1: pmml
        thread_ratio : float, optional
            Controls the proportion of available threads to use for prediction.

            The value range is from 0 to 1, where 0 indicates a single thread,
            and 1 indicates up to all available threads.

            Values between 0 and 1 will use that percentage of available threads.

            Values outside this range tell PAL to heuristically determine the number of threads to use.

            Defaults to 0.

        Returns
        -------

        DataFrame

            Predicted values, structured as follows:

                - ID column, with same name and type as ``data`` 's ID column.
                - VALUE, type DOUBLE, representing predicted values.

        .. note::

            predict() will pass the ``pmml_`` table to PAL as the model
            representation if there is a ``pmml_`` table, or the ``coefficients_``
            table otherwise.
        """
        # pylint: disable=too-many-locals

        # SQLTRACE
        conn = data.connection_context
        require_pal_usable(conn)
        conn.sql_tracer.set_sql_trace(self, self.__class__.__name__, 'Predict')

        key = self._arg('key', key, str, required=True)
        features = self._arg('features', features, ListOfStrings)
        thread_ratio = self._arg('thread_ratio', thread_ratio, float)
        if getattr(self, 'pmml_', None) is not None:
            model = self.pmml_
            model_format = 1
        elif hasattr(self, 'model_'):
            model = self.model_
        else:
            raise FitIncompleteError("Model not initialized. Perform a fit first.")

        if features is None:
            cols = data.columns
            cols.remove(key)
            features = cols

        data_ = data[[key] + features]

        unique_id = str(uuid.uuid1()).replace('-', '_').upper()

        tables = param_tbl, fitted_tbl = [
            '#PAL_GEOMETRIC_REGRESSION_{}_TBL_{}_{}'.format(name, self.id, unique_id)
            for name in ['PARAM', 'FITTED']]

        param_rows = [
            ('THREAD_RATIO', None, thread_ratio, None),
            ('MODEL_FORMAT', model_format, None, None),
        ]

        try:
            call_pal_auto(conn,
                          'PAL_GEOMETRIC_REGRESSION_PREDICT',
                          data_,
                          model,
                          ParameterTable(param_tbl).with_data(param_rows),
                          fitted_tbl)
        except dbapi.Error as db_err:
            logger.exception(str(db_err))
            try_drop(conn, tables)
            raise

        return conn.table(fitted_tbl)

    def score(self, data, key, features=None, label=None):
        r"""
        Returns the coefficient of determination R2 of the prediction.

        Parameters
        ----------

        data : DataFrame

            Data on which to assess model performance.

        key : str

            Name of the ID column.

        features : list of str, optional

            Names of the feature columns.

        label : str, optional

            Name of the dependent variable.

            Defaults to the last column(this is not the PAL default).

        Returns
        -------

        float

            The coefficient of determination R2 of the prediction on the
            given data.
        """
        conn = data.connection_context
        require_pal_usable(conn)
        # SQLTRACE
        conn.sql_tracer.set_sql_trace(self, self.__class__.__name__, 'Score')

        key = self._arg('key', key, str, required=True)
        features = self._arg('features', features, ListOfStrings)
        label = self._arg('label', label, str)

        cols = data.columns
        cols.remove(key)
        if label is None:
            label = cols[-1]
        cols.remove(label)
        if features is None:
            features = cols

        prediction = self.predict(data, key=key, features=features)
        prediction = prediction.rename_columns(['ID_P', 'PREDICTION'])

        actual = data.select(key, label).rename_columns(['ID_A', 'ACTUAL'])

        joined = actual.join(prediction, 'ID_P=ID_A').select(
            'ACTUAL', 'PREDICTION')
        return metrics.r2_score(joined,
                                label_true='ACTUAL',
                                label_pred='PREDICTION')

class BiVariateNaturalLogarithmicRegression(PALBase):
    r"""
    Bi-variate natural logarithmic regression is an approach to modeling the relationship between a scalar variable y and one variable denoted X. In natural logarithmic regression,
    data is modeled using natural logarithmic functions, and unknown model parameters are estimated from the data.
    Such models are called natural logarithmic models.

    Parameters
    ----------
    decomposition : {'LU', 'QR', 'SVD', 'Cholesky'}, optional
        Matrix factorization type to use. Case-insensitive.

          - 'LU': LU decomposition.
          - 'QR': QR decomposition.
          - 'SVD': singular value decomposition.
          - 'Cholesky': Cholesky(LDLT) decomposition.

        Defaults to QR decomposition.
    adjusted_r2 : bool, optional
        If true, include the adjusted R2 value in the statistics table.

        Defaults to False.
    pmml_export : {'no', 'single-row', 'multi-row'}, optional
        Controls whether to output a PMML representation of the model,
        and how to format the PMML. Case-insensitive.

          - 'no' or not provided: No PMML model.

          - 'single-row': Exports a PMML model in a maximum of
            one row. Fails if the model doesn't fit in one row.

          - 'multi-row': Exports a PMML model, splitting it
            across multiple rows if it doesn't fit in one.

        Prediction does not require a PMML model.
    thread_ratio : float, optional
        Controls the proportion of available threads to use for fitting.

        The value range is from 0 to 1, where 0 indicates a single thread,
        and 1 indicates up to all available threads.

        Values between 0 and 1 will use that percentage of available threads.

        Values outside this range tell PAL to heuristically determine the number of threads to use.

        Does not affect fitting.

        Defaults to 0.

    Attributes
    ----------

    coefficients_ : DataFrame

        Fitted regression coefficients.

    pmml_ : DataFrame

        PMML model. Set to None if no PMML model was requested.

    fitted_ : DataFrame

        Predicted dependent variable values for training data.
        Set to None if the training data has no row IDs.

    statistics_ : DataFrame

        Regression-related statistics, such as mean squared error.

    Examples
    --------

    >>> df.collect()
       ID    Y       X1
       0    10       1
       1    80       2
       2    130      3
       3    180      5
       4    190      6


    Training the model:

    >>> gr = BiVariateNaturalLogarithmicRegression(pmml_export='multi-row')
    >>> gr.fit(data=df, key='ID')

    Prediction:

    >>> df2.collect()
       ID    X1
       0     1
       1     2
       2     3
       3     4
       4     5

    >>> er.predict(data=df2, key='ID').collect()
       ID      VALUE
       0     14.86160299
       1     82.9935329364932
       2     122.8481570569525
       3     151.1254628829864
       4     173.05904529166017
    """

    # pylint: disable=too-many-arguments,too-many-instance-attributes

    decomposition_map = {'lu': 0, 'qr' : 1, 'svd': 2, 'cholesky': 5}
    pmml_export_map = {'no': 0, 'single-row': 1, 'multi-row': 2}
    def __init__(self,
                 decomposition=None,
                 adjusted_r2=None,
                 pmml_export=None,
                 thread_ratio=0.0):
        super(BiVariateNaturalLogarithmicRegression, self).__init__()
        setattr(self, 'hanaml_parameters', pal_param_register())
        self.decomposition = self._arg('decomposition', decomposition, self.decomposition_map)
        self.adjusted_r2 = self._arg('adjusted_r2', adjusted_r2, bool)
        self.pmml_export = self._arg('pmml_export', pmml_export, self.pmml_export_map)
        self.thread_ratio = self._arg('thread_ratio', thread_ratio, float)

    @trace_sql
    def fit(self, data, key=None, features=None, label=None):
        r"""
        Fit regression model based on training data.

        Parameters
        ----------

        data : DataFrame
            Training data.
        key : str, optional
            Name of the ID column.

            If ``key`` is not provided, it is assumed that the input has no ID column.
        features : list of str, optional
            Names of the feature columns.
        label : str, optional
            Name of the dependent variable.

            Defaults to the last column(this is not the PAL default).

        """
        # pylint: disable=too-many-locals

        # SQLTRACE
        conn = data.connection_context
        require_pal_usable(conn)
        conn.sql_tracer.set_sql_trace(self, self.__class__.__name__, 'Fit')

        key = self._arg('key', key, str)
        features = self._arg('features', features, ListOfStrings)
        label = self._arg('label', label, str)
        cols = data.columns
        if key is not None:
            maybe_id = [key]
            cols.remove(key)
        else:
            maybe_id = []
        if label is None:
            label = cols[-1]
        cols.remove(label)
        if features is None:
            features = cols
        str1 = ','.join(str(e) for e in features)
        data_ = data[maybe_id + [label] + features]
        unique_id = str(uuid.uuid1()).replace('-', '_').upper()
        tables = ['PARAM', 'COEF', 'PMML', 'FITTED', 'STATS']
        tables = ['#PAL_LOGARITHMIC_REGRESSION_{}_TBL_{}_{}'.format(name, self.id, unique_id)
                  for name in tables]
        param_tbl, coef_tbl, pmml_tbl, fitted_tbl, stats_tbl = tables

        param_rows = [
            ('ALG', self.decomposition, None, None),
            ('ADJUSTED_R2', self.adjusted_r2, None, None),
            ('PMML_EXPORT', self.pmml_export, None, None),
            ('SELECTED_FEATURES', None, None, str1),
            ('DEPENDENT_VARIABLE', None, None, label),
            ('HAS_ID', key is not None, None, None),
            ('THREAD_RATIO', None, self.thread_ratio, None)
        ]
        try:
            call_pal_auto(conn,
                          'PAL_LOGARITHMIC_REGRESSION',
                          data_,
                          ParameterTable(param_tbl).with_data(param_rows),
                          coef_tbl,
                          fitted_tbl,
                          stats_tbl,
                          pmml_tbl)
        except dbapi.Error as db_err:
            logger.exception(str(db_err))
            try_drop(conn, tables)
            raise

        # pylint: disable=attribute-defined-outside-init
        self.coefficients_ = conn.table(coef_tbl)
        self.pmml_ = conn.table(pmml_tbl) if self.pmml_export else None
        self.fitted_ = conn.table(fitted_tbl) if key is not None else None
        self.statistics_ = conn.table(stats_tbl)
        self.model_ = self.coefficients_

    @trace_sql
    def predict(self, data, key, features=None, model_format=None, thread_ratio=0.0):
        r"""
        Predict dependent variable values based on fitted model.

        Parameters
        ----------

        data : DataFrame
            Independent variable values used for prediction.
        key : str
            Name of the ID column.
        features : list of str, optional
            Names of the feature columns.
        model_format : int, optional

            - 0: coefficient
            - 1: pmml
        thread_ratio : float, optional
            Controls the proportion of available threads to use for prediction.

            The value range is from 0 to 1, where 0 indicates a single thread,
            and 1 indicates up to all available threads.

            Values between 0 and 1 will use that percentage of available threads.

            Values outside this range tell PAL to heuristically determine the number of threads to use.

            Does not affect fitting.

            Defaults to 0.

        Returns
        -------
        DataFrame

            Predicted values, structured as follows:

                - ID column, with same name and type as ``data`` 's ID column.
                - VALUE, type DOUBLE, representing predicted values.

        .. note::

            predict() will pass the ``pmml_`` table to PAL as the model
            representation if there is a ``pmml_`` table, or the ``coefficients_``
            table otherwise.
        """
        # pylint: disable=too-many-locals

        # SQLTRACE
        conn = data.connection_context
        require_pal_usable(conn)
        conn.sql_tracer.set_sql_trace(self, self.__class__.__name__, 'Predict')

        key = self._arg('key', key, str, required=True)
        features = self._arg('features', features, ListOfStrings)
        thread_ratio = self._arg('thread_ratio', thread_ratio, float)
        if getattr(self, 'pmml_', None) is not None:
            model = self.pmml_
            model_format = 1
        elif hasattr(self, 'model_'):
            model = self.model_
        else:
            raise FitIncompleteError("Model not initialized. Perform a fit first.")

        if features is None:
            cols = data.columns
            cols.remove(key)
            features = cols

        data_ = data[[key] + features]

        unique_id = str(uuid.uuid1()).replace('-', '_').upper()

        tables = param_tbl, fitted_tbl = [
            '#PAL_LOGARITHMIC_REGRESSION_{}_TBL_{}_{}'.format(name, self.id, unique_id)
            for name in ['PARAM', 'FITTED']]

        param_rows = [
            ('THREAD_RATIO', None, thread_ratio, None),
            ('MODEL_FORMAT', model_format, None, None),
        ]

        try:
            call_pal_auto(conn,
                          'PAL_LOGARITHMIC_REGRESSION_PREDICT',
                          data_,
                          model,
                          ParameterTable(param_tbl).with_data(param_rows),
                          fitted_tbl)
        except dbapi.Error as db_err:
            logger.exception(str(db_err))
            try_drop(conn, tables)
            raise

        return conn.table(fitted_tbl)

    def score(self, data, key, features=None, label=None):
        r"""
        Returns the coefficient of determination R2 of the prediction.

        Parameters
        ----------

        data : DataFrame
            Data on which to assess model performance.
        key : str
            Name of the ID column.
        features : list of str, optional
            Names of the feature columns.
        label : str, optional
            Name of the dependent variable.

            Defaults to the last column(this is not the PAL default).

        Returns
        -------

        float

            The coefficient of determination R2 of the prediction on the
            given data.
        """
        conn = data.connection_context
        require_pal_usable(conn)
        # SQLTRACE
        conn.sql_tracer.set_sql_trace(self, self.__class__.__name__, 'Score')

        key = self._arg('key', key, str, required=True)
        features = self._arg('features', features, ListOfStrings)
        label = self._arg('label', label, str)

        cols = data.columns
        cols.remove(key)
        if label is None:
            label = cols[-1]
        cols.remove(label)
        if features is None:
            features = cols

        prediction = self.predict(data, key=key, features=features)
        prediction = prediction.rename_columns(['ID_P', 'PREDICTION'])

        actual = data.select(key, label).rename_columns(['ID_A', 'ACTUAL'])

        joined = actual.join(prediction, 'ID_P=ID_A').select(
            'ACTUAL', 'PREDICTION')
        return metrics.r2_score(joined,
                                label_true='ACTUAL',
                                label_pred='PREDICTION')

class CoxProportionalHazardModel(PALBase):
    r"""
    Cox proportional hazard model (CoxPHM) is a special generalized linear model.
    It is a well-known realization-of-survival model that demonstrates failure or death at a certain time.

    Parameters
    ----------

    tie_method : {'breslow', 'efron'}, optional
        The method to deal with tied events.

        Defaults to 'efron'.
    status_col : bool, optional
        If a status column is defined for right-censored data:

          - False : No status column. All response times are failure/death.
          - True : The 3rd column of the data input table is a status column,
                   of which 0 indicates right-censored data and 1 indicates
                   failure/death.

        Defaults to True.
    max_iter : int, optional
        Maximum number of iterations for numeric optimization.
    convergence_criterion : float, optional
        Convergence criterion of coefficients for numeric optimization.

        Defaults to 0.
    significance_level : float, optional
        Significance level for the confidence interval of estimated coefficients.

        Defaults to 0.05.
    calculate_hazard : bool, optional
        Controls whether to calculate hazard function as well as survival function.

        - False : Does not calculate hazard function.
        - True: Calculates hazard function.

        Defaults to True.
    output_fitted : bool, optional
        Controls whether to output the fitted response:

        - False : Does not output the fitted response.
        - True: Outputs the fitted response.

        Defaults to False.
    type_kind : str, optional
        The prediction type:

          - 'risk': Predicts in risk space
          - 'lp': Predicts in linear predictor space

        Default Value is 'risk'
    thread_ratio : float, optional
        Controls the proportion of available threads to use for fitting.

        The value range is from 0 to 1, where 0 indicates a single thread,
        and 1 indicates up to all available threads.

        Values between 0 and 1 will use that percentage of available threads.

        Values outside that range tell PAL to heuristically determine the number of threads to use.

        Does not affect fitting.

        Defaults to 0.

    Attributes
    ----------

    statistics_ : DataFrame
        Regression-related statistics, such as r-square, log-likelihood, aic.

    coefficient\_ : DataFrame
        Fitted regression coefficients.

    covariance_variance : DataFrame
        Co-Variance related data.

    hazard\_ : DataFrame
        Statistics related to Time, Hazard, Survival.

    fitted\_ : DataFrame
        Predicted dependent variable values for training data.
        Set to None if the training data has no row IDs.

    Examples
    ----------

    >>> df1.collect()
        ID  TIME    STATUS  X1  X2
        1     4          1   0   0
        2     3          1   2   0
        3     1          1   1   0
        4     1          0   1   0
        5     2          1   1   1
        6     2          1   0   1
        7     3          0   0   1


    Training the model:

    >>> cox = CoxProportionalHazardModel(
    significance_level= 0.05, calculate_hazard='yes', type_kind='risk')
    >>> cox.fit(data=df1, key='ID', features=['STATUS', 'X1', 'X2'], label='TIME')

    Prediction:

    >>> df2.collect()
        ID  X1  X2
        1   0   0
        2   2   0
        3   1   0
        4   1   0
        5   1   1
        6   0   1
        7   0   1

    >>> cox.predict(data=full_tbl, key='ID',features=['STATUS', 'X1', 'X2']).collect()
        ID   PREDICTION    SE         CI_LOWER     CI_UPPER
        1   0.383590423 0.412526262 0.046607574 3.157032199
        2   1.829758442 1.385833778 0.414672719 8.073875617
        3   0.837781484 0.400894077 0.32795551  2.140161678
        4   0.837781484 0.400894077 0.32795551  2.140161678

    """

    # pylint: disable=too-many-arguments,too-many-instance-attributes
    map = {'no':0, 'yes': 1}

    def __init__(self,
                 tie_method=None,
                 status_col=None,
                 max_iter=None,
                 convergence_criterion=None,
                 significance_level=None,
                 calculate_hazard=None,
                 output_fitted=None,
                 type_kind=None,
                 thread_ratio=0.0):
        super(CoxProportionalHazardModel, self).__init__()
        setattr(self, 'hanaml_parameters', pal_param_register())
        self.tie_method = self._arg('tie_method', tie_method, str)
        self.status_col = self._arg('status_col', status_col, bool)
        self.max_iter = self._arg('max_iter', max_iter, int)
        self.convergence_criterion = self._arg('convergence_criterion', convergence_criterion, float)
        self.significance_level = self._arg('significance_level', significance_level, float)
        self.calculate_hazard = self._arg('calculate_hazard', calculate_hazard, bool)
        self.output_fitted = self._arg('output_fitted', output_fitted, bool)
        self.type_kind = self._arg('type_kind', type_kind, str)
        self.thread_ratio = self._arg('thread_ratio', thread_ratio, float)

    @trace_sql
    def fit(self, data, key=None, features=None, label=None): # pylint: disable=too-many-locals
        r"""
        Fit regression model based on training data.

        Parameters
        ----------

        data : DataFrame
            Training data.
        key : str, optional
            Name of the ID column. If ``key`` is not provided, it is assumed
            that the input has no ID column.
        features : list of str, optional
            Names of the feature columns.
        label : str, optional
            Name of the dependent variable.

            Defaults to the last column.
            (This is not the PAL default.)

        """
        conn = data.connection_context
        require_pal_usable(conn)
        # SQLTRACE
        conn.sql_tracer.set_sql_trace(self, self.__class__.__name__, 'Fit')

        # pylint: disable=too-many-locals
        key = self._arg('key', key, str)
        features = self._arg('features', features, ListOfStrings)
        label = self._arg('label', label, str)
        cols = data.columns
        if key is not None:
            maybe_id = [key]
            cols.remove(key)
        else:
            maybe_id = []
        if label is None:
            label = cols[-1]
        cols.remove(label)
        if features is None:
            features = cols
        data_ = data[maybe_id + [label] + features]
        unique_id = str(uuid.uuid1()).replace('-', '_').upper()
        tables = ['PARAM', 'COEF', 'CO_VARIANCE', 'FITTED', 'STATS', 'HAZARD']
        tables = ['#PAL_COXPH_{}_TBL_{}_{}'.format(name, self.id, unique_id)
                  for name in tables]
        param_tbl, coef_tbl, co_variance_tbl, fitted_tbl, stats_tbl, hazard_tbl = tables

        param_rows = [
            ('TIE_METHOD', None, None, self.tie_method),
            ('STATUS_COL', self.status_col, None, None),
            ('MAX_ITERATION', self.max_iter, None, None),
            ('CONVERGENCE_CRITERION', self.convergence_criterion, None, None),
            ('SIGNIFICANCE_LEVEL', None, self.significance_level, None),
            ('CALCULATE_HAZARD', self.calculate_hazard, None, None),
            ('OUTPUT_FITTED', self.output_fitted, None, None),
            ('THREAD_RATIO', None, self.thread_ratio, None)
        ]
        try:
            call_pal_auto(conn,
                          'PAL_COXPH',
                          data_,
                          ParameterTable(param_tbl).with_data(param_rows),
                          stats_tbl,
                          coef_tbl,
                          co_variance_tbl,
                          hazard_tbl,
                          fitted_tbl)
        except dbapi.Error as db_err:
            logger.exception(str(db_err))
            try_drop(conn, tables)
            raise

        # pylint: disable=attribute-defined-outside-init
        self.coefficients_ = conn.table(coef_tbl)
        self.covariance_variance_ = conn.table(co_variance_tbl)
        self.hazard_ = conn.table(hazard_tbl) if self.calculate_hazard is not None else None
        self.fitted_ = conn.table(fitted_tbl) if key is not None else None
        self.statistics_ = conn.table(stats_tbl)
        self.model_ = [self.statistics_, self.coefficients_, self.covariance_variance_]

    @trace_sql
    def predict(self, data, key, features=None, thread_ratio=0.0):
        r"""
        Predict dependent variable values based on fitted model.

        Parameters
        ----------

        data : DataFrame
            Independent variable values used for prediction.
        key : str
            Name of the ID column.
        features : list of str, optional
            Names of the feature columns.
        thread_ratio : float, optional
            Controls the proportion of available threads to use for prediction.

            The value range is from 0 to 1, where 0 indicates a single thread,
            and 1 indicates up to all available threads.

            Values between 0 and 1 will use that percentage of available threads.

            Values outside that range tell PAL to heuristically determine the number of threads to use.

            Does not affect fitting.

            Defaults to 0.

        Returns
        -------
        DataFrame
            Predicted values, structured as follows:

              - ID column, with same name and type as ``data`` 's ID column.
              - VALUE, type DOUBLE, representing predicted values.

        """

        # pylint: disable=too-many-locals
        conn = data.connection_context
        require_pal_usable(conn)
        # SQLTRACE
        conn.sql_tracer.set_sql_trace(self, self.__class__.__name__, 'Predict')

        key = self._arg('key', key, str, required=True)
        features = self._arg('features', features, ListOfStrings)
        thread_ratio = self._arg('thread_ratio', thread_ratio, float)
        if features is None:
            cols = data.columns
            cols.remove(key)
            features = cols

        data_ = data[[key] + features]

        unique_id = str(uuid.uuid1()).replace('-', '_').upper()

        tables = param_tbl, fitted_tbl = [
            '#PAL_COXPH_{}_TBL_{}_{}'.format(name, self.id, unique_id)
            for name in ['PARAM', 'FITTED']]

        param_rows = [
            ('TYPE', None, None, self.type_kind),
            ('SIGNIFICANCE_LEVEL', None, self.significance_level, None),
            ('THREAD_RATIO', None, thread_ratio, None)
        ]

        try:
            call_pal_auto(conn,
                          'PAL_COXPH_PREDICT',
                          data_,
                          self.model_[0],
                          self.model_[1],
                          self.model_[2],
                          ParameterTable(param_tbl).with_data(param_rows),
                          fitted_tbl)
        except dbapi.Error as db_err:
            logger.exception(str(db_err))
            try_drop(conn, tables)
            raise

        return conn.table(fitted_tbl)

    def score(self, data, key, features=None, label=None):
        r"""
        Returns the coefficient of determination R2 of the prediction.

        Parameters
        ----------

        data : DataFrame
            Data on which to assess model performance.

        key : str
            Name of the ID column.

        features : list of str, optional
            Names of the feature columns.

        label : str, optional
            Name of the dependent variable.

            Defaults to the last column(this is not the PAL default).

        Returns
        -------

        float
            The coefficient of determination R2 of the prediction on the
            given data.
        """
        conn = data.connection_context
        require_pal_usable(conn)
        # SQLTRACE
        conn.sql_tracer.set_sql_trace(self, self.__class__.__name__, 'Score')

        key = self._arg('key', key, str, required=True)
        features = self._arg('features', features, ListOfStrings)
        label = self._arg('label', label, str)

        cols = data.columns
        cols.remove(key)
        if label is None:
            label = cols[-1]
        cols.remove(label)
        if features is None:
            features = cols

        prediction = self.predict(data, key=key, features=features)
        prediction = prediction.select(key, 'PREDICTION')
        prediction = prediction.cast('PREDICTION', 'DOUBLE')
        prediction = prediction.rename_columns(['ID_P', 'PREDICTION'])

        actual = data.select(key, label).rename_columns(['ID_A', 'ACTUAL'])

        joined = actual.join(prediction, 'ID_P=ID_A').select(
            'ACTUAL', 'PREDICTION')
        return metrics.r2_score(joined,
                                label_true='ACTUAL',
                                label_pred='PREDICTION')
