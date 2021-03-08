"""
This module contains Python wrapper for PAL Vector ARIMA algorithm.

The following class are available:

    * :class:`VECTOR_ARIMA`
"""

#pylint: disable=too-many-lines, line-too-long, too-many-locals
import logging
import uuid

from hdbcli import dbapi
#from hana_ml.ml_exceptions import FitIncompleteError
from hana_ml.algorithms.pal.pal_base import (
    PALBase,
    ParameterTable,
    ListOfStrings,
    TupleOfIntegers,
    pal_param_register,
    try_drop,
    require_pal_usable,
    call_pal_auto
)
from hana_ml.algorithms.pal.sqlgen import trace_sql

logger = logging.getLogger(__name__)#pylint: disable=invalid-name

class VectorARIMA(PALBase):#pylint: disable=too-many-instance-attributes, too-few-public-methods
    r"""
    Vector Autoregressive Integrated Moving Average ARIMA(p, d, q) model.

    Parameters
    ----------

    order : (p, d, q), tuple of int, optional
        Indicate the order (p, d, q).

        - p: value of the auto regression order. -1 indicates auto and >=0 is user-defined.
        - d: value of the differentiation order.
        - q: value of the moving average order. -1 indicates auto and >=0 is user-defined.

        Defaults to (-1, 0, -1).

    seasonal_order : (P, D, Q, s), tuple of int, optional
        Indicate the seasonal order (P, D, Q, s).

        - P: value of the auto regression order for the seasonal part. -1 indicates auto and >=0 is user-defined.
        - D: value of the differentiation order for the seasonal part.
        - Q: value of the moving average order for the seasonal part. -1 indicates auto and >=0 is user-defined.
        - s: value of the seasonal period. -1 indicates auto and >=0 is user-defined.

        Defaults to (-1, 0, -1, 0).

    model_type : {'VAR', 'VMA', 'VARMA'}, optional
        The model type.

        Defaults to 'VARMA'.

    search_method : {'eccm', 'grid_search'}, optional
        Specifies the orders of the model. 'eccm' is valid only when seasonal period is less than 1.

        Defaults to 'grid_search'.

    lag_num : int, optional
        The lag number of explanatory variables. Valid only when ``model_type`` is 'VAR'.

        Defaults to auto.

    max_p : int, optional

        The maximum value of vector AR order p.

        Defaults to 6 if ``model_type`` is 'VAR' or if ``model_type`` is 'VARMA'
		and ``search_method`` is 'eccm'.

        Defaults to 2 if ``model_type`` is 'VARMA' and ``search_method`` is 'grid_search'.

    max_q : int, optional

        The maximum value of vector MA order q.

        Defaults to 8 if ``model_type`` is 'VMA'.

        Defaults to 5 if ``model_type`` is 'VARMA' and ``search_method`` is 'eccm'.

        Defaults to 2 if ``model_type`` is 'VARMA' and ``search_method`` is 'grid_search'.

    max_seasonal_p : int, optional

        The maximum value of seasonal vector AR order P.

        Defaults to 3 if ``model_type`` is 'VAR'.

        Defaults to 1 if ``model_type`` is 'VARMA' and ``search_method`` is 'grid_search'.

    max_seasonal_q : int, optional

        The maximum value of seasonal vector MA order Q.

        Defaults to 1.

    max_lag_num : int, optional
        The maximum lag number of explanatory variables. Valid only when ``model_type`` is 'VAR'.

        Defaults to 4.

    init_guess : {'ARMA', 'VAR'}, optional
        The model used as initial estimation for VARMA. Valid only for VARMA.

        Defaults to 'VAR'.

    information_criterion : {'AIC', 'BIC'}, optional
        Information criteria for order specification.

        Defaults to 'AIC'.

    include_mean : bool, optional
        ARIMA model includes a constant part if True.

        Valid only when d + D <= 1.

        Defaults to True if d + D = 0 else False.

    max_iter : int, optional
        Maximum number of iterations of L-BFGS-B optimizer. Valid only for VMA and VARMA.

        Defaults to 200.

    finite_diff_accuracy : int, optional
        Polynomial order of finite difference.

        Approximate the gradient of objective function with finite difference.

        The valid range is from 1 to 4.

        Defaults to 1.

    displacement : float, optional
        The step length for finite-difference method.

        Valid only for VMA and VARMA.

        Defaults to 2.2e-6.

    ftol : float, optional
        Tolerance for objective convergence test.

        Valid only for VMA and VARMA.

        Defaults to 1e-5.

    gtol : float, optional
        Tolerance for gradient convergence test.

        Valid only for VMA and VARMA.

        Defaults to 1e-5.

    calculate_hessian : bool, optional
        Specifies whether to calculate the Hessian matrix.

        VMA and VARMA will output standard error of parameter estimates only when calculate_hessian is True.

        Defaults to False.

    calculate_irf : bool, optional
        Specifies whether to calculate impulse response function.

        Defaults to False.

    irf_lags : int, optional
        The number of lags of the IRF to be calculated.

        Valid only when calculate_irf is True.

        Defaults to 8.

    alpha : float, optional
        Type-I error used in the Ljung-Box tests and eccm.

        Defaults to 0.05.

    output_fitted : bool, optional
        Output fitted result and residuals if True.

        Defaults to True.

    thread_ratio : float, optional
        Controls the proportion of available threads to use.

        The ratio of available threads.
            - 0: single thread
            - 0~1: percentage
            - Others: heuristically determined

        Defaults to -1.

    Attributes
    ----------

    model_ : DataFrame

        Model content.

    fitted_ : DateFrame

        Fitted values and residuals.

    irf_ : DataFrame
        Impulse response function.

    Examples
    --------

    Vector ARIMA example:

    Input dataframe df:

    >>> df.collect()
    TIMESTAMP Y1       X        Y2
    1         9.8      6.4      8.2
    2         9.7      6.4      8.1
    3         9.8      6.3      8
    4         9.7      6.2      7.9
    5         9.6      6.3      7.8
    6         9.6      6.8      7.6
    7         9.6      6.8      7.5
    8         9        6.8      7.5
    9         9.2      6.8      7.4
    10        9.2      6.7      7.5
    11        9.1      6.6      7.6
    12        9        6.6      7.5
    13        8.8      6        7.2
    14        8.8      6        7.7
    15        8.7      5.9      7
    16        8.3      5.8      6.5
    17        8.2      5.9      6.4
    18        8.2      6.3      6.3
    19        8.2      6.3      6.1
    20        8.4      6.4      6
    21        8.1      6.4      6.1
    22        7.8      6.5      6
    23        7.7      6.5      5.9
    24        7.5      6.3      5.9
    25        7.2      6.5      5.7
    26        7.2      6.4      5.8
    27        7        6.3      5.8
    28        7        6        5.5
    29        6.9      6.2      5.4
    30        7        5.9      5.4
    31        7.1      6        5.3
    32        7.4      6        5.4
    33        6.9      5.8      5.5
    34        6.8      5.8      5.4
    35        7        5.6      5.4
    36        7.1      5.6      5.4
    37        7        5.3      5.7
    38        7        5.3      5.6
    39        7.2      5.4      5.5
    40        7.6      5.5      5.8

    Create an VectorARIMA instance:

    >>> varima = VectorARIMA(``model_type`` is 'VAR', calculate_irf=True)

    Perform fit on the given data:

    >>> varima.fit(data=df, endog=['Y1', 'Y2'], exog='X')

    Expected output:

    >>> varima.model_.head(5).collect()
    CONTENT_INDEX	CONTENT_VALUE
    0	0	{"model":"VAR"}
    1	1	{"exogCols":["X"]}
    2	2	{"endogCols":["Y1","Y2"]}
    3	3	{"D":0,"P":0,"c":1,"d":0,"k":2,"m":2,"nT":40,"...
    4	4	{"AIC":-6.6759375491341144}

    >>> varima.fitted_.head(3).collect()
        NAMECOL	IDX	FITTING	  RESIDUAL
    0	Y1      1   NaN       NaN
    1	Y1      2   NaN       NaN
    2	Y1      3   9.622092  0.177908

    >>> varima.irf_.head(3).collect()
        COL1    COL2    IDX	RESPONSE
    0	Y1	    X	    0	0.243569
    1	Y1	    X	    1	0.139749
    2	Y1	    X	    2	-0.351429

    Perform predict on the model:

    >>> pred_df.collect()
    	TIMESTAMP	X
    0	41	        5.2
    1	42	        5.2
    2	43	        5.2
    3	44	        5.2
    4	45	        5.7
    >>> result_dict, result_all = varima.predict(pred_df)

    Expected output:

    >>> result_dict['Y1'].head(3).collect()
        IDX	FORECAST	SE	        LO95	    HI95
    0	41	7.577883	0.172352	7.240072	7.915694
    1	42	7.202759	0.233421	6.745254	7.660264
    2	43	7.074507	0.279358	6.526966	7.622049
    3	44	6.856650	0.316641	6.236034	7.477265
    4	45	6.773185	0.347997	6.091110	7.455259

    >>> result_dict['Y2'].head(3).collect()
        IDX	FORECAST	SE	        LO95	    HI95
    0	41	5.822953	0.171752	5.486320	6.159586
    1	42	5.837502	0.216817	5.412541	6.262464
    2	43	5.577920	0.249243	5.089403	6.066437
    3	44	5.395543	0.275731	4.855109	5.935976
    4	45	5.141598	0.298299	4.556933	5.726263

    >>> result_all.head(6).collect()
        COLNAME	IDX	FORECAST	SE	        LO95	    HI95
    0	Y1	    41	7.577883	0.172352	7.240072	7.915694
    1	Y1	    42	7.202759	0.233421	6.745254	7.660264
    2	Y1	    43	7.074507	0.279358	6.526966	7.622049
    3	Y1	    44	6.856650	0.316641	6.236034	7.477265
    4	Y1	    45	6.773185	0.347997	6.091110	7.455259
    5	Y2      41	5.822953	0.171752	5.486320	6.159586
    6	Y2	    42	5.837502	0.216817	5.412541	6.262464
    7	Y2	    43	5.577920	0.249243	5.089403	6.066437
    8	Y2	    44	5.395543	0.275731	4.855109	5.935976
    9	Y2      45	5.141598	0.298299	4.556933	5.726263
    """
    model_type_map = {'var' : 0, 'vma' : 1, 'varma' : 2}
    search_method_map = {'eccm' : 0, 'grid_search' : 1}
    init_guess_map = {'arma' : 0, 'var' : 1}
    information_criterion_map = {'aic' : 0, 'bic' : 1}

    def __init__(self,#pylint: disable=too-many-arguments
                 order=None,
                 seasonal_order=None,
                 model_type=None,
                 search_method=None,
                 lag_num=None,
                 max_p=None,
                 max_q=None,
                 max_seasonal_p=None,
                 max_seasonal_q=None,
                 max_lag_num=None,
                 init_guess=None,
                 information_criterion=None,
                 include_mean=None,
                 max_iter=None,
                 finite_diff_accuracy=None,
                 displacement=None,
                 ftol=None,
                 gtol=None,
                 calculate_hessian=None,
                 calculate_irf=None,
                 irf_lags=None,
                 alpha=None,
                 output_fitted=None,
                 thread_ratio=None):

        super(VectorARIMA, self).__init__()
        setattr(self, 'hanaml_parameters', pal_param_register())
        #P, D, Q in PAL has been combined to be one parameter `order`
        self.order = self._arg('order', order, TupleOfIntegers)
        if self.order is not None and len(self.order) != 3:
            msg = ('order must contain exactly 3 integers for regression order, ' +
                   'differentiation order and moving average order perspectively.')
            logger.error(msg)
            raise ValueError(msg)
        #seasonal P, D, Q and seasonal period in PAL has been combined
        #to be one parameter `seasonal order`
        self.seasonal_order = self._arg('seasonal_order', seasonal_order, TupleOfIntegers)
        if self.seasonal_order is not None and len(self.seasonal_order) != 4:
            msg = ('seasonal_order must contain exactly 4 integers for regression order, ' +
                   'differentiation order, moving average order for seasonal part' +
                   'and seasonal period.')
            logger.error(msg)
            raise ValueError(msg)
        if (self.seasonal_order is not None and
                any(s_order > 0 for s_order in self.seasonal_order[:3]) and
                self.seasonal_order[3] <= 1):
            msg = ('seasonal_period must be larger than 1.')
            logger.error(msg)
            raise ValueError(msg)
        self.model_type = self._arg('model_type', model_type, self.model_type_map)
        self.search_method = self._arg('search_method', search_method, self.search_method_map)
        self.lag_num = self._arg('lag_num', lag_num, int)
        self.max_p = self._arg('max_p', max_p, int)
        self.max_q = self._arg('max_q', max_q, int)
        self.max_seasonal_p = self._arg('max_seasonal_p', max_seasonal_p, int)
        self.max_seasonal_q = self._arg('max_seasonal_q', max_seasonal_q, int)
        self.max_lag_num = self._arg('max_lag_num', max_lag_num, int)
        self.init_guess = self._arg('init_guess', init_guess, self.init_guess_map)
        self.information_criterion = self._arg('information_criterion', information_criterion, self.information_criterion_map)
        self.include_mean = self._arg('include_mean', include_mean, bool)
        if (self.order is not None and
                self.seasonal_order is not None and
                self.order[1] + self.seasonal_order[1] > 1 and
                self.include_mean is not None):
            msg = ('include_mean is only valid when the sum of differentiation order ' +
                   'seasonal_period is not larger than 1.')
            logger.error(msg)
            raise ValueError(msg)
        self.max_iter = self._arg('max_iter', max_iter, int)
        self.finite_diff_accuracy = self._arg('finite_diff_accuracy', finite_diff_accuracy, int)
        self.displacement = self._arg('displacement', displacement, float)
        self.ftol = self._arg('ftol', ftol, float)
        self.gtol = self._arg('gtol', gtol, float)
        self.calculate_hessian = self._arg('calculate_hessian', calculate_hessian, bool)
        self.calculate_irf = self._arg('calculate_irf', calculate_irf, bool)
        self.irf_lags = self._arg('irf_lags', irf_lags, int)
        self.alpha = self._arg('alpha', alpha, float)
        self.output_fitted = self._arg('output_fitted', output_fitted, bool)
        self.thread_ratio = self._arg('thread_ratio', thread_ratio, float)
        self.conn_context = None
        self.endog = None

    @trace_sql
    def fit(self, data, key=None, endog=None, exog=None):#pylint: disable=too-many-arguments,too-many-branches, too-many-statements
        """
        Generates ARIMA models with given parameters.

        Parameters
        ----------

        data : DataFrame

            DataFrame includes key, endogenous variables and may contain exogenous variables.

        key : str, optional

            The timestamp column of data. The type of key column is int.

            Defaults to the first column of data if not provided.

        endog : list of str, optional

            The endogenous variables, i.e. time series. The type of endog column is int or float.

            Defaults to all non-key and non-exog columns of data if not provided.

        exog : list of str, optional

            An optional array of exogenous variables. The type of exog column is int or float.

            Defaults to None.
        """

        if data is None:
            msg = ('The data for fit cannot be None!')
            logger.error(msg)
            raise ValueError(msg)

        # validate key, endog, exog
        cols = data.columns
        key = self._arg('key', key, str)

        if key is not None:
            if key not in cols:
                msg = ('Please select key from name of columns!')
                logger.error(msg)
                raise ValueError(msg)
        else:
            key = cols[0]
        cols.remove(key)
        if endog is not None:
            if isinstance(endog, str):
                endog = [endog]
        if exog is not None:
            if isinstance(exog, str):
                exog = [exog]
        endog = self._arg('endog', endog, ListOfStrings)
        exog = self._arg('exog', exog, ListOfStrings)
        if endog is not None:
            if set(endog).issubset(set(cols)) is False:
                msg = ('Please select endog from name of columns!')
                logger.error(msg)
                raise ValueError(msg)
        else:
            if exog is None:
                endog = cols
            else:
                endog = list(set(cols) - set(exog))
        self.endog = endog
        cols = list(set(cols) - set(endog))
        non_exog = None
        if exog is not None:
            if set(exog).issubset(set(cols)) is True:
                non_exog = list(set(cols) - set(exog))
            else:
                msg = ('Please select exog from name of columns!')
                logger.error(msg)
                raise ValueError(msg)

        data_ = data
        if exog is not None and non_exog is None:
            data_ = data[[key] + endog + exog]
        if exog is not None and non_exog is not None:
            data_ = data[[key] + endog + exog + non_exog]
        if exog is None and non_exog is not None:
            data_ = data[[key] + endog + non_exog]
        if exog is None and non_exog is None:
            data_ = data[[key] + endog]
        conn = data.connection_context
        require_pal_usable(conn)
        self.conn_context = conn
        unique_id = str(uuid.uuid1()).replace('-', '_').upper()
        outputs = ['MODEL', 'FIT', 'IRF']
        outputs = ['#PAL_VARIMA_{}_TBL_{}_{}'.format(name, self.id, unique_id)
                   for name in outputs]
        model_tbl, fit_tbl, irf_tbl = outputs

        param_rows = [
            ('P', None if self.order is None else self.order[0], None, None),
            ('D', None if self.order is None else self.order[1], None, None),
            ('Q', None if self.order is None else self.order[2], None, None),
            ('SEASONAL_P',
             None if self.seasonal_order is None else self.seasonal_order[0],
             None, None),
            ('SEASONAL_D',
             None if self.seasonal_order is None else self.seasonal_order[1],
             None, None),
            ('SEASONAL_Q',
             None if self.seasonal_order is None else self.seasonal_order[2],
             None, None),
            ('SEASONAL_PERIOD',
             None if self.seasonal_order is None else self.seasonal_order[3],
             None, None),
            ('MODEL', self.model_type, None, None),
            ('SEARCH_METHOD', self.search_method, None, None),
            ('M', self.lag_num, None, None),
            ('MAX_P', self.max_p, None, None),
            ('MAX_Q', self.max_q, None, None),
            ('MAX_SEASONAL_P', self.max_seasonal_p, None, None),
            ('MAX_SEASONAL_Q', self.max_seasonal_q, None, None),
            ('MAX_M', self.max_lag_num, None, None),
            ('INITIAL_GUESS', self.init_guess, None, None),
            ('INCLUDE_MEAN', self.include_mean, None, None),
            ('MAX_ITERATION', self.max_iter, None, None),
            ('FINITE_DIFFERENCE_ACCURACY', self.finite_diff_accuracy, None, None),
            ('DISPLACEMENT', None, self.displacement, None),
            ('FTOL', None, self.ftol, None),
            ('GTOL', None, self.gtol, None),
            ('HESSIAN', self.calculate_hessian, None, None),
            ('CALCULATE_IRF', self.calculate_irf, None, None),
            ('IRF_LAGS', self.irf_lags, None, None),
            ('ALPHA', None, self.alpha, None),
            ('OUTPUT_FITTED', self.output_fitted, None, None),
            ('THREAD_RATIO', None, self.thread_ratio, None)
            ]

        if non_exog is not None:
            param_rows.extend(('EXCLUDED_FEATURE', None, None, excluded_feature)
                              for excluded_feature in non_exog)
        if exog is not None:
            param_rows.extend(('EXOGENEOUS_VARIABLE', None, None, exog_elem)
                              for exog_elem in exog)

        try:
            call_pal_auto(conn,
                          'PAL_VARMA',
                          data_,
                          ParameterTable().with_data(param_rows),
                          *outputs)
        except dbapi.Error as db_err:
            logger.exception(str(db_err))
            try_drop(conn, outputs)
            raise

        #pylint: disable=attribute-defined-outside-init
        self.model_ = conn.table(model_tbl)
        self.fitted_ = (conn.table(fit_tbl) if self.output_fitted
                        else None)
        self.irf_ = (conn.table(irf_tbl) if self.calculate_irf else None)

    def set_conn(self, connection_context):
        """
        Set connection context for ARIMA and AutoARIMA instance.

        Parameters
        ----------
        connection_context : ConnectionContext
            The connection to the SAP HANA system.

        Returns
        -------
        None.

        """
        self.conn_context = connection_context


    @trace_sql
    def predict(self, data=None, key=None, forecast_length=None):
        """
        Makes time series forecast based on the estimated ARIMA model.

        Parameters
        ----------

        data : DataFrame, optional

            Index and exogenous variables for forecast. The structure is as follows:

              - First column: Index (ID), int.
              - Other columns : exogenous variables, int or float.

            Defaults to None.

        key : str, optional

            The timestamp column of data. The type of key column is int.

            Defaults to the first column of data if data is not None.

        forecast_length : int, optional

            Number of points to forecast. Valid only when the first input table is absent.

            Defaults to None.

        Returns
        -------

        Dict of DataFrames
            Collection of forecasted value. Key is the column name.
            Forecasted values, structured as follows:

              - ID, type INTEGER, timestamp.
              - FORECAST, type DOUBLE, forecast value.
              - SE, type DOUBLE, standard error.
              - LO95, type DOUBLE, low 95% value.
              - HI95, type DOUBLE, high 95% value.

        DataFrame
            The aggerated forecasted values.
            Forecasted values, structured as follows:

              - COLNAME, type NVARCHAR(5000), name of endogs.
              - ID, type INTEGER, timestamp.
              - FORECAST, type DOUBLE, forecast value.
              - SE, type DOUBLE, standard error.
              - LO95, type DOUBLE, low 95% value.
              - HI95, type DOUBLE, high 95% value.
        """
        if getattr(self, 'model_', None) is None:
            msg = ('Model not initialized. Perform a fit first.')
            logger.error(msg)
            raise ValueError(msg)
        forecast_length = self._arg('forecast_length', forecast_length, int)
        # validate key
        key = self._arg('key', key, str)
        if ((key is not None) and (data is not None) and (key not in data.columns)):
            msg = ('Please select key from name of columns!')
            logger.error(msg)
            raise ValueError(msg)

        data_ = data
        # prepare the data, which could be empty or combination of key(must be the first column) and external data.
        if (data is not None) and (key is not None):
            exog = data.columns
            exog.remove(key)
            data_ = data[[key] + exog]
        conn = self.conn_context
        if data is not None:
            conn = data.connection_context
        param_rows = [
            ("FORECAST_LENGTH", forecast_length, None, None)]

        unique_id = str(uuid.uuid1()).replace('-', '_').upper()
        result_tbl = "#PAL_VARIMA_FORECAST_RESULT_TBL_{}_{}".format(self.id, unique_id)
        data_tbl = None
        try:
            if data_ is None:
                data_tbl = "#PAL_VARIMA_FORECAST_DATA_TBL_{}_{}".format(self.id, unique_id)
                with conn.connection.cursor() as cur:
                    cur.execute('CREATE LOCAL TEMPORARY COLUMN TABLE {} ("TIMESTAMP" INTEGER)'.format(data_tbl))
                    data_ = conn.table(data_tbl)
                if not conn.connection.getautocommit():
                    conn.connection.commit()
            call_pal_auto(conn,
                          'PAL_VARMA_FORECAST',
                          data_,
                          self.model_,
                          ParameterTable().with_data(param_rows),
                          result_tbl)
        except dbapi.Error as db_err:
            logger.exception(str(db_err))
            if data_tbl is not None:
                try_drop(conn, data_tbl)
            try_drop(conn, result_tbl)
            raise
        result_df = conn.table(result_tbl)
        result = {}
        if self.endog is not None:
            for col in self.endog:
                result[col] = result_df.filter("COLNAME='{}'".format(col)).deselect("COLNAME")
        return result, result_df
