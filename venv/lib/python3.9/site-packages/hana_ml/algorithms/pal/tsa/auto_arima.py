"""
This module contains Python wrapper for PAL auto ARIMA algorithm.

The following class is available:

    * :class:`AutoARIMA`
"""

# pylint:disable=line-too-long
import logging
import uuid
from hdbcli import dbapi
from hana_ml.algorithms.pal.pal_base import (
    ParameterTable,
    ListOfStrings,
    pal_param_register,
    try_drop,
    require_pal_usable,
    call_pal_auto
)
from hana_ml.algorithms.pal.sqlgen import trace_sql
from hana_ml.algorithms.pal.tsa.arima import ARIMA
logger = logging.getLogger(__name__)#pylint: disable=invalid-name

class _AutoARIMABase(ARIMA):#pylint: disable=too-many-instance-attributes, too-few-public-methods
    """
    AutoARIMA class.
    """

    method_map = {'css':0, 'mle':1, 'css-mle':2}
    forecast_method_map = {'formula_forecast':0, 'innovations_algorithm':1}
    information_criterion_map = {'aicc': 0, 'aic': 1, 'bic': 2}
    traverse_map = {'exhaustive': 0, 'stepwise': 1}

    def __init__(self,#pylint: disable=too-many-locals, too-many-arguments
                 seasonal_period=None,
                 seasonality_criterion=None,
                 d=None,
                 kpss_significance_level=None,
                 max_d=None,
                 seasonal_d=None,
                 ch_significance_level=None,
                 max_seasonal_d=None,
                 max_p=None,
                 max_q=None,
                 max_seasonal_p=None,
                 max_seasonal_q=None,
                 information_criterion=None,
                 search_strategy=None,
                 max_order=True,
                 initial_p=None,
                 initial_q=None,
                 initial_seasonal_p=None,
                 initial_seasonal_q=None,
                 guess_states=None,
                 max_search_iterations=None,
                 method=None,
                 allow_linear=None,
                 forecast_method=None,
                 output_fitted=True,
                 thread_ratio=None):

        super(_AutoARIMABase, self).__init__()
        setattr(self, 'hanaml_parameters', pal_param_register())
        #parameters check
        self.seasonal_period = self._arg('seasonal_period', seasonal_period, int)
        self.seasonality_criterion = self._arg('seasonality_criterion',
                                               seasonality_criterion, float)
        self.d = self._arg('d', d, int)#pylint: disable=invalid-name
        self.kpss_significance_level = self._arg('kpss_significance_level',
                                                 kpss_significance_level, float)
        self.max_d = self._arg('max_d', max_d, int)
        self.seasonal_d = self._arg('seasonal_d', seasonal_d, int)
        self.ch_significance_level = self._arg('ch_significance_level',
                                               ch_significance_level, float)
        self.max_seasonal_d = self._arg('max_seasonal_d', max_seasonal_d, int)
        self.max_p = self._arg('max_p', max_p, int)
        self.max_q = self._arg('max_q', max_q, int)
        self.max_seasonal_p = self._arg('max_seasonal_p', max_seasonal_p, int)
        self.max_seasonal_q = self._arg('max_seasonal_q', max_seasonal_q, int)
        self.information_criterion = self._arg('information_criterion',
                                               information_criterion, (int, str))
        if isinstance(self.information_criterion, str):
            self.information_criterion = self._arg('information_criterion',
                                                   information_criterion,
                                                   self.information_criterion_map)
        self.search_strategy = self._arg('search_strategy', search_strategy, (int, str))
        if isinstance(self.search_strategy, str):
            self.search_strategy = self._arg('search_strategy',
                                             search_strategy,
                                             self.traverse_map)
        self.max_order = self._arg('max_order', max_order, int)
        self.initial_p = self._arg('initial_p', initial_p, int)
        self.initial_q = self._arg('initial_q', initial_q, int)
        self.initial_seasonal_p = self._arg('initial_seasonal_p', initial_seasonal_p, int)
        self.initial_seasonal_q = self._arg('initial_seasonal_q', initial_seasonal_q, int)
        self.guess_states = self._arg('guess_states', guess_states, int)
        self.max_search_iterations = self._arg('max_search_iterations',
                                               max_search_iterations, int)
        self.method = self._arg('method', method, self.method_map)
        self.allow_linear = self._arg('allow_linear', allow_linear, (int, bool))
        self.forecast_method = self._arg('forecast_method',
                                         forecast_method,
                                         self.forecast_method_map)
        self.output_fitted = self._arg('output_fitted', output_fitted, bool)
        self.thread_ratio = self._arg('thread_ratio', thread_ratio, float)

    @trace_sql
    def _fit(self, data, endog, non_exog):
        """
        Generates AutoARIMA models with given orders.
        """
        conn = data.connection_context
        require_pal_usable(conn)
        self.conn_context = conn
        unique_id = str(uuid.uuid1()).replace('-', '_').upper()
        outputs = ['MODEL', 'FIT']
        outputs = ['#PAL_AUTOARIMA_{}_TBL_{}_{}'.format(name, self.id, unique_id)
                   for name in outputs]
        model_tbl, fit_tbl = outputs

        param_rows = [
            ('SEASONAL_PERIOD', self.seasonal_period, None, None),
            ('SEASONALITY_CRITERION', None, self.seasonality_criterion, None),
            ('D', self.d, None, None),
            ('KPSS_SIGNIFICANCE_LEVEL', None, self.kpss_significance_level, None),
            ('MAX_D', self.max_d, None, None),
            ('SEASONAL_D', self.seasonal_d, None, None),
            ('CH_SIGNIFICANCE_LEVEL', None, self.ch_significance_level, None),
            ('MAX_SEASONAL_D', self.max_seasonal_d, None, None),
            ('MAX_P', self.max_p, None, None),
            ('MAX_Q', self.max_q, None, None),
            ('MAX_SEASONAL_P', self.max_seasonal_p, None, None),
            ('MAX_SEASONAL_Q', self.max_seasonal_q, None, None),
            ('INFORMATION_CRITERION', self.information_criterion, None, None),
            ('SEARCH_STRATEGY', self.search_strategy, None, None),
            ('MAX_ORDER', self.max_order, None, None),
            ('INITIAL_P', self.initial_p, None, None),
            ('INITIAL_Q', self.initial_q, None, None),
            ('INITIAL_SEASONAL_P', self.initial_seasonal_p, None, None),
            ('INITIAL_SEASONAL_Q', self.initial_seasonal_q, None, None),
            ('GUESS_STATES', self.guess_states, None, None),
            ('MAX_SEARCH_ITERATIONS', self.max_search_iterations, None, None),
            ('METHOD', self.method, None, None),
            ('ALLOW_LINEAR', self.allow_linear, None, None),
            ('FORECAST_METHOD', self.forecast_method, None, None),#pylint: disable=duplicate-code
            ('OUTPUT_FITTED', self.output_fitted, None, None),#pylint: disable=duplicate-code
            ('THREAD_RATIO', None, self.thread_ratio, None),#pylint: disable=duplicate-code
            ('DEPENDENT_VARIABLE', None, None, endog)#pylint: disable=duplicate-code
            ]
        #pylint: disable=duplicate-code
        if non_exog is not None:
            param_rows.extend(('EXCLUDED_FEATURE', None, None, excluded_feature)
                              for excluded_feature in non_exog)

        #pylint: disable=duplicate-code
        try:
            call_pal_auto(conn,
                          'PAL_AUTOARIMA',
                          data,
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

class AutoARIMA(_AutoARIMABase):#pylint: disable=too-many-instance-attributes,too-few-public-methods
    """
    Although the ARIMA model is useful and powerful in time series analysis, it is somehow difficult to choose appropriate orders. It is necessary, therefore, to determine the orders automatically. Hence, AutoARIMA function identifies the orders of an ARIMA model.

    Parameters
    ----------

    seasonal_period : int, optional

        Value of the seasonal period.

        - Negative: Automatically identify seasonality by means of auto-correlation scheme.
        - 0 or 1: Non-seasonal.
        - Others: Seasonal period.

        Defaults to -1.

    seasonality_criterion : float, optional

        The criterion of the auto-correlation coefficient for accepting seasonality,
        in the range of (0, 1).

        The larger it is, the less probable a time series is regarded to be seasonal.

        Valid only when ``seasonal_period`` is negative.

        Defaults to 0.2.

    D : int, optional

        Order of first-differencing.

        - Others: Uses the specified value as the first-differencing order.
        - Negative: Automatically identifies first-differencing order with KPSS test.

        Defaults to -1.

    kpss_significance_level : float, optional

        The significance level for KPSS test. Supported values are 0.01, 0.025, 0.05, and 0.1.

        The smaller it is, the larger probable a time series is considered as first-stationary,
        that is, the less probable it needs first-differencing.

        Valid only when ``D`` is negative.

        Defaults to 0.05.

    max_d : int, optional

        The maximum value of D when KPSS test is applied.

        Defaults to 2.

    seasonal_d : int, optional

        Order of seasonal-differencing.

        - Negative: Automatically identifies seasonal-differencing order Canova-Hansen test.
        - Others: Uses the specified value as the seasonal-differencing order.

        Defaults to -1.

    ch_significance_level : float, optional

        The significance level for Canova-Hansen test. Supported values are 0.01, 0.025,
        0.05, 0.1, and 0.2.

		The smaller it is, the larger probable a time series
        is considered seasonal-stationary; that is, the less probable it needs
        seasonal-differencing.

        Valid only when ``seasonal_d`` is negative.

        Defaults to 0.05.

    max_seasonal_d : int, optional

        The maximum value of ``seasonal_d`` when Canova-Hansen test is applied.

        Defaults to 1.

    max_p : int, optional

        The maximum value of AR order p.

        Defaults to 5.

    max_q : int, optional

        The maximum value of MA order q.

        Defaults to 5.

    max_seasonal_p : int, optional

        The maximum value of SAR order P.

        Defaults to 2.

    max_seasonal_q : int, optional

        The maximum value of SMA order Q.

        Defaults to 2.

    information_criterion : {'aicc', 'aic', 'bic'}, optional

        The information criterion for order selection.

        - 'aicc': Akaike information criterion with correction(for small sample sizes)
        - 'aic': Akaike information criterion
        - 'bic': Bayesian information criterion

        Defaults to 'aicc'.

    search_strategy : {'exhaustive', 'stepwise'}, optional

        Specifies the search strategy for optimal ARMA model.

            - 'exhaustive': exhaustive traverse.
            - 'stepwise': stepwise traverse.

        Defaults to 'stepwise'.

    max_order : int, optional

        The maximum value of (``max_p`` + ``max_q`` + ``max_seasonal_p`` + ``max_seasonal_q``). \
        Valid only when ``search_strategy`` is 'exhuastive'.

        Defaults to 15.

    initial_p : int, optional

        Order p of user-defined initial model.

        Valid only when ``search_strategy`` is 'stepwise'.

        Defaults to 0.

    initial_q : int, optional

        Order q of user-defined initial model.

        Valid only when ``search_strategy`` is 'stepwise'.

        Defaults to 0.

    initial_seasonal_p : int, optional

        Order seasonal_p of user-defined initial model.

        Valid only when ``search_strategy`` is 'stepwise'.

        Defaults to 0.

    initial_seasonal_q : int, optional

        Order seasonal_q of user-defined initial model.

        Valid only when ``search_strategy`` is 'stepwise'.

        Defaults to 0.

    guess_states : int, optional

        If employing ACF/PACF to guess initial ARMA models, besides user-defined model:

            - 0: No guess. Besides user-defined model, uses states (2, 2) (1, 1)m, (1, 0) (1, 0)m,
            and (0, 1) (0, 1)m meanwhile as starting states.

            - 1: Guesses starting states taking advantage of ACF/PACF.

        Valid only when ``search_strategy`` is 'stepwise'.

        Defaults to 1.

    max_search_iterations : int, optional

        The maximum iterations for searching optimal ARMA states.

        Valid only when ``search_strategy`` is 'stepwise'.

        Defaults to (``max_p`` + 1) * (``max_q`` + 1) * (``max_seasonal_p`` + 1) * (``max_seasonal_q`` + 1).

    method : {'css', 'mle', 'css-mle'}, optional
        The object function for numeric optimization

        - 'css': use the conditional sum of squares.
        - 'mle': use the maximized likelihood estimation.
        - 'css-mle': use css to approximate starting values first and then mle to fit.

        Defaults to 'css-mle'.

    allow_linear : bool, optional

        Controls whether to check linear model ARMA(0,0)(0,0)m.

        Defaults to True.

    forecast_method : {'formula_forecast', 'innovations_algorithm'}, optional
        Store information for the subsequent forecast method.

        - 'formula_forecast': compute future series via formula.
        - 'innovations_algorithm': apply innovations algorithm to compute future
          series, which requires more original information to be stored.

        Defaults to 'innovations_algorithm'.

    output_fitted : bool, optional

        Output fitted result and residuals if True.

        Defaults to True.

    thread_ratio : float, optional
        Controls the proportion of available threads to use.

        The ratio of available threads.

            - 0: single thread.
            - 0~1: percentage.
            - Others: heuristically determined.

        Defaults to -1.

    Attributes
    ----------

    model_ : DataFrame

        Model content.

    fitted_: DateFrame

        Fitted values and residuals.

    Examples
    --------

    Input DataFrame df for AutoARIMA:

    >>> df.collect().head(5)
     TIMESTAMP  Y
     1         -24.525
     2          34.720
     3          57.325
     4          10.340
     5         -12.890

    Create AutoARIMA instance:

    >>> autoarima = AutoARIMA(search_strategy='stepwise', allow_linear=True, thread_ratio=1.0)

    Perform fit on the given data df:

    >>> autoarima.fit(data=df)

    Expected output:

    >>> autoarima.model_.collect().head(5)
         KEY    VALUE
    0    p      1
    1    AR     0.255777
    2    d      0
    3    q      1
    4    MA

    >>> autoarima.fitted_.collect().set_index('TIMESTAMP').head(6)
         TIMESTAMP   FITTED      RESIDUALS
    1    1           NaN         NaN
    2    2           NaN         NaN
    3    3           NaN         NaN
    4    4           NaN         NaN
    5    5           24.525000   11.635000
    6    6           37.583931   1.461069

    """

    def fit(self, data, key=None, endog=None, exog=None):#pylint: disable=too-many-arguments,too-many-branches, too-many-statements
        """
        Generates ARIMA models with given parameters.

        Parameters
        ----------
        data : DataFrame

            Input data which at least have two columns: key and endog.

            We also support ARIMAX which needs external data (exogenous variables).

        key : str, optional

            The timestamp column of data. The type of key column is int.

            Defaults to the first column of data if not provided.

        endog : str, optional

            The endogenous variable, i.e. time series. The type of endog column is int or float.

            Defaults to the first non-key column of data if not provided.

        exog : list of str, optional

            An optional array of exogenous variables. The type of exog column is int or float.

            Valid only for ARIMAX; exog cannot be the key column or endog column.

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

        endog = self._arg('endog', endog, str)
        if endog is not None:
            if endog not in cols:
                msg = ('Please select endog from name of columns!')
                logger.error(msg)
                raise ValueError(msg)
        else:
            endog = cols[0]
        cols.remove(endog)

        exog = self._arg('exog', exog, ListOfStrings)
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
            data_ = data[[key] + [endog] + exog]
        if exog is not None and non_exog is not None:
            data_ = data[[key] + [endog] + exog + non_exog]
        if exog is None and non_exog is not None:
            data_ = data[[key] + [endog] + non_exog]
        if exog is None and non_exog is None:
            data_ = data[[key] + [endog]]

        #print("key is ", key)
        #print("endog is ", endog)
        #print("exog is ", exog)
        #print("non_exog is ", non_exog)

        super(AutoARIMA, self)._fit(data_, endog, non_exog)
