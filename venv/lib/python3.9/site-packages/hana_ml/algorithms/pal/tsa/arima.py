"""
This module contains Python wrapper for PAL ARIMA algorithm.

The following class are available:

    * :class:`ARIMA`
"""

#pylint: disable=too-many-lines, line-too-long
import logging
import uuid

from hdbcli import dbapi
#from hana_ml.ml_exceptions import FitIncompleteError
from hana_ml.algorithms.pal.sqlgen import trace_sql
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

logger = logging.getLogger(__name__)#pylint: disable=invalid-name

class _ARIMABase(PALBase):#pylint: disable=too-many-instance-attributes, too-few-public-methods
    """
    Autoregressive Integrated Moving Average ARIMA base class.
    """

    method_map = {'css':0, 'mle':1, 'css-mle':2}
    forecast_method_map = {'formula_forecast':0, 'innovations_algorithm':1}
    forecast_method_predict_map = {'formula_forecast':0, 'innovations_algorithm':1, 'truncation_algorithm':2}

    def __init__(self,#pylint: disable=too-many-arguments
                 order=None,
                 seasonal_order=None,
                 method='css-mle',
                 include_mean=None,
                 forecast_method=None,
                 output_fitted=True,
                 thread_ratio=None):

        super(_ARIMABase, self).__init__()
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
        self.method = self._arg('method', method, self.method_map)
        self.forecast_method = self._arg('forecast_method',
                                         forecast_method,
                                         self.forecast_method_map)
        self.output_fitted = self._arg('output_fitted', output_fitted, bool)
        self.include_mean = self._arg('include_mean', include_mean, bool)
        if (self.order is not None and
                self.seasonal_order is not None and
                self.order[1] + self.seasonal_order[1] > 1 and
                self.include_mean is not None):
            msg = ('include_mean is only valid when the sum of differentiation order ' +
                   'seasonal_period is not larger than 1.')
            logger.error(msg)
            raise ValueError(msg)
        self.thread_ratio = self._arg('thread_ratio', thread_ratio, float)

    @trace_sql
    def _fit(self, data, endog, non_exog):
        """
        Generates ARIMA models with given orders.
        """
        conn = data.connection_context
        require_pal_usable(conn)
        self.conn_context = conn
        unique_id = str(uuid.uuid1()).replace('-', '_').upper()
        outputs = ['MODEL', 'FIT']
        outputs = ['#PAL_ARIMA_{}_TBL_{}_{}'.format(name, self.id, unique_id)
                   for name in outputs]
        model_tbl, fit_tbl = outputs

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
            ('METHOD', self.method, None, None),
            ('INCLUDE_MEAN', self.include_mean, None, None),
            ('FORECAST_METHOD', self.forecast_method, None, None),
            ('OUTPUT_FITTED', self.output_fitted, None, None),
            ('THREAD_RATIO', None, self.thread_ratio, None),
            ('DEPENDENT_VARIABLE', None, None, endog)
            ]

        if non_exog is not None:
            param_rows.extend(('EXCLUDED_FEATURE', None, None, excluded_feature)
                              for excluded_feature in non_exog)

        try:
            call_pal_auto(conn,
                          'PAL_ARIMA',
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
    def _predict(self, data,
                 forecast_method,
                 forecast_length):
        """
        Makes time series forecast based on the estimated ARIMA model.
        """

        conn = self.conn_context

        param_rows = [
            ("FORECAST_METHOD", forecast_method, None, None),
            ("FORECAST_LENGTH", forecast_length, None, None)]

        unique_id = str(uuid.uuid1()).replace('-', '_').upper()
        result_tbl = "#PAL_ARIMA_FORECAST_RESULT_TBL_{}_{}".format(self.id, unique_id)
        data_tbl = None

        try:
            if data is None:
                data_tbl = "#PAL_ARIMA_FORECAST_DATA_TBL_{}_{}".format(self.id, unique_id)
                with conn.connection.cursor() as cur:
                    cur.execute('CREATE LOCAL TEMPORARY COLUMN TABLE {} ("TIMESTAMP" INTEGER,"SERIES" DOUBLE)'.format(data_tbl))
                    data = conn.table(data_tbl)
                if not conn.connection.getautocommit():
                    conn.connection.commit()
            call_pal_auto(conn,
                          'PAL_ARIMA_FORECAST',
                          data,
                          self.model_,
                          ParameterTable().with_data(param_rows),
                          result_tbl)
        except dbapi.Error as db_err:
            logger.exception(str(db_err))
            if data_tbl is not None:
                try_drop(conn, data_tbl)
            try_drop(conn, result_tbl)
            raise

        return conn.table(result_tbl)

class ARIMA(_ARIMABase):#pylint: disable=too-many-instance-attributes
    r"""
    Autoregressive Integrated Moving Average ARIMA(p, d, q) model.

    .. note::
        PAL ARIMA algorithm contains four functions: ARIMA, SARIMA, ARIMAX and SARIMA depending on \n
        whether seasonal information and external (intervention) data are provided.

    Parameters
    ----------

    order : (p, d, q), tuple of int, optional
        - p: value of the auto regression order.
        - d: value of the differentiation order.
        - q: value of the moving average order.

        Defaults to (0, 0, 0).

    seasonal_order : (P, D, Q, s), tuple of int, optional
        - P: value of the auto regression order for the seasonal part.
        - D: value of the differentiation order for the seasonal part.
        - Q: value of the moving average order for the seasonal part.
        - s: value of the seasonal period.

        Defaults to (0, 0, 0, 0).

    method : {'css', 'mle', 'css-mle'}, optional
        - 'css': use the conditional sum of squares.
        - 'mle': use the maximized likelihood estimation.
        - 'css-mle': use css to approximate starting values first and then mle to fit.

        Defaults to 'css-mle'.

    include_mean : bool, optional
        ARIMA model includes a constant part if True.
        Valid only when d + D <= 1.

        Defaults to True if d + D = 0 else False.

    forecast_method : {'formula_forecast', 'innovations_algorithm'}, optional

        - 'formula_forecast': compute future series via formula.
        - 'innovations_algorithm': apply innovations algorithm to compute future series,
        which requires more original information to be stored.

        Store information for the subsequent forecast method.

        Defaults to 'innovations_algorithm'.

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


    Examples
    --------

    ARIMA example:

    Input dataframe df for ARIMA:

    >>> df.collect()
    TIMESTAMP    Y
        1    -0.636126431
        2     3.092508651
        3    -0.73733556
        4    -3.142190983
        5     2.088819813
        6     3.179302734
        7    -0.871376102
        8    -3.475633275
        9     1.779244219
        10    3.609159416
        11    0.082170143
        12   -4.42439631
        13    0.499210261
        14    4.514017351
        15   -0.320607187
        16   -3.70219307
        17    0.100228116
        18    4.553625233
        19    0.261489853
        20   -4.474116429
        21   -0.372574233
        22    2.872305281
        23    1.289850031
        24   -3.662763983
        25   -0.168962933
        26    4.018728154
        27   -1.306247869
        28   -2.182690245
        29   -0.845114493
        30    0.99806763
        31   -0.641201109
        32   -2.640777923
        33    1.493840358
        34    4.326449202
        35   -0.653797151
        36   -4.165384227

    Create an ARIMA instance:

    >>> arima = ARIMA(order=(0, 0, 1), seasonal_order=(1, 0, 0, 4), method='mle', thread_ratio=1.0)

    Perform fit on the given data:

    >>> arima.fit(data=df)

    Expected output:

    >>> arima.model_.collect().head(5)
         KEY    VALUE
    0    p      0
    1    AR
    2    d      0
    3    q      1
    4    MA    -0.141073

    >>> arima.fitted_.collect().set_index('TIMESTAMP').head(3)
         TIMESTAMP   FITTED      RESIDUALS
    1    1           0.023374   -0.659500
    2    2           0.114596    2.977913
    3    3          -0.396567   -0.340769

    Perform predict on the model:

    >>> result = arima.predict(forecast_method='innovations_algorithm', forecast_length=10)

    Expected output:

    >>> result.collect().head(3)
        TIMESTAMP    FORECAST    SE          LO80         HI80        LO95        HI95
    0   0           1.557783     1.302436   -0.111357    3.226922    -0.994945    4.110511
    1   1           3.765987     1.315333    2.080320    5.451654     1.187983    6.343992
    2   2          -0.565599     1.315333   -2.251266    1.120068    -3.143603    2.012406

    ARIMAX example:

    Input dataframe df for ARIMAX:

    >>> df.collect()
       ID    Y                   X
       1     1.2                 0.8
       2     1.34845613096197    1.2
       3     1.32261090809898    1.34845613096197
       4     1.38095306748554    1.32261090809898
       5     1.54066648969168    1.38095306748554
       6     1.50920806756785    1.54066648969168
       7     1.48461408893443    1.50920806756785
       8     1.43784887380224    1.48461408893443
       9     1.64251548718992    1.43784887380224
       10    1.74292337447476    1.64251548718992
       11    1.91137546943257    1.74292337447476
       12    2.07735796176367    1.91137546943257
       13    2.01741246166924    2.07735796176367
       14    1.87176938196573    2.01741246166924
       15    1.83354723357744    1.87176938196573
       16    1.66104978144571    1.83354723357744
       17    1.65115984070812    1.66104978144571
       18    1.69470966154593    1.65115984070812
       19    1.70459802935728    1.69470966154593
       20    1.61246059980916    1.70459802935728
       21    1.53949706614636    1.61246059980916
       22    1.59231354902055    1.53949706614636
       23    1.81741927705578    1.59231354902055
       24    1.80224252773564    1.81741927705578
       25    1.81881576781466    1.80224252773564
       26    1.78089755157948    1.81881576781466
       27    1.61473635574416    1.78089755157948
       28    1.42002147867225    1.61473635574416
       29    1.49971641345022    1.42002147867225

    Create an ARIMAX instance:

    >>> arimax = ARIMA(order=(1, 0, 1), method='mle', thread_ratio=1.0)

    Perform fit on the given data:

    >>> arimax.fit(data=df, endog='Y')

    Expected output:

    >>> arimax.model_.collect().head(5)
         KEY    VALUE
    0    p      1
    1    AR     0.302207
    2    d      0
    3    q      1
    4    MA     0.291575

    >>> arimax.fitted_.collect().set_index('TIMESTAMP').head(3)
         TIMESTAMP   FITTED      RESIDUALS
         1           1.182363    0.017637
         2           1.416213   -0.067757
         3           1.453572   -0.130961

    Perform predict on the ARIMAX model:

    Input dataframe df2 for ARIMAX predict:

    >>> df2.collect()
        TIMESTAMP    X
    0   1            0.800000
    1   2            1.200000
    2   3            1.348456
    3   4            1.322611
    4   5            1.380953
    5   6            1.540666
    6   7            1.509208
    7   8            1.484614
    8   9            1.437849
    9   10           1.642515

    >>> result = arimax.predict(df2, forecast_method='innovations_algorithm', forecast_length=5)

    Expected output:

    >>> result.collect().head(3)
        TIMESTAMP    FORECAST    SE          LO80         HI80        LO95        HI95
    0   0            1.195952    0.093510    1.076114     1.315791    1.012675    1.379229
    1   1            1.411284    0.108753    1.271912     1.550657    1.198132    1.624436
    2   2            1.491856    0.110040    1.350835     1.632878    1.276182    1.707530

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
        #print(data_.collect())
        super(ARIMA, self)._fit(data_, endog, non_exog)

    def predict(self, data=None, key=None,
                forecast_method=None, forecast_length=None):
        """
        Makes time series forecast based on the estimated ARIMA model.

        Parameters
        ----------

        data : DataFrame, optional

            Index and exogenous variables for forecast. For ARIMAX only. \

            Defaults to None.

        key : str, optional

            The timestamp column of data. The type of key column is int.

            Defaults to the first column of data if data is not None.

        forecast_method : {'formula_forecast', 'innovations_algorithm', 'truncation_algorithm'}, optional
            Specify the forecast method.

            If 'forecast_method' in fit function is 'formula_forecast', no enough information is stored, so the formula method is adopted instead.

            Truncation algorithm is much faster than innovations algorithm when the AR representation of
            ARIMA model can be truncated to finite order.

            - 'formula_forecast': forecast via formula.
            - 'innovations_algorithm': apply innovations algorithm to forecast.
            - 'truncation_algorithm'

            Defaults to 'innovations_algorithm'.

        forecast_length : int, optional

            Number of points to forecast.

            Valid only when ``data`` is None.

            In ARIMAX, the forecast length is the same as the length of the input data.

            Defaults to None.

        Returns
        -------

        DataFrame
            Forecasted values, structured as follows:

              - ID, type INTEGER, timestamp.
              - FORECAST, type DOUBLE, forecast value.
              - SE, type DOUBLE, standard error.
              - LO80, type DOUBLE, low 80% value.
              - HI80, type DOUBLE, high 80% value.
              - LO95, type DOUBLE, low 95% value.
              - HI95, type DOUBLE, high 95% value.

        """
        if getattr(self, 'model_', None) is None:
            msg = ('Model not initialized. Perform a fit first.')
            logger.error(msg)
            raise ValueError(msg)

        if (self.forecast_method == 0) and (forecast_method == 'innovations_algorithm' or forecast_method is None):
            msg = ('not enough information is stored for innovations_algorithm, ' +
                   'use formula_forecast instead.')
            logger.error(msg)
            raise ValueError(msg)

        forecast_method = self._arg('forecast_method',
                                    forecast_method,
                                    self.forecast_method_predict_map)
        forecast_length = self._arg('forecast_length', forecast_length, int)

        # validate key
        key = self._arg('key', key, str)

        if ((key is not None) and (data is not None) and (key not in data.columns)):
            msg = ('Please select key from name of columns!')
            logger.error(msg)
            raise ValueError(msg)

        data_ = data

        # prepare the data, which could be empty or combination of key(must be the first column) and external data.
        if data is None:
            unique_id = str(uuid.uuid1()).replace('-', '_').upper()
            data_name = "#PAL_ARIMA_PREDICT_DATA_{}".format(unique_id)
            with self.conn_context.connection.cursor() as cur:
                #cur.execute('DROP TABLE {}'.format(data_name))
                cur.execute('CREATE LOCAL TEMPORARY  COLUMN TABLE {} ("TIMESTAMP" INTEGER, "Y" DOUBLE)'.format(data_name))
            if not self.conn_context.connection.getautocommit():
                self.conn_context.connection.commit()
            data_ = self.conn_context.table(data_name)
        elif key is not None:
        #if data is not None and key is not None:
            exog = data.columns
            exog.remove(key)
            data_ = data[[key] + exog]

        return super(ARIMA, self)._predict(data_,
                                           forecast_method,
                                           forecast_length)
