#pylint: disable=too-many-lines, line-too-long, invalid-name, relative-beyond-top-level, too-few-public-methods, too-many-arguments
#pylint: disable=too-many-instance-attributes, attribute-defined-outside-init
#pylint: disable=too-many-locals
"""
This module contains Python wrapper for PAL linear regression with damped trend and seasonal adjust algorithm.

The following class is available:

    * :class:`LR_seasonal_adjust`
"""

import logging
import uuid
from hdbcli import dbapi
from hana_ml.algorithms.pal.pal_base import (
    PALBase,
    ParameterTable,
    try_drop,
    require_pal_usable,
    call_pal_auto
)

logger = logging.getLogger(__name__)

class LR_seasonal_adjust(PALBase):
    """
    Linear regression with damped trend and seasonal adjust is an approach for forecasting when a time series presents a trend.

    Parameters
    ----------
    forecast_length : int, optional
        Specifies the length of forecast results.

        Defaults to 1.

    trend : float, optional
        Specifies the damped trend factor, with valid range (0, 1].

        Defaults to 1.

    affect_future_only : bool, optional
        Specifies whether the damped trend affect future only or it affects the entire history.

        Defaults to True.

    seasonality : int, optional
        Specifies whether the data represents seasonality.

          - 0: Non-seasonality.
          - 1: Seasonality exists and user inputs the value of periods.
          - 2: Automatically detects seasonality.

        Defaults to 0.

    seasonal_period : int, optional

        Length of seasonal_period.seasonal_period is only valid when seasonality is 1.

        If this parameter is not specified, the seasonality value will be changed from 1 to 2,
        that is, from user-defined to automatically-detected.

        No default value.

    seasonal_handle_method : {'average', 'lr'}, optional

        Method used for calculating the index value in the seasonal_period.

          - 'average': Average method.
          - 'lr': Fitting linear regression.

        Defaults to 'average'.

    accuracy_measure : {'mpe', 'mse', 'rmse', 'et', 'mad', 'mase', 'wmape', 'smape', 'mape'}, optional
        The criterion used for the optimization.

        No default value.

    ignore_zero : bool, optional
         - False: Uses zero values in the input dataset when calculating 'mpe' or 'mape'.
         - True: Ignores zero values in the input dataset when calculating 'mpe' or 'mape'.

        Only valid when ``accuracy_measure`` is 'mpe' or 'mape'.

        Defaults to False.

    expost_flag : bool, optional

         - False: Does not output the expost forecast, and just outputs the forecast values.
         - True: Outputs the expost forecast and the forecast values.

       Defaults to True.

    Attributes
    ----------

    forecast_ : DataFrame

        Forecast values.

    stats_ : DataFrame

        Statistics analysis content.

    Examples
    --------

    Input Dataframe df:

    >>> df.collect()
             TIMESTAMP    Y
             1            5384
             2            8081
             3            10282
             4            9156
             5            6118
             6            9139
             7            12460
             8            10717
             9            7825
             10           9693
             11           15177
             12           10990

    Create a LR_seasonal_adjust instance:

    >>> lr = LR_seasonal_adjust(forecast_length=10,
                                trend=0.9, affect_future_only=True,
                                seasonality=1, seasonal_period=4,
                                accuracy_measure='mse')

    Perform fit_predict on the given data:

    >>> lr.fit_predict(data=df)

    Output:

    >>> lr.forecast_.collect().set_index('TIMESTAMP').head(3)
        TIMESTAMP    VALUE
        1            5328.171741
        2            7701.608247
        3            11248.606332

    >>> lr.stats_.collect()
           STAT_NAME        STAT_VALUE
           Intercept        7626.072428
           Slope            301.399114
           Periods          4.000000
           Index0           0.672115
           Index1           0.935925
           Index2           1.318669
           Index3           1.073290
           MSE              332202.479082
           HandleZero       0.000000
    """
    accuracy_measure_list = ["mpe", "mse", "rmse", "et", "mad", "mase", "wmape", "smape", "mape"]
    seasonal_handle_method_map = {'average':0, 'lr':1}

    def __init__(self,
                 forecast_length=None,
                 trend=None,
                 affect_future_only=None,
                 seasonality=None,
                 seasonal_period=None,
                 accuracy_measure=None,
                 seasonal_handle_method=None,
                 expost_flag=None,
                 ignore_zero=None):
        super(LR_seasonal_adjust, self).__init__()
        self.forecast_length = self._arg('forecast_length', forecast_length, int)
        self.trend = self._arg('trend', trend, float)
        self.affect_future_only = self._arg('affect_future_only', affect_future_only, bool)
        self.seasonality = self._arg('seasonality', seasonality, int)
        self.seasonal_period = self._arg('seasonal_period', seasonal_period, int)
        self.accuracy_measure = self._arg('accuracy_measure', accuracy_measure, str)
        self.seasonal_handle_method = self._arg('seasonal_handle_method', seasonal_handle_method, self.seasonal_handle_method_map)
        self.expost_flag = self._arg('expost_flag', expost_flag, bool)
        self.ignore_zero = self._arg('ignore_zero', ignore_zero, bool)

        if self.accuracy_measure is not None:
            if self.accuracy_measure not in self.accuracy_measure_list:
                msg = ('Please select accuracy_measure from the list ["mpe", "mse", "rmse", "et", "mad", "mase", "wmape", "smape", "mape"]!')
                logger.error(msg)
                raise ValueError(msg)
            self.accuracy_measure = self.accuracy_measure.upper()

        measure_list = ["mpe", "mape"]
        if self.ignore_zero is not None and self.accuracy_measure not in measure_list:
            msg = ('Please select accuracy_measure from "mpe" and "mape" when ignore_zero is not None!')
            logger.error(msg)
            raise ValueError(msg)

        if self.seasonality != 1 and self.seasonal_period is not None:
            msg = ('seasonal_period is only valid when seasonality is 1!')
            logger.error(msg)
            raise ValueError(msg)

        if self.seasonality != 2 and self.seasonal_handle_method is not None:
            msg = ('seasonal_handle_method is only valid when seasonality is 2!')
            logger.error(msg)
            raise ValueError(msg)

    def fit_predict(self, data, endog=None, key=None):
        """
        Fit and predict based on the given time series.

        Parameters
        ----------

        data : DataFrame

            Input data. At least two columns, one is ID column, the other is raw data.

        endog : str, optional
            The column of series to be fitted and predicted.

            Defaults to the second column.

        key : str, optional
            The ID column.

            Defaults to the first column.
        """
        conn = data.connection_context
        require_pal_usable(conn)
        endog = self._arg('endog', endog, str)
        key = self._arg('key', key, str)

        cols = data.columns
        if len(cols) < 2:
            msg = ("Input data should contain at least 2 columns: " +
                   "one for ID, another for raw data.")
            logger.error(msg)
            raise ValueError(msg)

        if endog is not None and endog not in cols:
            msg = ('The endog should be selected from columns of data!')
            logger.error(msg)
            raise ValueError(msg)

        if key is not None and key not in cols:
            msg = ('The key should be selected from columns of data!')
            logger.error(msg)
            raise ValueError(msg)

        if key is None:
            key = cols[0]

        if endog is None:
            endog = cols[1]

        if key == endog:
            msg = ('The key and endog cannot be same!')
            logger.error(msg)
            raise ValueError(msg)

        data_ = data[[key] + [endog]]

        unique_id = str(uuid.uuid1()).replace('-', '_').upper()
        outputs = ['FORECAST', 'STATISTICS']
        outputs = ['#PAL_LR_SEASONAL_ADJUST_{}_TBL_{}_{}'.format(name, self.id, unique_id)
                   for name in outputs]
        forecast_tbl, stats_tbl = outputs

        param_rows = [
            ('FORECAST_LENGTH', self.forecast_length, None, None),
            ('TREND', None, self.trend, None),
            ('AFFECT_FUTURE_ONLY', self.affect_future_only, None, None),
            ('SEASONALITY', self.seasonality, None, None),
            ('PERIODS', self.seasonal_period, None, None),
            ('MEASURE_NAME', None, None, self.accuracy_measure),
            ('SEASONAL_HANDLE_METHOD', self.seasonal_handle_method, None, None),
            ('IGNORE_ZERO', self.ignore_zero, None, None),
            ('EXPOST_FLAG', self.expost_flag, None, None)
        ]

        try:
            call_pal_auto(conn,
                          "PAL_LR_SEASONAL_ADJUST",
                          data_,
                          ParameterTable().with_data(param_rows),
                          forecast_tbl,
                          stats_tbl)
        except dbapi.Error as db_err:
            logger.exception(str(db_err))
            try_drop(conn, forecast_tbl)
            try_drop(conn, stats_tbl)
            raise

        self.stats_ = conn.table(stats_tbl)
        return conn.table(forecast_tbl)
