"""
This module contains Python wrapper for PAL online time series algorithms.

The following class are available:

    * :class:`OnlineARIMA`
"""

#pylint: disable=too-many-lines, line-too-long
import logging
import uuid
from hdbcli import dbapi
from hana_ml.ml_exceptions import FitIncompleteError

from hana_ml.algorithms.pal.pal_base import (
    PALBase,
    ParameterTable,
    TupleOfIntegers,
    pal_param_register,
    try_drop,
    require_pal_usable,
    call_pal_auto
)

logger = logging.getLogger(__name__)#pylint: disable=invalid-name

class OnlineARIMA(PALBase):
    r"""
    Online Autoregressive Integrated Moving Average ARIMA(p, d, q) model.

    Note that this function is a new function in SAP HANA SPS05 and Cloud.

    Parameters
    ----------

    order : (p, q, d), tuple of int, optional
        - p: value of the auto regression order.
        - d: value of the differentiation order.
        - q: value of the moving average order.

        Defaults to (0, 0, 0).

    learning_rate : float
        Learning rate.

    epsilon : float
        Convergence criterion.

    output_fitted : bool, optional
        Output fitted result and residuals if True.
        Defaults to True.

    Attributes
    ----------

    model_ : DataFrame

        Model content.

    fitted_ : DateFrame

        Fitted values and residuals.

    """
    def __init__(self,#pylint: disable=too-many-arguments
                 order=None,
                 learning_rate=None,
                 epsilon=None,
                 output_fitted=True):

        super(OnlineARIMA, self).__init__()
        setattr(self, 'hanaml_parameters', pal_param_register())
        #P, D, Q in PAL has been combined to be one parameter `order`
        self.order = self._arg('order', order, TupleOfIntegers)
        if self.order is not None and len(self.order) != 3:
            msg = ('order must contain exactly 3 integers for regression order, ' +
                   'differentiation order and moving average order perspectively.')
            logger.error(msg)
            raise ValueError(msg)

        self.learning_rate = self._arg('learning_rate', learning_rate, float)
        self.epsilon = self._arg('epsilon', epsilon, float)
        self.output_fitted = self._arg('output_fitted', output_fitted, bool)
        self.model_ = None

    def partial_fit(self, data):
        """
        Generates ARIMA models with given orders.

        Parameters
        ----------

        data : DataFrame

            Input data. The structure is as follows.

            - The first column: index (ID), int.
            - The second column: raw data, int or float.
        """
        conn = data.connection_context
        require_pal_usable(conn)
        self.conn_context = conn
        unique_id = str(uuid.uuid1()).replace('-', '_').upper()
        input_model = "#PAL_ONLINE_ARIMA_MODEL_IN_TBL_{}_{}".format(self.id, unique_id)
        try_drop(conn, input_model)
        try:
            with conn.connection.cursor() as cur:
                cur.execute('CREATE LOCAL TEMPORARY COLUMN TABLE {} ("KEY" NVARCHAR(100), "VALUE" NVARCHAR(5000))'.format(input_model))
                if self.model_ is not None:
                    cur.execute('INSERT INTO {} SELECT * FROM ({})'.format(input_model, self.model_.select_statement))
            conn.connection.commit()
        except dbapi.Error as db_err:
            logger.exception(str(db_err))
        model = conn.table(input_model)
        outputs = ['MODEL', 'FIT']
        outputs = ['#PAL_ONLINE_ARIMA_{}_TBL_{}_{}'.format(name, self.id, unique_id)
                   for name in outputs]
        model_tbl, fit_tbl = outputs

        param_rows = [
            ('P', None if self.order is None else self.order[0], None, None),
            ('D', None if self.order is None else self.order[1], None, None),
            ('M', None if self.order is None else self.order[2], None, None),
            ('OUTPUT_FITTED', self.output_fitted, None, None),
            ('LEARNING_RATE', None, self.learning_rate, None),
            ('EPSILON', None, self.epsilon, None)
            ]

        try:
            call_pal_auto(conn,
                          'PAL_ONLINE_ARIMA',
                          data,
                          ParameterTable().with_data(param_rows),
                          model,
                          *outputs)
            try_drop(conn, input_model)
        except dbapi.Error as db_err:
            logger.exception(str(db_err))
            try_drop(conn, input_model)
            try_drop(conn, outputs)
            raise
        #pylint: disable=attribute-defined-outside-init
        self.model_ = conn.table(model_tbl)
        self.fitted_ = (conn.table(fit_tbl) if self.output_fitted
                        else None)

    def predict(self, forecast_length=1):
        """
        Makes time series forecast based on the estimated online ARIMA model.

        Parameters
        ----------
        forecast_length : int, optional
            Forecast horizon, i.e. number of future points to forecast.

        Returns
        -------
        DataFrame
            Prediction result, i.e. forecasted values within specified horizon, structured as follows:
                - 1st column : timestamp
                - 2nd column : forecast value
        """
        conn = self.conn_context
        if getattr(self, 'model_', None) is None:
            raise FitIncompleteError("Model not initialized. Perform a fit first.")

        unique_id = str(uuid.uuid1()).replace('-', '_').upper()
        result_tbl = "#PAL_ONLINE_ARIMA_FORECAST_RESULT_TBL_{}_{}".format(self.id, unique_id)

        param_rows = [
            ("FORECAST_LENGTH", forecast_length, None, None)]

        try:
            call_pal_auto(conn,
                          'PAL_ONLINE_ARIMA_FORECAST',
                          self.model_,
                          ParameterTable().with_data(param_rows),
                          result_tbl)
        except dbapi.Error as db_err:
            logger.exception(str(db_err))
            try_drop(conn, result_tbl)
            raise

        return conn.table(result_tbl)
