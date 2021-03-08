"""
This module contains Python wrapper for PAL LSTM algorithm.

The following class are available:

    * :class:`LSTM`
"""

#pylint: disable=too-many-lines, line-too-long, too-many-locals
import logging
import uuid

from hdbcli import dbapi
#from hana_ml.ml_exceptions import FitIncompleteError
from hana_ml.algorithms.pal.pal_base import (
    PALBase,
    ParameterTable,
    pal_param_register,
    try_drop,
    require_pal_usable,
    call_pal_auto
)
from hana_ml.algorithms.pal.sqlgen import trace_sql

logger = logging.getLogger(__name__)#pylint: disable=invalid-name

class LSTM(PALBase):#pylint: disable=too-many-instance-attributes, too-few-public-methods
    r"""
    Long short-term memory (LSTM).

    Parameters
    ----------

    learning_rate : float, optional
        Learning rate for gradient descent

        Defaults to 0.01.
    gru : {'gru', 'lstm'}, optional
        Choose GRU or LSTM.

        Defaults to 'lstm'.
    batch_size : int, optional
        Number of pieces of data for training in one iteration.

        Defaults to 32.
    time_dim : int, optional
        It specifies how many time steps in a sequence that will be trained by LSTM/GRU and then for time series prediction.

        The value of it must be smaller than the length of input time series minus 1.

        Defaults to 16.
    hidden_dim : int, optional
        Number of hidden neuron in LSTM/GRU unit.

        Defaults to 128.
    num_layers : int, optional
        Number of layers in LSTM/GRU unit.

        Defaults to 1.
    max_iter : int, optional
        Number of batches of data by which LSTM/GRU is trained.

        Defaults to 1000.
    interval : int, optional
        Output the average loss within every INTERVAL iterations.

        Defaults to 100.
    optimizer_type : {'SGD', 'RMSprop', 'Adam', 'Adagrad'}, optional
        Choose the optimizer.

        Defaults to 'Adam'.
    stateful : bool, optional
        If the value is True, it enables stateful LSTM/GRU.

        Defaults to True.
    bidirectional : bool, optional
        If the value is True, it uses BiLSTM/BiGRU. Otherwise, it uses LSTM/GRU.

        Defaults to False.

    Attributes
    ----------
    loss_ : DateFrame

        LOSS.

    model_ : DataFrame

        Model content.

    Examples
    --------
    Input dataframe df:

    >>> df.head(3).collect()
        TIMESTAMP  SERIES
    0          0    20.7
    1          1    17.9
    2          2    18.8

    Create LSTM model:

    >>> lstm = lstm.LSTM(gru='lstm',
                         bidirectional=False,
                         time_dim=16,
                         max_iter=1000,
                         learning_rate=0.01,
                         batch_size=32,
                         hidden_dim=128,
                         num_layers=1,
                         interval=1,
                         stateful=False,
                         optimizer_type='Adam')

    Perform fit on the given data:

    >>> lstm.fit(self.df)

    Peform predict on the fittd model:

    >>> res = lstm.predict(self.df_predict)

    Expected output:

    >>> res.head(3).collect()
       ID      VALUE                                        REASON_CODE
    0   0  11.673560  [{"attr":"T=0","pct":28.926935203430372,"val":...
    1   1  14.057195  [{"attr":"T=3","pct":24.729787064691735,"val":...
    2   2  15.119411  [{"attr":"T=2","pct":41.616207151605458,"val":...

    """
    gru_map = {'lstm' : 0, 'gru' : 1}
    optimizer_map = {'sgd' : 0, 'rmsprop' : 1, 'adam' : 2, 'adagrad' : 3}

    def __init__(self,#pylint: disable=too-many-arguments
                 learning_rate=None,
                 gru=None,
                 batch_size=None,
                 time_dim=None,
                 hidden_dim=None,
                 num_layers=None,
                 max_iter=None,
                 interval=None,
                 optimizer_type=None,
                 stateful=None,
                 bidirectional=None):

        super(LSTM, self).__init__()
        setattr(self, 'hanaml_parameters', pal_param_register())
        self.learning_rate = self._arg('learning_rate', learning_rate, float)
        self.gru = self._arg('gru', gru, self.gru_map)
        self.batch_size = self._arg('batch_size', batch_size, int)
        self.time_dim = self._arg('time_dim', time_dim, int)
        self.hidden_dim = self._arg('hidden_dim', hidden_dim, int)
        self.num_layers = self._arg('num_layers', num_layers, int)
        self.max_iter = self._arg('max_iter', max_iter, int)
        self.interval = self._arg('interval', interval, int)
        self.optimizer_type = self._arg('optimizer_type', optimizer_type, self.optimizer_map)
        self.stateful = self._arg('stateful', stateful, (int, bool))
        self.bidirectional = self._arg('bidirectional', bidirectional, (int, bool))

    @trace_sql
    def fit(self, data):#pylint: disable=too-many-arguments,too-many-branches, too-many-statements
        """
        Generates LSTM models with given parameters.

        Parameters
        ----------
        data : DataFrame
            Input data. The structure is as follows.

              - The first column: index (ID), int.
              - The second column: raw data, int or float.
        """

        conn = data.connection_context
        require_pal_usable(conn)
        unique_id = str(uuid.uuid1()).replace('-', '_').upper()
        loss_tbl = '#PAL_LSTM_LOSS_TBL_{}_{}'.format(self.id, unique_id)
        model_tbl = '#PAL_LSTM_MODEL_TBL_{}_{}'.format(self.id, unique_id)
        outputs = [loss_tbl, model_tbl]
        param_rows = [
            ('LEARNING_RATE', None, self.learning_rate, None),
            ('GRU', self.gru, None, None),
            ('BATCH_SIZE', self.batch_size, None, None),
            ('TIME_DIM', self.time_dim, None, None),
            ('HIDDEN_DIM', self.hidden_dim, None, None),
            ('NUM_LAYERS', self.num_layers, None, None),
            ('MAX_ITER', self.max_iter, None, None),
            ('INTERVAL', self.interval, None, None),
            ('OPTIMIZER_TYPE', self.optimizer_type, None, None),
            ('STATEFUL', self.stateful, None, None),
            ('BIDIRECTIONAL', self.bidirectional, None, None)
            ]
        try:
            call_pal_auto(conn,
                          'PAL_LSTM_TRAIN',
                          data,
                          ParameterTable().with_data(param_rows),
                          *outputs)
        except dbapi.Error as db_err:
            msg = str(conn.hana_version())
            logger.exception("HANA version: %s. %s", msg, str(db_err))
            try_drop(conn, outputs)
            raise

        #pylint: disable=attribute-defined-outside-init
        self.loss_ = conn.table(loss_tbl)
        self.model_ = conn.table(model_tbl)

    @trace_sql
    def predict(self, data, top_k_attributions=None):
        """
        Makes time series forecast based on the LSTM model.

        Parameters
        ----------

        data : DataFrame, optional

            - First column: Index (ID), int.
            - Other columns : external data, int or float.

            Defaults to None.

        top_k_attributions : int, optional
            Specifies the number of features with highest attributions to output.

            Defaults to 10.

        Returns
        -------

        DataFrame
            The aggerated forecasted values.
            Forecasted values, structured as follows:

              - ID, type INTEGER, timestamp.
              - VALUE, type DOUBLE, forecast value.
              - REASON_CODE, type NCLOB, Sorted SHAP values for test data at each time step.
        """
        if getattr(self, 'model_', None) is None:
            msg = ('Model not initialized. Perform a fit first.')
            logger.error(msg)
            raise ValueError(msg)
        conn = data.connection_context
        param_rows = [
            ("TOP_K_ATTRIBUTIONS", top_k_attributions, None, None)]

        unique_id = str(uuid.uuid1()).replace('-', '_').upper()
        result_tbl = "#PAL_LSTM_FORECAST_RESULT_TBL_{}_{}".format(self.id, unique_id)
        try:
            call_pal_auto(conn,
                          'PAL_LSTM_PREDICT',
                          data,
                          self.model_,
                          ParameterTable().with_data(param_rows),
                          result_tbl)
        except dbapi.Error as db_err:
            logger.exception(str(db_err))
            try_drop(conn, result_tbl)
            raise
        result_df = conn.table(result_tbl)
        return result_df
