"""
This module contains Python wrapper for PAL seasonality test algorithm.

The following function is available:

    * :func:`seasonal_decompose`
"""

# pylint:disable=line-too-long
import logging
import uuid

from hdbcli import dbapi

from hana_ml.algorithms.pal.pal_base import (
    ParameterTable,
    arg,
    try_drop,
    require_pal_usable,
    call_pal_auto
)

logger = logging.getLogger(__name__)#pylint: disable=invalid-name
#pylint:disable=too-many-arguments, too-few-public-methods, unused-argument, too-many-locals, attribute-defined-outside-init, unused-variable
def seasonal_decompose(data,
                       key=None,
                       endog=None,
                       alpha=None,
                       thread_ratio=None,
                       model=None,
                       decompose_type=None,
                       extrapolation=None,
                       smooth_width=None,
                       auxiliary_normalitytest=None):
    """
    seasonal_decompose function is to decompose a time series into three components: trend, seasonality and random noise.

    Parameters
    ----------
    data : DataFrame
        Input data. At least two columns, one is ID column, the other is raw data.
    key : str, optional
        The ID column.

        Defaults to the first column.
    endog : str, optional
        The column of series to be decomposed.

        Defaults to the first non-ID column.

    alpha : float, optional
        The criterion for the autocorrelation coefficient.
        The value range is (0, 1). A larger value indicates stricter requirement for seasonality.

        Defaults to 0.2.
    thread_ratio : float, optional
        Controls the proportion of available threads to use.
        The ratio of available threads.

            - 0: single thread.
            - 0~1: percentage.
            - Others: heuristically determined.

        Defaults to -1.
    decompose_type : {'additive', 'multiplicative', 'auto'}, optional
        Specifies decompose type.
          - 'additive': Additive decomposition model
          - 'multiplicative': Multiplicative decomposition model
          - 'auto': Decomposition model automatically determined from input data

        Defaults to 'auto'.
    extrapolation : bool, optional
       Specifies whether to extrapolate the endpoints.

       Set to True when there is an end-point issue.

       Defaults to False.
    smooth_width : int, optional
       Specifies the width of the moving average applied to non-seasonal data.

       0 indicates linear fitting to extract trends.

       Can not be larger than half of the data length.

       Defaults to 0.
    auxiliary_normalitytest : bool, optional
       Specifies whether to use normality test to identify model types.

       Defaults to False.

    Returns
    -------

    DataFrame
        Statistics for time series, structured as follows:
            - STAT_NAME: includes type (additive or multiplicative), period (number of seasonality),
              acf (autocorrelation coefficient).
            - STAT_VALUE: value of stats above.

        Seasonal decomposition table, structured as follows:
            - ID : index/time stamp.
            - SEASONAL: seasonality component.
            - TREND: trend component.
            - RANDOM: white noise component.


    Examples
    --------

    Time series data df:

    >>> df.collect().head(3)
           TIME_STAMP   SERIES
    0      1            10.0
    1      2            7.0
    2      3            17.0

    Perform seasonal_decompose function:

    >>> stats, decompose = seasonal_decompose(data, engod='SERIES', alpha=0.2, thread_ratio=0.5)

    Outputs:

    >>> stats.collect()
         STAT_NAME     STAT_VALUE
    0    type          multiplicative
    1    period        4
    2    acf           0.501947

    >>> decompose.collect().head(3)
         ID    SEASONAL     TREND        RANDOM
    0    1     1.252660     10.125       0.788445
    1    2     0.349952     14.000       1.428769
    2    3     0.748851     16.875       1.345271

    """
    seasonal_decompose_map = {'additive': 1, 'multiplicity' : 2,
                              'multiplicative': 2,
                              'none' : None, 'auto': 0}
    conn = data.connection_context
    require_pal_usable(conn)
    alpha = arg('alpha', alpha, float)
    thread_ratio = arg('thread_ratio', thread_ratio, float)
    decompose_type = arg('decompose_type', decompose_type, seasonal_decompose_map)
    extrapolation = arg('extrapolation', extrapolation, bool)
    smooth_width = arg('smooth_width', smooth_width, int)
    auxiliary_normalitytest = arg('auxiliary_normalitytest', auxiliary_normalitytest, bool)
    key = arg('key', key, str)
    endog = arg('endog', endog, str)

    unique_id = str(uuid.uuid1()).replace('-', '_').upper()
    outputs = ['STATS', 'DECOMPOSE',]
    outputs = ['#PAL_SEASONAL_DECOMPOSE_TBL_{}_{}'.format(name, unique_id) for name in outputs]
    stats_tbl, decompose_tbl = outputs
    param_rows = [('ALPHA', None, alpha, None),
                  ('THREAD_RATIO', None, thread_ratio, None),
                  ('DECOMPOSE_TYPE', decompose_type, None, None),
                  ('EXTRAPOLATION', extrapolation, None, None),
                  ('SMOOTH_WIDTH', smooth_width, None, None),
                  ('AUXILIARY_NORMALITYTEST', auxiliary_normalitytest, None, None)]

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
    cols.remove(key)

    if endog is None:
        endog = cols[0]

    data_ = data[[key] + [endog]]

    try:
        call_pal_auto(conn,
                      'PAL_SEASONALITY_TEST',
                      data_,
                      ParameterTable().with_data(param_rows),
                      *outputs)

    except dbapi.Error as db_err:
        logger.exception(str(db_err))
        try_drop(conn, stats_tbl)
        try_drop(conn, decompose_tbl)
        raise

    return conn.table(stats_tbl), conn.table(decompose_tbl)
