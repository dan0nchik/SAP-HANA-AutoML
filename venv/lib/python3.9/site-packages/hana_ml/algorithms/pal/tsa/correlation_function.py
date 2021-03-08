"""
This module contains Python wrapper for PAL correlation function algorithm.

The following function is available:

    * :func:`correlation`
"""

#pylint: disable=too-many-lines, line-too-long, relative-beyond-top-level
import logging
import uuid

from hdbcli import dbapi
from ..pal_base import (
    ParameterTable,
    arg,
    try_drop,
    require_pal_usable,
    call_pal_auto
)
#pylint: disable=invalid-name
logger = logging.getLogger(__name__)

def correlation(data, key, x=None, y=None,#pylint:disable=too-many-arguments, too-many-locals
                thread_ratio=None,
                method=None,
                max_lag=None,
                calulate_pacf=None):
    """
    This correlation function gives the statistical correlation between random variables.

    Parameters
    ----------

    data : DataFrame
        Input data.
    key : str
        Name of the ID column.
    x : str, optional
        Name of the first series data column.
    y : str, optional
        Name of the second series data column.
    thread_ratio : float, optional

        The ratio of available threads.

           - 0: single thread
           - 0~1: percentage
           - Others: heuristically determined

        Valid only when ``method`` is set as 'brute_force'.

        Defaults to -1.
    method : {'auto', 'brute_force', 'fft'}, optional
        Indicates the method to be used to calculate the correlation function.

        Defaults to 'auto'.
    max_lag : int, optional
        Maximum lag for the correlation function.

        Defaults to sqrt(n), where n is the data number.
    calculate_pacf : bool, optional
        Controls whether to calculate PACF or not.

        Valid only when only one series is provided.

        Defaults to True.

    Returns
    -------
    DataFrame
        Result of the correlation function, structured as follows:
            - LAG: ID column.
            - CV: ACV/CCV.
            - CF: ACF/CCF.
            - PACF: PACF. Null if cross-correlation is calculated.

    Examples
    --------

    Data for correlation:

    >>> df_cor.collect().head(10)
         ID      X
    0     1   88.0
    1     2   84.0
    2     3   85.0
    3     4   85.0
    4     5   84.0
    5     6   85.0
    6     7   83.0
    7     8   85.0
    8     9   88.0
    9    10   89.0

    Perform correlation function on the input dataframe:

    >>> res = correlation(data=df_cor,
                          key='ID',x='X', thread_ratio=0.4, method='auto',
                          calulate_pacf=True)
    >>> res.collect()
        LAG           CV        CF      PACF
    0     0  1583.953600  1.000000  1.000000
    1     1  1520.880736  0.960180  0.960180
    2     2  1427.356272  0.901135 -0.266618
    3     3  1312.695808  0.828746 -0.154417
    4     4  1181.606944  0.745986 -0.120176
    5     5  1041.042480  0.657243 -0.071546
    6     6   894.493216  0.564722 -0.065065
    7     7   742.178352  0.468561 -0.083686
    8     8   587.453488  0.370878 -0.065213
    9     9   434.287824  0.274180 -0.045501
    10   10   286.464160  0.180854 -0.029586
    """
    conn = data.connection_context
    require_pal_usable(conn)
    key = arg('key', key, str, required=True)
    cols = data.columns
    cols.remove(key)
    x, y = arg('x', x, str), arg('y', y, str)
    if x is None:
        msg = ("The first series must be given.")
        logger.error(msg)
        raise ValueError(msg)
    cols = [x]
    if x is not None and y is not None:
        cols.append(y)
    data_ = data[[key] + cols]
    method_map = {'auto': -1, 'brute_force': 0, 'fft': 1}
    thread_ratio = arg('thread_ratio', thread_ratio, float)
    method = arg('method', method, method_map)
    max_lag = arg('max_lag', max_lag, int)
    calulate_pacf = arg('calulate_pacf', calulate_pacf, bool)
    unique_id = str(uuid.uuid1()).replace('-', '_').upper()
    table = ["#CORREALATION_RESULT_{}".format(unique_id)]
    result_tbl = table[0]
    param_array = [('THREAD_RATIO', None, thread_ratio, None),
                   ('USE_FFT', method, None, None),
                   ('MAX_LAG', max_lag, None, None),
                   ('CALCULATE_PACF', calulate_pacf, None, None)]

    try:
        call_pal_auto(conn,
                      'PAL_CORRELATION_FUNCTION',
                      data_,
                      ParameterTable().with_data(param_array),
                      *table)
        return conn.table(result_tbl)
    except dbapi.Error as db_err:
        logger.exception(str(db_err))
        try_drop(conn, result_tbl)
        raise
