#pylint:disable=too-many-lines, relative-beyond-top-level, too-many-arguments
'''
This module contains PAL wrappers for abc_analysis algorithm.

The following functions is available:

    * :func:`abc_analysis`
'''
import logging
import uuid
from hdbcli import dbapi
from hana_ml.ml_base import try_drop
from .pal_base import (
    ParameterTable,
    arg,
    call_pal_auto
)
#pylint: disable=invalid-name, too-many-arguments, too-many-locals, line-too-long
logger = logging.getLogger(__name__)

def abc_analysis(data, key, percent_A, percent_B, percent_C,
                 revenue=None, thread_ratio=None):
    """
    Perform the abc_analysis to classify objects based on a particular
    measure. Group the inventories into three categories.

    Parameters
    ----------
    data : DataFrame
        Input data.
    key : str
        Name of the ID column.
    revenue : str, optional
        Name of column for revenue (or profits).

        If not given, the input dataframe must only have two columns.

        The first of the non-ID columns will be ``revenue``.
    percent_A : float
        Interval for A class.
    percent_B : float
        Interval for B class.
    percent_C : float
        Interval for C class.
    thread_ratio : float, optional
        Specifies the ratio of total number of threads that can be used by this function.

        The value range is from 0 to 1, where 0 means only using 1 thread, and 1 means
        using at most all the currently available threads.

        Values outside the range will be ignored and this function heuristically determines the number of threads to use.

        Default to 0.

    Returns
    -------
    DataFrame
        Returns a DataFrame containing the ABC class result of partitioning the data into three categories.

    Examples
    --------
    Data to analyze:

    >>> df_train = cc.table('AA_DATA_TBL')
    >>> df_train.collect()
         ITEM     VALUE
    0    item1    15.4
    1    item2    200.4
    2    item3    280.4
    3    item4    100.9
    4    item5    40.4
    5    item6    25.6
    6    item7    18.4
    7    item8    10.5
    8    item9    96.15
    9    item10   9.4

    Perform abc_analysis:

    >>> res = abc_analysis(data = self.df_train, key = 'ITEM', thread_ratio = 0.3,
                           percent_A = 0.7, percent_B = 0.2, percent_C = 0.1)
    >>> res.collect()
           ABC_CLASS   ITEM
    0      A        item3
    1      A        item2
    2      A        item4
    3      B        item9
    4      B        item5
    5      B        item6
    6      C        item7
    7      C        item1
    8      C        item8
    9      C        item10
    """
    conn_context = data.connection_context
    key = arg('key', key, str, True)
    revenue = arg('revenue', revenue, str)
    if revenue is None:
        if len(data.columns) != 2:
            msg = ("If 'revenue' is not given, the input dataframe " +
                   "must only have two columns.")
            logger.error(msg)
            raise ValueError(msg)
        revenue = data.columns[-1]
    data_ = data[[key, revenue]]
    thread_ratio = arg('thread_ratio', thread_ratio, float)
    param_rows = [('THREAD_RATIO', None, thread_ratio, None),
                  ('PERCENT_A', None, percent_A, None),
                  ('PERCENT_B', None, percent_B, None),
                  ('PERCENT_C', None, percent_C, None)
                 ]
    unique_id = str(uuid.uuid1()).replace('-', '_').upper()
    result_tbl = "#ABC_ANALYSIS_RESULT_{}".format(unique_id)
    try:
        call_pal_auto(conn_context,
                      'PAL_ABC',
                      data_,
                      ParameterTable().with_data(param_rows),
                      result_tbl)
    except dbapi.Error as db_err:
        logger.exception(str(db_err))
        try_drop(conn_context, result_tbl)
        raise
    return conn_context.table(result_tbl)
