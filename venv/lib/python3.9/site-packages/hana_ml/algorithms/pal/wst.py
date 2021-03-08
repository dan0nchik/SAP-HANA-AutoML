#pylint:disable=too-many-lines, relative-beyond-top-level
'''
This module contains PAL wrappers for weighted_score_table algorithm.

The following functions is available:

    * :func:`weighted_score_table`
'''
import logging
import uuid
from hdbcli import dbapi
from .pal_base import (
    ParameterTable,
    arg,
    try_drop,
    ListOfStrings,
    require_pal_usable,
    call_pal_auto
)

logger = logging.getLogger(__name__)#pylint: disable=invalid-name

def weighted_score_table(data, maps, weights, key, features=None, thread_ratio=None):#pylint: disable=too-many-arguments
    """
    Perform the weighted_score_table to weight the score by the importance of each criterion.
    The alternative with the highest total score should be the best alternative.

    Parameters
    ----------

    data : DataFrame
        Input data.
    maps : DataFrame
        Every attribute (except ID) in the input data table maps to two columns
        in the map Function table: Key column and Value column.

        The Value column must be of DOUBLE type.
    weights : DataFrame
        This table has three columns.

        When the data table has n attributes (except ID), the weights table will have n rows.
    key : str
        Name of the ID column.
    features : str/ListOfStrings, optional
        Name of the feature columns.

        If not given, the feature columns should be all columns in the DataFrame
        except the ID column.
    thread_ratio : float, optional
        Specifies the ratio of total number of threads that can be used by this function.

        The value range is from 0 to 1, where 0 means only using 1 thread, and 1 means
        using at most all the currently available threads.

        Values outside the range will be ignored and this function heuristically
		determines the number of threads to use.

        Default to 0.

    Returns
    -------
    DataFrame
        The result value of weight for each score.

    Examples
    --------
    Data to analyze:

    >>> df_train = cc.table('WST_DATA_TBL')
    >>> df_train.collect()
            ID   GENDER   INCOME    HEIGHT
        0   0    male      5000     1.73
        1   1    male      9000     1.80
        2   2    female    6000     1.55
        3   3    male      15000    1.65
        4   4    female    2000     1.70
        5   5    female    12000    1.65
        6   6    male      1000     1.65
        7   7    male      8000     1.60
        8   8    female    5500     1.85
        9   9    female    9500     1.85

    >>> df_map = cc.table('WST_MAP_TBL')
    >>> df_map.collect()
             GENDER  VAL1   INCOME   VAL2   HEIGHT  VAL3
        0    male    2.0     0        0.0    1.5    0.0
        1    female  1.5     5500     1.0    1.6    1.0
        2    None    0.0     9000     2.0    1.71   2.0
        3    None    0.0     12000    3.0    1.80   3.0

    >>> df_weight = cc.table('WST_WEIGHT_TBL')
    >>> df_weight.collect()
            WEIGHT  ISDIS   ROWNUM
        0   0.5      1       2
        1   2.0      -1      4
        2   1.0      -1      4

    Perform weighted_score_table:

    >>> res = weighted_score_table(data = self.df_train,
                                   maps=self.df_map, weights=self.df_weight,
                                   key='ID', thread_ratio=0.3)
    >>> res.collect()
           ID  SCORE
        0   0   3.00
        1   1   8.00
        2   2   2.75
        3   3   8.00
        4   4   1.75
        5   5   7.75
        6   6   2.00
        7   7   4.00
        8   8   5.75
        9   9   7.75
    """
    conn = data.connection_context
    require_pal_usable(conn)
    key = arg('key', key, str)
    data_ = data
    if features is not None:
        if isinstance(features, str):
            features = [features]
        try:
            features = arg('features', features, ListOfStrings)#pylint: disable=undefined-variable
            data_ = data[[key] + features]
        except:
            msg = ("'features' must be list of string or string.")
            logger.error(msg)
            raise TypeError(msg)

    thread_ratio = arg('thread_ratio', thread_ratio, float)
    param_rows = [('THREAD_RATIO', None, thread_ratio, None)]
    unique_id = str(uuid.uuid1()).replace('-', '_').upper()
    result_tbl = "#WEIGHTED_TABLE_RESULT_{}".format(unique_id)
    try:
        call_pal_auto(conn,
                      'PAL_WEIGHTED_TABLE',
                      data_, maps, weights,
                      ParameterTable().with_data(param_rows),
                      result_tbl)
    except dbapi.Error as db_err:
        logger.exception(str(db_err))
        try_drop(conn, result_tbl)#pylint: disable=no-value-for-parameter
        raise
    return conn.table(result_tbl)
