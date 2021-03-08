"""
This module contains Python wrapper for PAL fast dtw algorithm.

The following function is available:

    * :func:`fast_dtw`
"""

#pylint:disable=line-too-long, too-many-arguments, too-many-locals
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

def fast_dtw(data,
             radius,
             thread_ratio=None,
             distance_method=None,
             minkowski_power=None,
             save_alignment=None):#pylint:disable=too-many-arguments, too-few-public-methods, unused-argument, too-many-locals
    """
    DTW is an abbreviation for Dynamic Time Warping. It is a method for calculating distance or similarity between two time series.
    fast DTW is a twisted version of DTW to accelerate the computation when size of time series is huge.
    It recursively reduces the size of time series and calculate the DTW path on the reduced version,
    then refine the DTW path on the original ones. It may loss some accuracy of actual DTW distance in exchange of acceleration of computing.

    Note that this function is a new function in SAP HANA SPS05 and Cloud.

    Parameters
    ----------

    data : DataFrame
        Input data.
            - ID for multiple time series
            - Timestamps
            - Attributes of time series

    radius : int
        Parameter used for fast DTW algorithm. It is for balancing DTW accuracy and runtime.
        The bigger, the more accuracy but slower. Must be positive.

    thread_ratio : float, optional
        Controls the proportion of available threads to use.
        The ratio of available threads.
            - 0: single thread.
            - 0~1: percentage.
            - Others: heuristically determined.

        Defaults to -1.

    distance_method : {'manhattan', 'euclidean', 'minkowski', 'chebyshev', 'cosine'}, optional

        Specifies the method to compute the distance between two points.
            - 'manhattan': Manhattan distance
            - 'euclidean': Euclidean distance
            - 'minkowski': Minkowski distance
            - 'chebyshev': Chebyshev distance
            - 'cosine': Cosine distance

        Defaults to 'euclidean'.

    minkowski_power : double, optional
        Specifies the power of the Minkowski distance method.

        Only valid when ``distance_method`` is 'minkowski'.

        Defaults to 3.

    save_alignment : bool, optional
        Specifies if output alignment information. If True, output the table.

        Defaults to False.

    Returns
    -------

    DataFrame
        Result for fast dtw, structured as follows:
            - LEFT_<ID column name of input table>: ID of one time series.
            - RIGHT_<ID column name of input table>: ID of the other time series.
            - DISTANCE: DTW distance of two time series.

        Alignment table, structured as follows:
            - LEFT_<ID column name of input table>: ID of one time series.
            - RIGHT_<ID column name of input table>: ID of the other time series.
            - LEFT_INDEX: Corresponding to index of timestamps of time series with ID of 1st column.
            - RIGHT_INDEX : Corresponding to index of timestamps of time series with ID of 2nd column.

        Statistics for time series, structured as follows:
            - STAT_NAME: Statistics name.
            - STAT_VALUE: Statistics value.

    Examples
    --------
    >>> result, stats = fast_dtw(data, 5)

    """
    conn = data.connection_context
    require_pal_usable(conn)
    distance_map = {'manhattan': 1, 'euclidean': 2, 'minkowski': 3, 'chebyshev': 4, 'cosine': 6}#pylint: disable=line-too-long
    radius = arg('radius', radius, int)
    thread_ratio = arg('thread_ratio', thread_ratio, float)
    distance_method = arg('distance_method', distance_method, (int, str))
    if isinstance(distance_method, str):
        distance_method = arg('distance_method', distance_method, distance_map)
    minkowski_power = arg('minkowski_power', minkowski_power, float)
    save_alignment = arg('save_alignment', save_alignment, bool)

    unique_id = str(uuid.uuid1()).replace('-', '_').upper()
    outputs = ['RESULT', 'ALIGNMENT', 'STATS']
    outputs = ['#PAL_FAST_DTW_{}_TBL_{}_{}'.format(name, id, unique_id) for name in outputs]
    res_tbl, align_tbl, stats_tbl = outputs

    param_rows = [('RADIUS', radius, None, None),
                  ('THREAD_RATIO', None, thread_ratio, None),
                  ('DISTANCE_METHOD', distance_method, None, None),
                  ('MINKOWSKI_POWER', None, minkowski_power, None),
                  ('SAVE_ALIGNMENT ', save_alignment, None, None)]

    try:
        call_pal_auto(conn,
                      'PAL_FAST_DTW',
                      data,
                      ParameterTable().with_data(param_rows),
                      *outputs)

    except dbapi.Error as db_err:
        logger.exception(str(db_err))
        try_drop(conn, outputs)
        raise

    #pylint: disable=attribute-defined-outside-init, unused-variable
    return conn.table(res_tbl), conn.table(align_tbl), conn.table(stats_tbl)
