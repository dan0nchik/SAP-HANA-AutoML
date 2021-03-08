"""This module contains python wrapper for PAL PageRank algorithm.

The following class is available:

    * :class:`PageRank`
"""

import logging
import uuid
from hdbcli import dbapi
from hana_ml.ml_base import try_drop
from .pal_base import (
    PALBase,
    ParameterTable,
    require_pal_usable,
    call_pal_auto
)

logger = logging.getLogger(__name__)#pylint: disable=invalid-name

class PageRank(PALBase):#pylint:disable=too-few-public-methods
    r"""
    A page rank model.

    Parameters
    ----------

    damping : float, optional
        The damping factor d.

        Defaults to 0.85.
    max_iter : int, optional
        The maximum number of iterations of power method.

        The value 0 means no maximum number of iterations is set
        and the calculation stops when the result converges.

        Defaults to 0.
    tol : float, optional
        Specifies the stop condition.

        When the mean improvement value of ranks is less than this value,
        the program stops calculation.

        Defaults to 1e-6.
    thread_ratio : float, optional
        Specifies the ratio of total number of threads that can be used by this function.

        The value range is from 0 to 1, where 0 means only using 1 thread, and 1 means
        using at most all the currently available threads.

        Values outside the range will be ignored and this function heuristically
        determines the number of threads to use.

        Defaults to 0.

    Attributes
    ----------

    None

    Examples
    --------

    Input dataframe df for training:

    >>> df.collect()
       FROM_NODE    TO_NODE
    0   Node1       Node2
    1   Node1       Node3
    2   Node1       Node4
    3   Node2       Node3
    4   Node2       Node4
    5   Node3       Node1
    6   Node4       Node1
    7   Node4       Node3

    Create a PageRank instance:

    >>> pr = PageRank()

    Call run() on given data sequence:

    >>> result = pr.run(data=df)
    >>> result.collect()
       NODE     RANK
    0   NODE1   0.368152
    1   NODE2   0.141808
    2   NODE3   0.287962
    3   NODE4   0.202078
    """
    #pylint: disable=too-many-arguments
    def __init__(self,
                 damping=None, # float
                 max_iter=None, # int
                 tol=None, # float
                 thread_ratio=None):  # float
        super(PageRank, self).__init__()
        self.damping = self._arg('damping', damping, float)
        self.max_iter = self._arg('max_iter', max_iter, int)
        self.tol = self._arg('tol', tol, float)
        self.thread_ratio = self._arg('thread_ratio', thread_ratio, float)

    def run(self, data):
        r"""
        This method reads link information and calculates rank for each node.

        Parameters
        ----------

        data : DataFrame
            Data for predicting the class labels.

        Returns
        -------

        DataFrame
            Calculated rank values and corresponding node names, structured as follows:

              - NODE: node names.
              - RANK: the PageRank of the corresponding node.

        """
        conn = data.connection_context
        require_pal_usable(conn)
        unique_id = str(uuid.uuid1()).replace('-', '_').upper()
        result_tbl = '#PAL_PAGERANK_RESULT_TBL_{}_{}'.format(self.id, unique_id)
        param_rows = [
            ('DAMPING', None, None if self.damping is None else float(self.damping), None),
            ('MAX_ITERATION', None if self.max_iter is None else int(self.max_iter), None, None),
            ('THRESHOLD', None, None if self.tol is None else float(self.tol), None),
            ('THREAD_RATIO', None,
             None if self.thread_ratio is None else float(self.thread_ratio), None)
            ]
        try:
            call_pal_auto(conn,
                          'PAL_PAGERANK',
                          data,
                          ParameterTable().with_data(param_rows),
                          result_tbl)
        except dbapi.Error as db_err:
            logger.exception(str(db_err))
            try_drop(conn, result_tbl)
            raise
        return conn.table(result_tbl)
