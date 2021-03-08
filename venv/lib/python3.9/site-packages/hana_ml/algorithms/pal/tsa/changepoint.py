"""
This module contains Python wrapper for PAL change-point detection algorithm.

The following class is available:

    * :class:`CPD`
"""

#pylint:disable=too-many-lines, line-too-long, too-many-arguments, too-few-public-methods, too-many-instance-attributes
#pylint:disable=too-many-locals, no-else-return
import logging
import uuid
from hdbcli import dbapi
from hana_ml.algorithms.pal.pal_base import (
    PALBase,
    ParameterTable,
    ListOfStrings,
    try_drop,
    require_pal_usable,
    call_pal_auto
)

logger = logging.getLogger(__name__)#pylint:disable=invalid-name

class CPD(PALBase):
    r"""
    Change-point detection (CPDetection) methods aim at detecting multiple abrupt changes such as change in mean,
    variance or distribution in an observed time-series data.

    Parameters
    ----------

    cost : {'normal_mse', 'normal_rbf', 'normal_mhlb', 'normal_mv', 'linear', 'gamma', 'poisson', 'exponential', 'normal_m', 'negbinomial'}, optional

        The cost function for change-point detection.

        Defaults to 'normal_mse'.

    penalty : {'aic', 'bic', 'mbic', 'oracle', 'custom'}, optional

        The penalty function for change-point detection.

        Defaults to
            (1)'aic' if ``solver`` is 'pruneddp', 'pelt' or 'opt',

            (2)'custom' if ``solver`` is 'adppelt'.

    solver : {'pelt', 'opt', 'adppelt', 'pruneddp'}, optional

        Method for finding change-points of given data, cost and penalty.

        Each solver supports different cost and penalty functions.

          - 1.  For cost functions, 'pelt', 'opt' and 'adpelt' support the following eight:
                'normal_mse', 'normal_rbf', 'normal_mhlb', 'normal_mv',
                'linear', 'gamma', 'poisson', 'exponential';
                while 'pruneddp' supports the following four cost functions:
                'poisson', 'exponential', 'normal_m', 'negbinomial'.
          - 2.  For penalty functions, 'pruneddp' supports all penalties, 'pelt', 'opt' and 'adppelt' support the following three:
                'aic','bic','custom', while 'adppelt' only supports 'custom' cost.

        Defaults to 'pelt'.

    lamb : float, optional

        Assigned weight of the penalty w.r.t. the cost function, i.e. penalizaion factor.

        It can be seen as trade-off between speed and accuracy of running the detection algorithm.

        A small values (usually less than 0.1) will dramatically improve the efficiency.

        Defaults to 0.02, and valid only when ``solver`` is 'pelt' or 'adppelt'.

    min_size : int, optional

        The minimal length from the very begining within which change would not happen.

        Valid only when ``solver`` is 'opt', 'pelt' or 'adppelt'.

        Defaults to 2.

    min_sep : int, optional

        The minimal length of speration between consecutive change-points.

        Defaults to 1, valid only when ``solver`` is 'opt', 'pelt' or 'adppelt'.

    max_k : int, optional

        The maximum number of change-points to be detected.

        If the given value is less than 1, this number would be determined automatically from the input data.

        Defaults to 0, vaild only when ``solver`` is 'pruneddp'.

    dispersion : float, optinal

        Dispersion coefficient for Gamma and negative binomial distribution.

        Valid only when `cost` is 'gamma' or 'negbinomial'.

        Defaults to 1.0.

    lamb_range : list of two numericals(float and int) values, optional(deprecated)

        User-defined range of penalty.

        Only valid when ``solver`` is 'adppelt'.

        Deprecated, please use ``range_penalty`` instead.

    max_iter : int, optional

        Maximum number of iterations for searching the best penalty.

        Valid only when ``solver`` is 'adppelt'.

        Defaults to 40.

    range_penalty : list of two numerical values, optional

        User-defined range of penalty.

        Valid only when ``solver`` is 'adppelt' and ``value_penalty`` is not provided.

        Defaults to [0.01, 100].

    value_penalty : float, optional

        Value of user-defined penalty.

        Valid when ``penalty`` is 'custom' or ``solver`` is 'adppelt'.

        No default value.

    Attributes
    ----------

    stats_ : DataFrame

         Statistics for running change-point detection on the input data, structured as follows:
            - 1st column: statistics name,
            - 2nd column: statistics value.

    Examples
    --------

    First check the input time-series DataFrame df:

    >>> df.collect()
            TIME_STAMP  SERIES
    0        1-1       -5.36
    1        1-2       -5.14
    2        1-3       -4.94
    3        2-1       -5.15
    4        2-2       -4.95
    5        2-3        0.55
    6        2-4        0.88
    7        3-1        0.95
    8        3-2        0.68
    9        3-3        0.86

    Now create a CPD instance with 'pelt' solver and 'aic' penalty:

    >>> cpd = CPD(solver='pelt',
    ...           cost='normal_mse',
    ...           penalty='aic',
    ...           lamb=0.02)

    Apply the above CPD instance to the input data, check the detection result and
    related statistics:

    >>> cp = cpd.fit_predict(data=df)
    >>> cp.collect()
          TIME_STAMP
    0            2-2

    >>> cpd.stats_.collect()
                 STAT_NAME    STAT_VAL
    0               solver        Pelt
    1        cost function  Normal_MSE
    2         penalty type         AIC
    3           total loss     4.13618
    4  penalisation factor        0.02

    Create another CPD instance with 'adppelt' solver and 'normal_mv' cost:

    >>> cpd = CPD(solver='adppelt',
    ...           cost='normal_mv',
    ...           range_penalty=[0.01, 100],
    ...           lamb=0.02)

    Again, apply the above CPD instance to the input data, check the detection result and
    related statistics:

    >>> cp.collect()
               TIME_STAMP
         0            2-2

    >>> cpd.stats_.collect()
                 STAT_NAME   STAT_VAL
    0               solver    AdpPelt
    1        cost function  Normal_MV
    2         penalty type     Custom
    3           total loss   -28.1656
    4  penalisation factor       0.02
    5            iteration          2
    6      optimal penalty    2.50974

    Create a third CPD instance with 'pruneddp' solver and 'oracle' penalty:

    >>> cpd = CPD(solver='pruneddp', cost='normal_m', penalty='oracle', max_k=3)

    Simiar as before, apply the above CPD instance to the input data, check the detection result and
    related statistics:

    >>> cp = cpd.fit_predict(data=df)
    >>> cp.collect()
          TIME_STAMP
    0            2-2

    >>> cpd.stats_.collect()
                 STAT_NAME   STAT_VAL
    0               solver    AdpPelt
    1        cost function  Normal_MV
    2         penalty type     Custom
    3           total loss   -28.1656
    4  penalisation factor       0.02
    5            iteration          2
    6      optimal penalty    2.50974
    """

    solver_map = {'pelt':'Pelt', 'opt':'Opt', 'adppelt':'AdpPelt', 'pruneddp':'PrunedDP'}
    penalty_map = {'aic':'AIC', 'bic':'BIC', 'mbic':'mBIC', 'oracle':'Oracle', 'custom':'Custom'}
    cost_map = {'normal_mse':'Normal_MSE', 'normal_rbf':'Normal_RBF',
                'normal_mhlb':'Normal_MHLB', 'normal_mv':'Normal_MV',
                'linear':'Linear', 'gamma':'Gamma', 'poisson':'Poisson',
                'exponential':'Exponential', 'normal_m':'Normal_M',
                'negbinomial':'NegBinomial'}
    def __init__(self,#pylint: disable=too-many-arguments, too-many-locals, too-many-branches
                 cost=None,
                 penalty=None,
                 solver=None,
                 lamb=None,
                 min_size=None,
                 min_sep=None,
                 max_k=None,
                 dispersion=None,
                 lamb_range=None,
                 max_iter=None,
                 range_penalty=None,
                 value_penalty=None):
        super(CPD, self).__init__()
        self.cost = self._arg('cost', cost, self.cost_map)
        self.penalty = self._arg('penalty', penalty, self.penalty_map)
        self.solver = self._arg('solver', solver, self.solver_map)
        if self.solver in ('Pelt', 'Opt', None) and self.penalty not in ('AIC', 'BIC', 'Custom', None):
            msg = ("When 'solver' is 'pelt' or 'opt', "+
                   "only 'aic', 'bic' and 'custom' are valid penalty functions.")
            raise ValueError(msg)
        if self.solver == 'AdpPelt' and self.penalty not in ('Custom', None):
            msg = ("When 'solver' is 'adppelt', penalty function must be 'custom'.")
            raise ValueError(msg)
        cost_list_one = ['Normal_MSE', 'Normal_RBF', 'Normal_MHLB', 'Normal_MV',
                         'Linear', 'Gamma', 'Poisson', 'Exponential']
        cost_list_two = ['Poisson', 'Exponential', 'Normal_M', 'NegBinomial']
        if self.solver in ('Pelt', 'Opt', 'AdpPelt', None):
            if  self.cost is not None and self.cost not in cost_list_one:
                msg = ("'solver' is currently one of the following: pelt, opt and adppelt, "+
                       "in this case cost function must be one of the following: normal_mse, normal_rbf, "+
                       "normal_mhlb, normal_mv, linear, gamma, poisson, exponential.")
                raise ValueError(msg)
        elif self.cost is not None and self.cost not in cost_list_two:
            msg = ("'solver' is currently PrunedDP, in this case 'cost' must be assigned a valid value listed as follows: poisson, exponential, normal_m, negbinomial")
            raise ValueError(msg)
        self.lamb = self._arg('lamb', lamb, float)
        self.min_size = self._arg('min_size', min_size, int)
        self.min_sep = self._arg('min_sep', min_sep, int)
        self.max_k = self._arg('max_k', max_k, int)
        self.dispersion = self._arg('dispersion', dispersion, float)
        if lamb_range is not None:
            if isinstance(lamb_range, list) and len(lamb_range) == 2 and all(isinstance(val, (int, float)) for val in lamb_range):#pylint:disable=line-too-long
                self.lamb_range = lamb_range
            else:
                msg = ("Wrong setting for parameter 'lamb_range', correct setting "+
                       "should be a list of two numerical values that corresponds to "+
                       "lower- and upper-limit of the penelty weight.")
                raise ValueError(msg)
        else:
            self.lamb_range = None
        self.max_iter = self._arg('max_iter', max_iter, int)
        if range_penalty is not None:
            if isinstance(range_penalty, (list, tuple)) and len(range_penalty) == 2 and all(isinstance(val, (int, float)) for val in range_penalty):#pylint:disable=line-too-long
                self.lamb_range = list(range_penalty)
            else:
                msg = ("Wrong setting for parameter 'range_penalty', correct setting "+
                       "should be a list of two numerical values that corresponds to "+
                       "lower- and upper-limit of the penelty value.")
                raise ValueError(msg)
        else:
            self.lamb_range = None
        self.value_penalty = self._arg('value_penalty', value_penalty, float)

    def fit_predict(self, data, key=None, features=None):
        """
        Detecting change-points of the input data.

        Parameters
        ----------

        data : DataFrame

            Input time-series data for change-point detection.

        key : str, optional

            Column name for time-stamp of the input time-series data.

        features : str, optional

            Column name(s) for the value(s) of the input time-series data.

        Returns
        -------

        DataFrame

            Detected change-points of the input time-series data.
        """
        conn = data.connection_context
        require_pal_usable(conn)
        cols = data.columns
        key = self._arg('key', key, str)
        if key is None:
            key = cols[0]
        if features is not None:
            if isinstance(features, str):
                features = [features]
            try:
                features = self._arg('features', features, ListOfStrings)
            except:
                msg = ("'features' must be list of string or string.")
                #logger.error(msg)
                raise TypeError(msg)
        else:
            cols.remove(key)
            features = cols
        used_cols = [key] + features
        if any(col not in data.columns for col in used_cols):
            msg = "'key' or 'features' parameter contains unrecognized column name."
            raise ValueError(msg)
        data_ = data[used_cols]
        param_rows = [
            ('COSTFUNCTION', None, None, self.cost),
            ('SOLVER', None, None, self.solver),
            ('PENALIZATION_FACTOR', None, self.lamb, None),
            ('MIN_SIZE', self.min_size, None, None),
            ('MIN_SEP', self.min_sep, None, None),
            ('MaxK', self.max_k, None, None),
            ('DISPERSION', None, self.dispersion, None),
            ('MAX_ITERATION', self.max_iter, None, None)]
        if (self.penalty == 'Custom' or self.solver == 'AdpPelt') and self.value_penalty is not None:
            param_rows.extend([('PENALTY', None, self.value_penalty, 'Custom')])
        elif self.penalty not in ['Custom', None]:
            param_rows.extend([('PENALTY', None, None, self.penalty)])
        if self.lamb_range is not None:
            param_rows.extend([('RANGE_PENALTIES', None, None, str(self.lamb_range))])
        unique_id = str(uuid.uuid1()).replace('-', '_').upper()
        tables = ['RESULT', 'STATS']
        tables = ["#PAL_CPDETECTION_{}_TBL_{}_{}".format(tbl, self.id, unique_id) for tbl in tables]
        result_tbl, stats_tbl = tables
        try:
            call_pal_auto(conn,
                          "PAL_CPDETECTION",
                          data_,
                          ParameterTable().with_data(param_rows),
                          *tables)
        except dbapi.Error as db_err:
            logger.exception(str(db_err))
            try_drop(conn, tables)
        self.stats_ = conn.table(stats_tbl)#pylint:disable=attribute-defined-outside-init
        return conn.table(result_tbl)
