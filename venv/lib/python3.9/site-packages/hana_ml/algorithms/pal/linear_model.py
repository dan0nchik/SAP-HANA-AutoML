"""
This module contains Python wrappers for PAL linear model algorithms.

The following classes are available:

    * :class:`LinearRegression`
    * :class:`LogisticRegression`
"""

#pylint: disable=too-many-lines
#pylint: disable=line-too-long
#pylint: disable=relative-beyond-top-level
import itertools
import logging
import uuid

from hdbcli import dbapi
from hana_ml.ml_base import try_drop
from hana_ml.ml_exceptions import FitIncompleteError
from .sqlgen import trace_sql
from .pal_base import (
    PALBase,
    ParameterTable,
    parse_one_dtype,
    ListOfStrings,
    ListOfTuples,
    pal_param_register,
    require_pal_usable,
    call_pal_auto
)
from . import metrics

logger = logging.getLogger(__name__) #pylint: disable=invalid-name

class LinearRegression(PALBase):
    r"""
    Linear regression is an approach to model the linear relationship between a variable,
    usually referred to as dependent variable, and one or more variables, usually referred to as independent variables, denoted as predictor vector .

    Parameters
    ----------

    solver : {'QR', 'SVD', 'CD', 'Cholesky', 'ADMM'}, optional

        Algorithms to use to solve the least square problem. Case-insensitive.

              - 'QR': QR decomposition.
              - 'SVD': singular value decomposition.
              - 'CD': cyclical coordinate descent method.
              - 'Cholesky': Cholesky decomposition.
              - 'ADMM': alternating direction method of multipliers.

        'CD' and 'ADMM' are supported only when ``var_select`` is 'all'.

        Defaults to QR decomposition.

    var_select : {'all', 'forward', 'backward', 'stepwise'}, optional

        Method to perform variable selection.

            - 'all': all variables are included.
            - 'forward': forward selection.
            - 'backward': backward selection.
            - 'stepwise': stepwise selection.

        'forward' and 'backward' selection are supported only when ``solver``
        is 'QR', 'SVD' or 'Cholesky'.

        Defaults to 'all'.

    features_must_select: str or list of str, optional

        Specifies the column name that needs to be included in the final training model when executing the variable selection.

        This parameter can be specified multiple times, each time with one column name as feature.

        Only valid when ``var_select`` is not 'all'.

        Note that This parameter is a hint.
        There are exceptional cases that a specified mandatory feature is excluded in the final model.

        For instance, some mandatory features can be represented as a linear combination of other features, among which some are also mandatory features.

        New parameter added in SAP HANA Cloud and SPS05.

        No default value.

    intercept : bool, optional

        If true, include the intercept in the model.

        Defaults to True.

    alpha_to_enter : float, optional

        P-value for forward selection.

        Valid only when ``var_select`` is 'forward' or 'stepwise'.

        Defaults to 0.05 when ``var_select`` is 'forward', 0.15 when ``var_select`` is 'stepwise'.

    alpha_to_remove : float, optional

        P-value for backward selection.

        Valid only when ``var_select`` is 'backward' or 'stepwise'.

        Defaults to 0.1 when `var_select`` is 'backward', and 0.15  when ``var_select`` is 'stepwise'.

    enet_lambda : float, optional

        Penalized weight. Value should be greater than or equal to 0.

        Valid only when ``solver`` is 'CD' or 'ADMM'.

    enet_alpha : float, optional

        Elastic net mixing parameter.

        Ranges from 0 (Ridge penalty) to 1 (LASSO penalty) inclusively.

        Valid only when ``solver`` is 'CD' or 'ADMM'.

        Defaults to 1.0.

    max_iter : int, optional

        Maximum number of passes over training data.

        If convergence is not reached after the specified number of
        iterations, an error will be generated.

        Valid only when ``solver`` is 'CD' or 'ADMM'.

        Defaults to 1e5.

    tol : float, optional

        Convergence threshold for coordinate descent.

        Valid only when ``solver`` is 'CD'.

        Defaults to 1.0e-7.

    pho : float, optional

        Step size for ADMM. Generally, it should be greater than 1.

        Valid only when ``solver`` is 'ADMM'.

        Defaults to 1.8.

    stat_inf : bool, optional

        If true, output t-value and Pr(>|t|) of coefficients.

        Defaults to False.

    adjusted_r2 : bool, optional

        If true, include the adjusted R2 value in statistics.

        Defaults to False.

    dw_test : bool, optional

        If true, conduct Durbin-Watson test under null hypothesis that errors do not follow a first order autoregressive process.

        Not available if elastic net regularization is enabled or intercept is ignored.

        Defaults to False.

    reset_test : int, optional

        Specifies the order of Ramsey RESET test.

        Ramsey RESET test with power of variables ranging from 2 to this value (greater than 1) will be conducted.

        Value 1 means RESET test will not be conducted. Not available if elastic net regularization is enabled or intercept is ignored.

        Defaults to 1.

    bp_test : bool, optional

        If true, conduct Breusch-Pagan test under null hypothesis that homoscedasticity is satisfied.

        Not available if elastic net regularization is enabled or intercept is ignored.

        Defaults to False.

    ks_test : bool, optional

        If true, conduct Kolmogorov-Smirnov normality test under null hypothesis that errors follow a normal distribution.

        Not available if elastic net regularization is enabled or intercept is ignored.

        Defaults to False.

    thread_ratio : float, optional

        Controls the proportion of available threads to use.

        The value range is from 0 to 1, where 0 means only using 1 thread, and 1 means using at most all the currently available threads.

        Values outside this range tell PAL to heuristically determine the number of threads to use.

        Valid only when ``solver`` is 'QR', 'CD', 'Cholesky' or 'ADMM'.

        Defaults to 0.0.

    categorical_variable : str or ist of str, optional

        Specifies INTEGER columns specified that should be be treated as categorical.

        Other INTEGER columns will be treated as continuous.

    pmml_export : {'no', 'multi-row'}, optional

        Controls whether to output a PMML representation of the model, and how to format the PMML. Case-insensitive.

            - 'no' or not provided: No PMML model.
            - 'multi-row': Exports a PMML model, splitting it
              across multiple rows if it doesn't fit in one.

        Prediction does not require a PMML model.

    resampling_method : {'cv', 'bootstrap'}, optional

        Specifies the resampling method for model evaluation/parameter selection.

        If no value is specified for this parameter, neither model evaluation

        nor parameter selection is activated.

        Must be set together with `evaluation_metric`.

        No default value.

    evaluation_metric : {'rmse'}, optional

        Specifies the evaluation metric for model evaluation or parameter selection.

        Must be set together with `resampling_method`.

        No default value.

    fold_num : int, optional

        Specifies the fold number for the cross validation method.
        Mandatory and valid only when `resampling_method` is set to 'cv'.

        No default value.

    repeat_times : int, optional

        Specifies the number of repeat times for resampling.

        Defaults to 1.

    search_strategy : {'grid', 'random'}, optional

        Specifies the method to activate parameter selection.

        No default value.

    random_search_times : int, optional

        Specifies the number of times to randomly select candidate parameters for selection.

        Mandatory and valid when `search_strategy` is set to 'random'.

        No default value.

    random_state : int, optional

        Specifies the seed for random generation. Use system time when 0 is specified.

        Defaults to 0.

    timeout : int, optional

        Specifies maximum running time for model evaluation or parameter

        selection, in seconds. No timeout when 0 is specified.

        Defaults to 0.

    progress_indicator_id : str, optional

        Sets an ID of progress indicator for model evaluation or parameter selection.

        No progress indicator is active if no value is provided.

        No default value.

    param_values : dict or list of tuples, optional

        Specifies values of specific parameters to be selected.

        Valid only when ``resampling_method`` and ``search_strategy`` are both specified.

        Specified parameters could be ``enet_lambda`` and ``enet_alpha``.

        No default value.

    param_range : dict or list of tuples, optional

        Specifies range of specific parameters to be selected.

        Valid only when ``resampling_method`` and ``search_strategy`` are both specified.

        Specified parameters could be ``enet_lambda``, ``enet_alpha``.

        No default value.


    Attributes
    ----------

    coefficients_ : DataFrame

        Fitted regression coefficients.

    pmml_ : DataFrame

        PMML model. Set to None if no PMML model was requested.

    fitted_ : DataFrame

        Predicted dependent variable values for training data.
        Set to None if the training data has no row IDs.

    statistics_ : DataFrame

        Regression-related statistics, such as mean squared error.

    optim_param_ : DataFrame
        If parameter selection is enabled, the optimal parameters will be selected.

    Examples
    --------

    Training data:

    >>> df.collect()
      ID       Y    X1 X2  X3
    0  0  -6.879  0.00  A   1
    1  1  -3.449  0.50  A   1
    2  2   6.635  0.54  B   1
    3  3  11.844  1.04  B   1
    4  4   2.786  1.50  A   1
    5  5   2.389  0.04  B   2
    6  6  -0.011  2.00  A   2
    7  7   8.839  2.04  B   2
    8  8   4.689  1.54  B   1
    9  9  -5.507  1.00  A   2

    Training the model:

    >>> lr = LinearRegression(thread_ratio=0.5,
    ...                       categorical_variable=["X3"])
    >>> lr.fit(data=df, key='ID', label='Y')

    Prediction:

    >>> df2.collect()
       ID     X1 X2  X3
    0   0  1.690  B   1
    1   1  0.054  B   2
    2   2  0.123  A   2
    3   3  1.980  A   1
    4   4  0.563  A   1
    >>> lr.predict(data=df2, key='ID').collect()
       ID      VALUE
    0   0  10.314760
    1   1   1.685926
    2   2  -7.409561
    3   3   2.021592
    4   4  -3.122685
    """
    # pylint: disable=too-many-arguments,too-many-instance-attributes, too-many-locals, too-many-statements, too-many-branches
    solver_map = {'qr': 1, 'svd': 2, 'cd': 4, 'cholesky': 5, 'admm': 6}
    var_select_map = {'all': 0, 'forward': 1, 'backward': 2, 'stepwise':3}
    pmml_export_map = {'no': 0, 'multi-row': 2}
    values_list = {"enet_alpha": "ENET_ALPHA", "enet_lambda": "ENET_LAMBDA"}
    resampling_method_map = {'cv': 'cv', 'bootstrap': 'bootstrap'}
    evaluation_metric_map = {'rmse': 'RMSE'}
    search_strategy_map = {'grid': 'grid', 'random': 'random'}
    def __init__(self,
                 solver=None,
                 var_select=None,
                 features_must_select=None,
                 intercept=True,
                 alpha_to_enter=None,
                 alpha_to_remove=None,
                 enet_lambda=None,
                 enet_alpha=None,
                 max_iter=None,
                 tol=None,
                 pho=None,
                 stat_inf=False,
                 adjusted_r2=False,
                 dw_test=False,
                 reset_test=None,
                 bp_test=False,
                 ks_test=False,
                 thread_ratio=None,
                 categorical_variable=None,
                 pmml_export=None,
                 resampling_method=None,
                 evaluation_metric=None,
                 fold_num=None,
                 repeat_times=None,
                 search_strategy=None,
                 random_search_times=None,
                 random_state=None,
                 timeout=None,
                 progress_indicator_id=None,
                 param_values=None,
                 param_range=None):
        super(LinearRegression, self).__init__()
        setattr(self, 'hanaml_parameters', pal_param_register())
        self.solver = self._arg('solver', solver, self.solver_map)
        self.var_select = self._arg('var_select', var_select, self.var_select_map)
        if isinstance(features_must_select, str):
            features_must_select = [features_must_select]
        self.features_must_select = self._arg('features_must_select',
                                              features_must_select, ListOfStrings)
        self.intercept = self._arg('intercept', intercept, bool)
        self.alpha_to_enter = self._arg('alpha_to_enter', alpha_to_enter, float)
        self.alpha_to_remove = self._arg('alpha_to_remove', alpha_to_remove, float)
        self.enet_lambda = self._arg('enet_lambda', enet_lambda, float)
        self.enet_alpha = self._arg('enet_alpha', enet_alpha, float)
        self.max_iter = self._arg('max_iter', max_iter, int)
        self.tol = self._arg('tol', tol, float)
        self.pho = self._arg('pho', pho, float)
        self.stat_inf = self._arg('stat_inf', stat_inf, bool)
        self.adjusted_r2 = self._arg('adjusted_r2', adjusted_r2, bool)
        self.dw_test = self._arg('dw_test', dw_test, bool)
        self.reset_test = self._arg('reset_test', reset_test, int)
        self.bp_test = self._arg('bp_test', bp_test, bool)
        self.ks_test = self._arg('ks_test', ks_test, bool)
        self.thread_ratio = self._arg('thread_ratio', thread_ratio, float)
        if isinstance(categorical_variable, str):
            categorical_variable = [categorical_variable]
        self.categorical_variable = self._arg('categorical_variable',
                                              categorical_variable, ListOfStrings)
        self.pmml_export = self._arg('pmml_export', pmml_export, self.pmml_export_map)
        if solver is not None:
            if solver.lower() == 'cd' or solver.lower() == 'admm':
                if var_select is not None and var_select.lower() != 'all':
                    msg = ('var_select cannot be {} when solver ' +
                           'is {}.').format(var_select.lower(), solver.lower())
                    logger.error(msg)
                    raise ValueError(msg)
        if solver is None or (solver.lower() != 'cd' and solver.lower() != 'admm'):
            if enet_lambda is not None:
                msg = ('enet_lambda is applicable only when solver is ' +
                       'coordinate descent or admm.')
                logger.error(msg)
                raise ValueError(msg)
            if enet_alpha is not None:
                msg = ('enet_alpha is applicable only when solver is ' +
                       'coordinate descent or admm.')
                logger.error(msg)
                raise ValueError(msg)
            if max_iter is not None:
                msg = ('max_iter is applicable only when solver is ' +
                       'coordinate descent or admm.')
                logger.error(msg)
                raise ValueError(msg)
        if (solver is None or solver.lower() != 'cd') and tol is not None:
            msg = 'tol is applicable only when solver is coordinate descent.'
            logger.error(msg)
            raise ValueError(msg)
        if (solver is None or solver.lower() != 'admm') and pho is not None:
            msg = 'pho is applicable only when solver is admm.'
            logger.error(msg)
            raise ValueError(msg)
        if var_select is None or var_select.lower() != 'forward':
            if alpha_to_enter is not None:
                msg = 'alpha_to_enter is applicable only when var_select is forward.'
                logger.error(msg)
                raise ValueError(msg)
        if var_select is None or var_select.lower() != 'backward':
            if alpha_to_remove is not None:
                msg = 'alpha_to_remove is applicable only when var_select is backward.'
                logger.error(msg)
                raise ValueError(msg)
        if enet_lambda is not None or enet_alpha is not None or intercept is False:
            if dw_test is not None and dw_test is not False:
                msg = ('dw_test is applicable only when elastic net regularization ' +
                       'is disabled and the model includes an intercept.')
                logger.error(msg)
                raise ValueError(msg)
            if reset_test is not None and reset_test != 1:
                msg = ('reset_test is applicable only when elastic net regularization ' +
                       'is disabled and the model includes an intercept.')
                logger.error(msg)
                raise ValueError(msg)
            if bp_test is not None and bp_test is not False:
                msg = ('bp_test is applicable only when elastic net regularization ' +
                       'is disabled and the model includes an intercept.')
                logger.error(msg)
                raise ValueError(msg)
            if ks_test is not None and ks_test is not False:
                msg = ('ks_test is applicable only when elastic net regularization ' +
                       'is disabled and the model includes an intercept.')
                logger.error(msg)
                raise ValueError(msg)
        if isinstance(reset_test, bool):
            msg = ('reset_test should be an integer, not a boolean ' +
                   'indicating whether or not to conduct the Ramsey RESET test.')
            logger.error(msg)
            raise TypeError(msg)
        if enet_alpha is not None and not 0 <= enet_alpha <= 1:
            msg = 'enet_alpha {!r} is out of bounds.'.format(enet_alpha)
            logger.error(msg)
            raise ValueError(msg)
        self.resampling_method = self._arg('resampling_method', resampling_method,
                                           self.resampling_method_map)
        self.evaluation_metric = self._arg('evaluation_metric', evaluation_metric,
                                           self.evaluation_metric_map)
        self.fold_num = self._arg('fold_num', fold_num, int)
        self.repeat_times = self._arg('repeat_times', repeat_times, int)
        self.search_strategy = self._arg('search_strategy', search_strategy,
                                         self.search_strategy_map)
        self.random_search_times = self._arg('random_search_times', random_search_times, int)
        self.random_state = self._arg('random_state', random_state, int)
        self.timeout = self._arg('timeout', timeout, int)
        self.progress_indicator_id = self._arg('progress_indicator_id',
                                               progress_indicator_id, str)
        if isinstance(param_range, dict):
            param_range = [(x, param_range[x]) for x in param_range]
        self.param_range = self._arg('param_range', param_range, ListOfTuples)
        if isinstance(param_values, dict):
            param_values = [(x, param_values[x]) for x in param_values]
        self.param_values = self._arg('param_values', param_values, ListOfTuples)
        search_param_count = 0
        for param in (self.resampling_method, self.evaluation_metric):
            if param is not None:
                search_param_count += 1
        if search_param_count not in (0, 2):
            msg = ("`resampling_method`, and `evaluation_metric` must be set together.")
            logger.error(msg)
            raise ValueError(msg)
        if self.search_strategy is not None and self.resampling_method is None:
            msg = ("`search_strategy` cannot be set if `resampling_method` is not specified.")
            logger.error(msg)
            raise ValueError(msg)
        if self.resampling_method == 'cv' and self.fold_num is None:
            msg = ("`fold_num` must be set when "+
                   "`resampling_method` is set as 'cv'.")
            logger.error(msg)
            raise ValueError(msg)
        if self.resampling_method != 'cv' and self.fold_num is not None:
            msg = ("`fold_num` is not valid when parameter selection is not" +
                   " enabled, or `resampling_method` is not set as 'cv'.")
            logger.error(msg)
            raise ValueError(msg)
        if self.search_strategy == 'random' and self.random_search_times is None:
            msg = ("`random_search_times` must be set when "+
                   "`search_strategy` is set as 'random'.")
            logger.error(msg)
            raise ValueError(msg)
        if self.search_strategy in (None, 'grid') and self.random_search_times is not None:
            msg = ("`random_search_times` is not valid " +
                   "when parameter selection is not enabled"+
                   ", or `search_strategy` is not set as 'random'.")
            logger.error(msg)
            raise ValueError(msg)
        if self.search_strategy is None:
            if self.param_values is not None:
                msg = ("`param_values` can only be specified "+
                       "when `search_strategy` is enabled.")
                logger.error(msg)
                raise ValueError(msg)
            if self.param_range is not None:
                msg = ("`param_range` can only be specified "+
                       "when `search_strategy` is enabled.")
                logger.error(msg)
                raise ValueError(msg)
        if self.search_strategy is not None:
            set_param_list = list()
            if self.enet_lambda is not None:
                set_param_list.append("enet_lambda")
            if self.enet_alpha is not None:
                set_param_list.append("enet_alpha")
            if self.param_values is not None:
                for x in self.param_values:#pylint:disable=invalid-name
                    if len(x) != 2:#pylint:disable=bad-option-value
                        msg = ("Each tuple that specifies the values of a parameter should"+
                               " contain exactly 2 elements: 1st is parameter name,"+
                               " 2nd is a list of valid values.")
                        logger.error(msg)
                        raise ValueError(msg)
                    if x[0] not in self.values_list:
                        msg = ("Specifying the values of `{}` for ".format(x[0])+
                               "parameter selection is invalid.")
                        logger.error(msg)
                        raise ValueError(msg)
                    if x[0] in set_param_list:
                        msg = ("Parameter `{}` is invalid for ".format(x[0])+
                               "re-specification in parameter selection.")
                        logger.error(msg)
                        raise ValueError(msg)
                    if self.solver not in (4, 6):
                        msg = ("Parameter `{}` is invalid when ".format(x[0])+
                               "`solver` is not set as 'cd' or 'admm'.")
                        logger.error(msg)
                        raise ValueError(msg)
                    if not (isinstance(x[1], list) and all(isinstance(t, (int, float)) for t in x[1])):#pylint:disable=line-too-long
                        msg = "Valid values of `{}` must be a list of numericals.".format(x[0])
                        logger.error(msg)
                        raise TypeError(msg)
                    set_param_list.append(x[0])

            if self.param_range is not None:
                rsz = [3] if self.search_strategy == 'grid'else [2, 3]
                for x in self.param_range:#pylint:disable=invalid-name
                    if len(x) != 2:#pylint:disable=bad-option-value
                        msg = ("Each tuple that specifies the range of a parameter should contain"+
                               " exactly 2 elements: 1st is parameter name, 2nd is value range.")
                        logger.error(msg)
                        raise ValueError(msg)
                    if x[0] not in self.values_list:
                        msg = ("Specifying the values of `{}` for ".format(x[0])+
                               "parameter selection is invalid.")
                        logger.error(msg)
                        raise ValueError(msg)
                    if x[0] in set_param_list:
                        msg = ("Parameter `{}` is invalid for ".format(x[0])+
                               "re-specification in parameter selection.")
                        logger.error(msg)
                        raise ValueError(msg)
                    if self.solver not in (4, 6):
                        msg = ("Parameter `{}` is invalid when ".format(x[0])+
                               "`solver` is not set as 'cd' or 'admm'.")
                        logger.error(msg)
                        raise ValueError(msg)
                    if not (isinstance(x[1], list) and len(x[1]) in rsz and all(isinstance(t, (int, float)) for t in x[1])):#pylint:disable=line-too-long
                        msg = ("The provided `{}` is either not ".format(x[0])+
                               "a list of numericals, or it contains wrong number of values.")
                        logger.error(msg)
                        raise TypeError(msg)

    @trace_sql
    def fit(self, data, key=None, features=None,
            label=None, categorical_variable=None):
        r"""
        Fit regression model based on training data.

        Parameters
        ----------

        data : DataFrame

            Training data.

        key : str, optional

            Name of the ID column.

            If ``key`` is not provided, it is assumed
            that the input has no ID column.

        features : list of str, optional

            Names of the feature columns.

            If ``features`` is not provided, it defaults to all non-ID,
            non-label columns.

        label : str, optional

            Name of the dependent variable.
            If ``label`` is not provided, it defaults to the last column.

        categorical_variable : str or list of str, optional

            Specifies INTEGER column(s) that should be treated as categorical.

            Other INTEGER columns will be treated as continuous.
        """
        # pylint: disable=too-many-locals
        conn = data.connection_context
        require_pal_usable(conn)
        key = self._arg('key', key, str)
        features = self._arg('features', features, ListOfStrings)
        label = self._arg('label', label, str)
        if isinstance(categorical_variable, str):
            categorical_variable = [categorical_variable]
        categorical_variable = self._arg('categorical_variable', categorical_variable, ListOfStrings)
        cols = data.columns
        if key is not None:
            id_col = [key]
            cols.remove(key)
        else:
            id_col = []
        if label is None:
            label = cols[-1]
        cols.remove(label)
        if features is None:
            features = cols
        data_ = data[id_col + [label] + features]
        unique_id = str(uuid.uuid1()).replace('-', '_').upper()
        outputs = ['COEF', 'PMML', 'FITTED', 'STATS', 'OPTIMAL_PARAM']
        outputs = ['#PAL_LINEAR_REGRESSION_{}_TBL_{}_{}'.format(name, self.id, unique_id)
                   for name in outputs]
        coef_tbl, pmml_tbl, fitted_tbl, stats_tbl, opt_param_tbl = outputs
        param_rows = [('ALG', self.solver, None, None),
                      ('VARIABLE_SELECTION', self.var_select, None, None),
                      ('NO_INTERCEPT', not self.intercept, None, None),
                      ('ALPHA_TO_ENTER', None, self.alpha_to_enter, None),
                      ('ALPHA_TO_REMOVE', None, self.alpha_to_remove, None),
                      ('ENET_LAMBDA', None, self.enet_lambda, None),
                      ('ENET_ALPHA', None, self.enet_alpha, None),
                      ('MAX_ITERATION', self.max_iter, None, None),
                      ('THRESHOLD', None, self.tol, None),
                      ('PHO', None, self.pho, None),
                      ('STAT_INF', self.stat_inf, None, None),
                      ('ADJUSTED_R2', self.adjusted_r2, None, None),
                      ('DW_TEST', self.dw_test, None, None),
                      ('RESET_TEST', self.reset_test, None, None),
                      ('BP_TEST', self.bp_test, None, None),
                      ('KS_TEST', self.ks_test, None, None),
                      ('PMML_EXPORT', self.pmml_export, None, None),
                      ('HAS_ID', key is not None, None, None),
                      ('RESAMPLING_METHOD', None, None, self.resampling_method),
                      ('EVALUATION_METRIC', None, None, self.evaluation_metric),
                      ('SEED', self.random_state, None, None),
                      ('REPEAT_TIMES', self.repeat_times, None, None),
                      ('PARAM_SEARCH_STRATEGY', None, None, self.search_strategy),
                      ('FOLD_NUM', self.fold_num, None, None),
                      ('RANDOM_SEARCH_TIMES', self.random_search_times, None, None),
                      ('TIMEOUT', self.timeout, None, None),
                      ('PROGRESS_INDICATOR_ID', None, None, self.progress_indicator_id)]
        if self.param_values is not None:
            for x in self.param_values:#pylint:disable=invalid-name
                values = str(x[1]).replace('[', '{').replace(']', '}')
                param_rows.extend([(self.values_list[x[0]]+"_VALUES",
                                    None, None, values)])
        if self.param_range is not None:
            for x in self.param_range:#pylint:disable=invalid-name
                range_ = str(x[1])
                if len(x[1]) == 2 and self.search_strategy == 'random':
                    range_ = range_.replace(',', ',,')
                param_rows.extend([(self.values_list[x[0]]+"_RANGE",
                                    None, None, range_)])
        if self.solver != 2:
            param_rows.append(('THREAD_RATIO', None, self.thread_ratio, None))
        if self.features_must_select is not None:
            param_rows.extend(('MANDATORY_FEATURE', None, None, variable)
                              for variable in self.features_must_select)
        if self.categorical_variable is not None:
            param_rows.extend(('CATEGORICAL_VARIABLE', None, None, variable)
                              for variable in self.categorical_variable)
        if categorical_variable is not None:
            param_rows.extend(('CATEGORICAL_VARIABLE', None, None, variable)
                              for variable in categorical_variable)
        try:
            call_pal_auto(conn,
                          'PAL_LINEAR_REGRESSION',
                          data_,
                          ParameterTable().with_data(param_rows),
                          *outputs)
        except dbapi.Error as db_err:
            logger.exception(str(db_err))
            try_drop(conn, outputs)
            raise
        # pylint: disable=attribute-defined-outside-init
        self.coefficients_ = conn.table(coef_tbl)
        self.pmml_ = conn.table(pmml_tbl) if self.pmml_export else None
        self.fitted_ = conn.table(fitted_tbl) if key is not None else None
        self.statistics_ = conn.table(stats_tbl)
        self.optim_param_ = conn.table(opt_param_tbl)
        self.model_ = self.coefficients_

    @trace_sql
    def predict(self, data, key, features=None):
        r"""
        Predict dependent variable values based on fitted model.

        Parameters
        ----------

        data : DataFrame

            Independent variable values to predict for.

        key : str

            Name of the ID column.

        features : list of str, optional

            Names of the feature columns.

            If ``features`` is not provided, it defaults to all non-ID columns.

        Returns
        -------

        DataFrame

            Predicted values, structured as follows:

                - ID column: with same name and type as ``data`` 's ID column.
                - VALUE: type DOUBLE, representing predicted values.

            .. note  ::

                predict() will pass the ``pmml_`` table to PAL as the model
                representation if there is a ``pmml_`` table, or the ``coefficients_``
                table otherwise.
        """
        conn = data.connection_context
        require_pal_usable(conn)
        if getattr(self, 'pmml_', None) is not None:
            model = self.pmml_
        elif hasattr(self, 'model_'):
            model = self.model_
        else:
            raise FitIncompleteError("Model not initialized. Perform a fit first.")
        key = self._arg('key', key, str, required=True)
        features = self._arg('features', features, ListOfStrings)
        cols = data.columns
        cols.remove(key)
        if features is None:
            features = cols
        data_ = data[[key] + features]
        unique_id = str(uuid.uuid1()).replace('-', '_').upper()
        fitted_tbl = '#PAL_LINEAR_REGRESSION_FITTED_TBL_{}_{}'.format(self.id, unique_id)
        param_rows = [('THREAD_RATIO', None, self.thread_ratio, None)]
        try:
            call_pal_auto(conn,
                          'PAL_LINEAR_REGRESSION_PREDICT',
                          data_,
                          model,
                          ParameterTable().with_data(param_rows),
                          fitted_tbl)
        except dbapi.Error as db_err:
            logger.exception(str(db_err))
            try_drop(conn, fitted_tbl)
            raise

        return conn.table(fitted_tbl)

    def score(self, data, key, features=None, label=None):
        r"""
        Returns the coefficient of determination R2 of the prediction.

        Parameters
        ----------

        data : DataFrame

            Data on which to assess model performance.

        key : str

            Name of the ID column.

        features : list of str, optional

            Names of the feature columns.

            If ``features`` is not provided, it defaults all non-ID,
            non-label columns.

        label : str, optional

            Name of the dependent variable.

            If ``label`` is not provided, it defaults to the last column.

        Returns
        -------

        float

            Returns the coefficient of determination R2 of the prediction.

            .. note ::

                score() will pass the ``pmml_`` table to PAL as the model
                representation if there is a ``pmml_`` table, or the ``coefficients_``
                table otherwise.
        """
        if getattr(self, 'pmml_', None) is None and not hasattr(self, 'coefficients_'):
            raise FitIncompleteError("Model not initialized. Perform a fit first.")
        key = self._arg('key', key, str, required=True)
        features = self._arg('features', features, ListOfStrings)
        label = self._arg('label', label, str)
        cols = data.columns
        cols.remove(key)
        if label is None:
            label = cols[-1]
        cols.remove(label)
        if features is None:
            features = cols
        prediction = self.predict(data, key=key, features=features)
        prediction = prediction.rename_columns(['ID_P', 'PREDICTION'])
        original = data[[key, label]].rename_columns(['ID_A', 'ACTUAL'])
        joined = original.join(prediction, 'ID_P=ID_A').select('ACTUAL', 'PREDICTION')
        return metrics.r2_score(joined,
                                label_true='ACTUAL',
                                label_pred='PREDICTION')

class LogisticRegression(PALBase):#pylint:disable=too-many-instance-attributes
    r"""
    Logistic regression model that handles binary-class and multi-class
    classification problems.

    Parameters
    ----------

    multi_class : bool, optional

        If true, perform multi-class classification. Otherwise, there must be only two classes.

        Defaults to False.

    max_iter : int, optional

        Maximum number of iterations taken for the solvers to converge. If convergence is not reached after this number, an error will be generated.

                - multi-class: Defaults to 100.
                - binary-class: Defaults to 100000 when ``solver`` is cyclical, 1000 when ``solver`` is proximal, otherwise 100.

    pmml_export : {'no', 'single-row', 'multi-row'}, optional

        Controls whether to output a PMML representation of the model, and how to format the PMML. Case-insensitive.

          - multi-class:

            - 'no' or not provided: No PMML model.
            - 'multi-row': Exports a PMML model, splitting it across multiple rows if it doesn't fit in one.

          - binary-class:

            - 'no' or not provided: No PMML model.
            - 'single-row': Exports a PMML model in a maximum of one row. Fails if the model doesn't fit in one row.
            - 'multi-row': Exports a PMML model, splitting it across multiple rows if it doesn't fit in one.

        Defaults to 'no'.

    categorical_variable : str or list of str, optional(deprecated)

        Specifies INTEGER column(s) in the data that should be treated category variable.

    standardize : bool, optional

        If true, standardize the data to have zero mean and unit
        variance.

        Defaults to True.

    stat_inf : bool, optional

        If true, proceed with statistical inference.

        Defaults to False.

    solver : {'auto', 'newton', 'cyclical', 'lbfgs', 'stochastic', 'proximal'}, optional

        Optimization algorithm.

            - 'auto' : automatically determined by system based on input data and parameters.
            - 'newton': Newton iteration method, can only solve ridge regression problems.
            - 'cyclical': Cyclical coordinate descent method to fit elastic net regularized logistic regression.
            - 'lbfgs': LBFGS method (recommended when having many independent variables, can only solve ridge regression \
			problems when ``multi_class`` is True).
            - 'stochastic': Stochastic gradient descent method (recommended when dealing with very large dataset).
            - 'proximal': Proximal gradient descent method to fit elastic net regularized logistic regression.

        When ``multi_class`` is True, only 'auto', 'lbfgs' and 'cyclical' are valid solvers.

        Defaults to 'auto'.

        .. note : If it happens that the enet regularization term contains LASSO penalty, while a solver that \
        can only solve ridge regression problems is specified, then the specified solver will be ignored(hence default value \
        is used). The users can check the statistical table for the solver that has been adopted finally.

    enet_alpha : float, optional

        Elastic net mixing parameter.

        Only valid when ``multi_class`` is False and ``solver`` is 'auto', 'newton', 'cyclical', 'lbfgs' or 'proximal'.

        Defaults to 1.0.

    enet_lambda : float, optional

        Penalized weight.
        Only valid when ``multi_class`` is False and ``solver`` is 'auto', 'newton', 'cyclical', 'lbfgs' or 'proximal'.

        Defaults to 0.0.

    tol : float, optional

        Convergence threshold for exiting iterations.

        Only valid when ``multi_class`` is False.

        Defaults to 1.0e-7 when ``solver`` is cyclical, 1.0e-6 otherwise.

    epsilon : float, optional

        Determines the accuracy with which the solution is to
        be found.

        Only valid when ``multi_class`` is False and the ``solver`` is newton or lbfgs.

        Defaults to 1.0e-6 when ``solver`` is newton, 1.0e-5 when ``solver`` is lbfgs.

    thread_ratio : float, optional

        Controls the proportion of available threads to use for fit() method.

        The value range is from 0 to 1, where 0 indicates a single thread, and 1 indicates up to all available threads.

        Values between 0 and 1 will use that percentage of available threads.

        Values outside this range tell PAL to heuristically determine the number of threads to use.

        Defaults to 1.0.

    max_pass_number : int, optional

        The maximum number of passes over the data.

        Only valid when ``multi_class`` is False and ``solver`` is 'stochastic'.

        Defaults to 1.

    sgd_batch_number : int, optional

        The batch number of Stochastic gradient descent.

        Only valid when ``multi_class`` is False and ``solver`` is 'stochastic'.

        Defaults to 1.

    precompute : bool, optional

        Whether to pre-compute the Gram matrix.

        Only valid when ``solver`` is 'cyclical'.

        Defaults to True.

    handle_missing : bool, optional

        Whether to handle missing values.

        Defaults to True.

    categorical_variable : str or list of str, optional

        Specifies INTEGER column(s) in the data that should be treated as categorical.

        By default, string is categorical, while int and double are numerical.

    lbfgs_m : int, optional

        Number of previous updates to keep.

        Only applicable when ``multi_class`` is False and ``solver`` is 'lbfgs'.

        Defaults to 6.

    resampling_method : {'cv', 'stratified_cv', 'bootstrap', 'stratified_bootstrap'}, optional

        The resampling method for model evaluation and parameter selection.

        If no value specified, neither model evaluation nor parameter selection is activated.

    metric : {'accuracy', 'f1_score', 'auc', 'nll'}, optional(deprecated)

        The evaluation metric used for model evaluation/parameter selection.

    evaluation_metric : {'accuracy', 'f1_score', 'auc', 'nll'}, optional

        The evaluation metric used for model evaluation/parameter selection.

    fold_num : int, optional

        The number of folds for cross-validation.

        Mandatory and valid only when ``resampling_method`` is 'cv' or 'stratified_cv'.

    repeat_times : int, optional

        The number of repeat times for resampling.

        Defaults to 1.

    search_strategy : {'grid', 'random'}, optional

        The search method for parameter selection.

    random_search_times : int, optional

        The number of times to randomly select candidate parameters for selection.

        Mandatory and valid when ``search_strategy`` is 'random'.

    random_state : int, optional

        The seed for random generation. 0 indicates using system time as seed.

        Defaults to 0.

    progress_indicator_id : str, optional

        The ID of progress indicator for model evaluation/parameter selection.

        Progress indicator deactivated if no value provided.

    param_values : dict or list of tuples, optional

        Specifies values of specific parameters to be selected.

        Valid only when ``resampling_method`` and ``search_strategy`` are specified.

        Specific parameters can be `enet_lambda`, `enet_alpha`.

        No default value.

    param_range : dict or list of tuples, optional

        Specifies range of specific parameters to be selected.

        Valid only when ``resampling_method`` and ``search_strategy`` are specified.

        Specific parameters can be `enet_lambda`, `enet_alpha`.

        No default value.

    class_map0 : str, optional (deprecated)

        Categorical label to map to 0.

        ``class_map0`` is mandatory when ``label`` column type is VARCHAR or NVARCHAR

        Only valid when ``multi_class`` is False during binary class fit and score.

    class_map1 : str, optional (deprecated)

        Categorical label to map to 1.

        ``class_map1`` is mandatory when ``label`` column type is VARCHAR or NVARCHAR during binary class fit and score.

        Only valid when ``multi_class`` is False.

    Attributes
    ----------

    coef_ : DataFrame

        Values of the coefficients.

    result_ : DataFrame

        Model content.

    optim_param_ : DataFrame

        The optimal parameter set selected via cross-validation.
        Empty if cross-validation is not activated.

    stat_ : DataFrame

        Statistics info for the trained model, structured as follows:

            - 1st column: 'STAT_NAME', NVARCHAR(256)
            - 2nd column: 'STAT_VALUE', NVARCHAR(1000)

    pmml_ : DataFrame

        PMML model. Set to None if no PMML model was requested.

    Examples
    --------

    Training data:

    >>> df.collect()
       V1     V2  V3  CATEGORY
    0   B  2.620   0         1
    1   B  2.875   0         1
    2   A  2.320   1         1
    3   A  3.215   2         0
    4   B  3.440   3         0
    5   B  3.460   0         0
    6   A  3.570   1         0
    7   B  3.190   2         0
    8   A  3.150   3         0
    9   B  3.440   0         0
    10  B  3.440   1         0
    11  A  4.070   3         0
    12  A  3.730   1         0
    13  B  3.780   2         0
    14  B  5.250   2         0
    15  A  5.424   3         0
    16  A  5.345   0         0
    17  B  2.200   1         1
    18  B  1.615   2         1
    19  A  1.835   0         1
    20  B  2.465   3         0
    21  A  3.520   1         0
    22  A  3.435   0         0
    23  B  3.840   2         0
    24  B  3.845   3         0
    25  A  1.935   1         1
    26  B  2.140   0         1
    27  B  1.513   1         1
    28  A  3.170   3         1
    29  B  2.770   0         1
    30  B  3.570   0         1
    31  A  2.780   3         1

    Create LogisticRegression instance and call fit:

    >>> lr = linear_model.LogisticRegression(solver='newton',
    ...                                      thread_ratio=0.1, max_iter=1000,
    ...                                      pmml_export='single-row',
    ...                                      stat_inf=True, tol=0.000001)
    >>> lr.fit(data=df, features=['V1', 'V2', 'V3'],
    ...        label='CATEGORY', categorical_variable=['V3'])
    >>> lr.coef_.collect()
                                           VARIABLE_NAME  COEFFICIENT
    0                                  __PAL_INTERCEPT__    17.044785
    1                                 V1__PAL_DELIMIT__A     0.000000
    2                                 V1__PAL_DELIMIT__B    -1.464903
    3                                                 V2    -4.819740
    4                                 V3__PAL_DELIMIT__0     0.000000
    5                                 V3__PAL_DELIMIT__1    -2.794139
    6                                 V3__PAL_DELIMIT__2    -4.807858
    7                                 V3__PAL_DELIMIT__3    -2.780918
    8  {"CONTENT":"{\"impute_model\":{\"column_statis...          NaN
    >>> pred_df.collect()
        ID V1     V2  V3
    0    0  B  2.620   0
    1    1  B  2.875   0
    2    2  A  2.320   1
    3    3  A  3.215   2
    4    4  B  3.440   3
    5    5  B  3.460   0
    6    6  A  3.570   1
    7    7  B  3.190   2
    8    8  A  3.150   3
    9    9  B  3.440   0
    10  10  B  3.440   1
    11  11  A  4.070   3
    12  12  A  3.730   1
    13  13  B  3.780   2
    14  14  B  5.250   2
    15  15  A  5.424   3
    16  16  A  5.345   0
    17  17  B  2.200   1

    Call predict():

    >>> result = lgr.predict(data=pred_df,
    ...                      key='ID',
    ...                      categorical_variable=['V3'],
    ...                      thread_ratio=0.1)
    >>> result.collect()
        ID CLASS   PROBABILITY
    0    0     1  9.503618e-01
    1    1     1  8.485210e-01
    2    2     1  9.555861e-01
    3    3     0  3.701858e-02
    4    4     0  2.229129e-02
    5    5     0  2.503962e-01
    6    6     0  4.945832e-02
    7    7     0  9.922085e-03
    8    8     0  2.852859e-01
    9    9     0  2.689207e-01
    10  10     0  2.200498e-02
    11  11     0  4.713726e-03
    12  12     0  2.349803e-02
    13  13     0  5.830425e-04
    14  14     0  4.886177e-07
    15  15     0  6.938072e-06
    16  16     0  1.637820e-04
    17  17     1  8.986435e-01

    Input data for score():

    >>> df_score.collect()
        ID V1     V2  V3  CATEGORY
    0    0  B  2.620   0         1
    1    1  B  2.875   0         1
    2    2  A  2.320   1         1
    3    3  A  3.215   2         0
    4    4  B  3.440   3         0
    5    5  B  3.460   0         0
    6    6  A  3.570   1         1
    7    7  B  3.190   2         0
    8    8  A  3.150   3         0
    9    9  B  3.440   0         0
    10  10  B  3.440   1         0
    11  11  A  4.070   3         0
    12  12  A  3.730   1         0
    13  13  B  3.780   2         0
    14  14  B  5.250   2         0
    15  15  A  5.424   3         0
    16  16  A  5.345   0         0
    17  17  B  2.200   1         1

    Call score():

    >>> lgr.score(data=df_score,
    ...           key='ID',
    ...           categorical_variable=['V3'],
    ...           thread_ratio=0.1)
    0.944444
    """
    solver_map = {'auto':-1, 'newton':0, 'cyclical':2, 'lbfgs':3, 'stochastic':4, 'proximal':6}
    multi_solver_map = {'auto': None, 'lbfgs': 0, 'cyclical': 1}
    pmml_map_multi = {'no': 0, 'multi-row': 1}
    pmml_map_binary = {'no': 0, 'single-row': 1, 'multi-row': 2}
    resampling_method_list = ('cv', 'stratified_cv', 'bootstrap', 'stratified_bootstrap')
    valid_metric_map = {'accuracy':'ACCURACY', 'f1_score':'F1_SCORE', 'auc':'AUC', 'nll':'NLL'}
    values_list = {"enet_alpha": "ENET_ALPHA", "enet_lambda": "ENET_LAMBDA"}
    #pylint:disable=too-many-arguments, too-many-branches, too-many-statements
    def __init__(self,
                 multi_class=False,
                 max_iter=None,
                 pmml_export=None,
                 categorical_variable=None,
                 standardize=True,
                 stat_inf=False,
                 solver=None,
                 enet_alpha=None,
                 enet_lambda=None,
                 tol=None,
                 epsilon=None,
                 thread_ratio=None,
                 max_pass_number=None,
                 sgd_batch_number=None,
                 precompute=None,#adding new parameters
                 handle_missing=None,
                 resampling_method=None,
                 metric=None,
                 evaluation_metric=None,
                 fold_num=None,
                 repeat_times=None,
                 search_strategy=None,
                 random_search_times=None,
                 random_state=None,
                 timeout=None,
                 lbfgs_m=None,
                 class_map0=None,
                 class_map1=None,
                 progress_indicator_id=None,
                 param_values=None,
                 param_range=None):
        #pylint:disable=too-many-locals
        super(LogisticRegression, self).__init__()
        setattr(self, 'hanaml_parameters', pal_param_register())
        self.multi_class = self._arg('multi_class', multi_class, bool)
        self.max_iter = self._arg('max_iter', max_iter, int)
        if self.multi_class:
            pmml_map = self.pmml_map_multi
            solver_map = self.multi_solver_map
        else:
            pmml_map = self.pmml_map_binary
            solver_map = self.solver_map
            #solver = 'newton' if solver is None else solver#maybe something wrong here
        self.pmml_export = self._arg('pmml_export', pmml_export, pmml_map)
        if isinstance(categorical_variable, str):
            categorical_variable = [categorical_variable]
        self.categorical_variable = self._arg('categorical_variable', categorical_variable,
                                              ListOfStrings)
        self.standardize = self._arg('standardize', standardize, bool)
        self.stat_inf = self._arg('stat_inf', stat_inf, bool)
        binary_params = {'solver': solver, 'enet_alpha': enet_alpha, 'enet_lambda': enet_lambda, 'tol': tol,
                         'epsilon': epsilon, 'thread_ratio': thread_ratio,
                         'max_pass_number': max_pass_number, 'sgd_batch_number': sgd_batch_number,
                         'lbfgs_m': lbfgs_m, 'class_map0': class_map0, 'class_map1': class_map1}
        self._check_params_for_binary(binary_params)
        self.solver = self._arg('solver', solver, solver_map)
        self.enet_alpha = self._arg('enet_alpha', enet_alpha, float)
        self.enet_lambda = self._arg('enet_lambda', enet_lambda, float)
        if multi_class is not True:
            if self.solver not in (-1, 0, 2, 3, 6, None) and enet_lambda is not None:
                msg = ('Parameter enet_lambda is only applicable when solver is auto, ' +
                       'newton, cyclical, lbfgs or proximal.')
                logger.error(msg)
                raise ValueError(msg)
            if self.solver not in (-1, 0, 2, 3, 6, None) and enet_alpha is not None:
                msg = ('Parameter enet_alpha is only applicable when solver ' +
                       'is auto, newton, cyclical, lbfgs or proximal.')
                logger.error(msg)
                raise ValueError(msg)
        self.tol = self._arg('tol', tol, float)#Corresponds to EXIT_THRESHOLD
        self.epsilon = self._arg('epsilon', epsilon, float)
        self.thread_ratio = self._arg('thread_ratio', thread_ratio, float)
        if self.solver != 4 and max_pass_number is not None:
            msg = 'Parameter max_pass_number is only applicable when solver is stochastic.'
            logger.error(msg)
            raise ValueError(msg)
        self.max_pass_number = self._arg('max_pass_number', max_pass_number, int)
        if self.solver != 4 and sgd_batch_number is not None:
            msg = 'Parameter sgd_batch_number is only applicable when solver is stochastic.'
            logger.error(msg)
            raise ValueError(msg)
        self.sgd_batch_number = self._arg('sgd_batch_number', sgd_batch_number, int)
        if self.solver != 3 and lbfgs_m is not None:
            msg = ('Parameter lbfgs_m will be applicable '+
                   'only if method is lbfgs.')
            logger.error(msg)
            raise ValueError(msg)
        self.precompute = self._arg('precompute', precompute, bool)
        self.handle_missing = self._arg('handle_missing', handle_missing, bool)
        self.resampling_method = self._arg('resampling_method', resampling_method, str)
        if self.resampling_method is not None:
            if self.resampling_method not in self.resampling_method_list:
                msg = ("Resampling method '{}' is not supported ".format(self.resampling_method) +
                       "for model evaluation or parameter selection.")
                logger.error(msg)
                raise ValueError(msg)
        self.metric = self._arg('metric', metric, self.valid_metric_map)
        self.evaluation_metric = self._arg('evaluation_metric', evaluation_metric,
                                           self.valid_metric_map)
        self.fold_num = self._arg('fold_num', fold_num, int)
        self.repeat_times = self._arg('repeat_times', repeat_times, int)
        self.search_strategy = self._arg('search_stategy', search_strategy, str)
        if self.search_strategy is not None:
            if self.search_strategy not in ('grid', 'random'):
                msg = ("Search strategy '{}' is not available for ".format(self.search_strategy)+
                       "parameter selection.")
                logger.error(msg)
                raise ValueError(msg)
        self.random_search_times = self._arg('random_search_times', random_search_times, int)
        self.random_state = self._arg('random_state', random_state, int)
        self.timeout = self._arg('timeout', timeout, int)
        self.lbfgs_m = self._arg('lbfgs_m', lbfgs_m, int)
        self.class_map0 = self._arg('class_map0', class_map0, str)
        self.class_map1 = self._arg('class_map1', class_map1, str)
        self.progress_indicator_id = self._arg('progress_indicator_id', progress_indicator_id, str)
        if isinstance(param_range, dict):
            param_range = [(x, param_range[x]) for x in param_range]
        if isinstance(param_values, dict):
            param_values = [(x, param_values[x]) for x in param_values]
        self.param_range = self._arg('param_range', param_range, ListOfTuples)
        self.param_values = self._arg('param_values', param_values, ListOfTuples)
        if self.search_strategy is None:
            if self.param_values is not None:
                msg = ("`param_values` can only be specified "+
                       "when `search_strategy` is enabled.")
                logger.error(msg)
                raise ValueError(msg)
            if self.param_range is not None:
                msg = ("`param_range` can only be specified "+
                       "when `search_strategy` is enabled.")
                logger.error(msg)
                raise ValueError(msg)
        if self.search_strategy is not None:
            set_param_list = list()
            if self.enet_lambda is not None:
                set_param_list.append("enet_lambda")
            if self.enet_alpha is not None:
                set_param_list.append("enet_alpha")
            if self.param_values is not None:
                for x in self.param_values:#pylint:disable=invalid-name
                    if len(x) != 2:#pylint:disable=bad-option-value
                        msg = ("Each tuple that specifies the values of a parameter should"+
                               " contain exactly 2 elements: 1st is parameter name,"+
                               " 2nd is a list of valid values.")
                        logger.error(msg)
                        raise ValueError(msg)
                    if x[0] not in self.values_list:
                        msg = ("Specifying the values of `{}` for ".format(x[0])+
                               "parameter selection is invalid.")
                        logger.error(msg)
                        raise ValueError(msg)
                    if x[0] in set_param_list:
                        msg = ("Parameter `{}` is invalid for ".format(x[0])+
                               "re-specification in parameter selection.")
                        logger.error(msg)
                        raise ValueError(msg)
                    if self.solver not in (0, 2, 3, 6):
                        msg = ("Parameter `{}` is invalid when ".format(x[0])+
                               "`solver` is not set as newton, cyclical, lbfgs or proximal.")
                        logger.error(msg)
                        raise ValueError(msg)
                    if not (isinstance(x[1], list) and all(isinstance(t, (int, float)) for t in x[1])):#pylint:disable=line-too-long
                        msg = "Valid values of `{}` must be a list of numericals.".format(x[0])
                        logger.error(msg)
                        raise TypeError(msg)
                    set_param_list.append(x[0])

            if self.param_range is not None:
                rsz = [3] if self.search_strategy == 'grid' else [2, 3]
                for x in self.param_range:#pylint:disable=invalid-name
                    if len(x) != 2:#pylint:disable=bad-option-value
                        msg = ("Each tuple that specifies the range of a parameter should contain"+
                               " exactly 2 elements: 1st is parameter name, 2nd is value range.")
                        logger.error(msg)
                        raise ValueError(msg)
                    if x[0] not in self.values_list:
                        msg = ("Specifying the values of `{}` for ".format(x[0])+
                               "parameter selection is invalid.")
                        logger.error(msg)
                        raise ValueError(msg)
                    if x[0] in set_param_list:
                        msg = ("Parameter `{}` is invalid for ".format(x[0])+
                               "re-specification in parameter selection.")
                        logger.error(msg)
                        raise ValueError(msg)
                    if self.solver not in (0, 2, 3, 6):
                        msg = ("Parameter `{}` is invalid when ".format(x[0])+
                               "`solver` is not set as newton, cyclical, lbfgs or proximal.")
                        logger.error(msg)
                        raise ValueError(msg)
                    if not (isinstance(x[1], list) and len(x[1]) in rsz and all(isinstance(t, (int, float)) for t in x[1])):#pylint:disable=line-too-long
                        msg = ("The provided `{}` is either not ".format(x[0])+
                               "a list of numericals, or it contains wrong number of values.")
                        logger.error(msg)
                        raise TypeError(msg)
        self.label_type = None

    #pylint:disable=too-many-locals, too-many-statements, invalid-name
    @trace_sql
    def fit(self,
            data,
            key=None,
            features=None,
            label=None,
            categorical_variable=None,
            class_map0=None,
            class_map1=None):
        r"""
        Fit the LR model when given training dataset.

        Parameters
        ----------

        data : DataFrame

            DataFrame containing the data.

        key : str, optional

            Name of the ID column.

            If ``key`` is not provided, it is assumed that the input has no ID column.

        features : list of str, optional

            Names of the feature columns.

            If ``features`` is not provided, it defaults to all non-ID,
            non-label columns.

        label : str, optional

            Name of the label column.

            If ``label`` is not provided, it defaults to the last column.

        categorical_variable : str or list of str, optional

            Specifies INTEGER column(s) that shoud be treated as categorical.

            Otherwise All INTEGER columns are treated as numerical.

        class_map0 : str, optional

            Categorical label to map to 0.

            ``class_map0`` is mandatory when ``label`` column type is VARCHAR or NVARCHAR during binary class fit and score.

            Only valid if ``multi_class`` is not set to True when initializing the class instance.

        class_map1 : str, optional

            Categorical label to map to 1.

            ``class_map1`` is mandatory when ``label`` column type is VARCHAR or NVARCHAR during binary class fit and score.

            Only valid if ``multi_class`` is not set to True when initializing the class instance.

        """
        conn = data.connection_context
        require_pal_usable(conn)
        key = self._arg('key', key, str)
        features = self._arg('features', features, ListOfStrings)
        label = self._arg('label', label, str)
        class_map0 = self._arg('class_map0', class_map0, str)
        class_map1 = self._arg('class_map1', class_map1, str)
        if class_map0 is not None:
            self.class_map0 = class_map0
        if class_map1 is not None:
            self.class_map1 = class_map1
        if isinstance(categorical_variable, str):
            categorical_variable = [categorical_variable]
        categorical_variable = self._arg('categorical_variable', categorical_variable, ListOfStrings)
        if categorical_variable is not None:
            self.categorical_variable = categorical_variable
        cols = data.columns
        if key is not None:
            cols.remove(key)
        if label is None:
            label = cols[-1]
        self.label_type = data.dtypes([label])[0][1]
        cols.remove(label)
        if features is None:
            features = cols
        if not self.multi_class:
            self._check_label_sql_type(data, label)
        used_cols = [col for col in itertools.chain([key], features, [label])
                     if col is not None]
        data_ = data[used_cols]
        unique_id = str(uuid.uuid1()).replace('-', '_').upper()
        outputs = ['#LR_{}_TBL_{}_{}'.format(name, self.id, unique_id)
                   for name in ['RESULT', 'PMML', 'STAT', 'OPTIM']]
        result_tbl, pmml_tbl, stat_tbl, optim_tbl = outputs
        param_rows = [('MAX_ITERATION', self.max_iter, None, None),
                      ('PMML_EXPORT', self.pmml_export, None, None),
                      ('HAS_ID', key is not None, None, None),
                      ('STANDARDIZE', self.standardize, None, None),
                      ('STAT_INF', self.stat_inf, None, None),
                      ('ENET_ALPHA', None, self.enet_alpha, None),
                      ('ENET_LAMBDA', None, self.enet_lambda, None),
                      ('EXIT_THRESHOLD', None, self.tol, None),
                      ('METHOD', self.solver, None, None),
                      ('RESAMPLING_METHOD', None, None, self.resampling_method),
                      ('EVALUATION_METRIC', None, None,
                       self.metric if self.evaluation_metric is None else self.evaluation_metric),
                      ('FOLD_NUM', self.fold_num, None, None),
                      ('REPEAT_TIMES', self.repeat_times, None, None),
                      ('PARAM_SEARCH_STRATEGY', None, None, self.search_strategy),
                      ('RANDOM_SEARCH_TIMES', None, None, self.random_search_times),
                      ('SEED', self.random_state, None, None),
                      ('TIMEOUT', self.timeout, None, None),
                      ('PROGRESS_INDICATOR_ID', None, None, self.progress_indicator_id)]
        if self.param_values is not None:
            for x in self.param_values:#pylint:disable=invalid-name
                values = str(x[1]).replace('[', '{').replace(']', '}')
                param_rows.extend([(self.values_list[x[0]]+"_VALUES",
                                    None, None, values)])
        if self.param_range is not None:
            for x in self.param_range:#pylint:disable=invalid-name
                range_ = str(x[1])
                if len(x[1]) == 2 and self.search_strategy == 'random':
                    range_ = range_.replace(',', ',,')
                param_rows.extend([(self.values_list[x[0]]+"_RANGE",
                                    None, None, range_)])
        if self.categorical_variable is not None:
            param_rows.extend([('CATEGORICAL_VARIABLE', None, None, col)
                               for col in self.categorical_variable])
        if self.multi_class:
            proc_name = 'PAL_MULTICLASS_LOGISTIC_REGRESSION'
            coef_list = ['VARIABLE_NAME', 'CLASS', 'COEFFICIENT']
        else:
            proc_name = "PAL_LOGISTIC_REGRESSION"
            coef_list = ['VARIABLE_NAME', 'COEFFICIENT']
            param_rows.extend([('EPSILON', None, self.epsilon, None),
                               ('THREAD_RATIO', None, self.thread_ratio, None),
                               ('MAX_PASS_NUMBER', self.max_pass_number, None, None),
                               ('SGD_BATCH_NUMBER', self.sgd_batch_number, None, None),
                               ('PRECOMPUTE', self.precompute, None, None),
                               ('HANDLE_MISSING', self.handle_missing, None, None),
                               ('LBFGS_M', self.lbfgs_m, None, None)])
            if self.label_type in ('VARCHAR', 'NVARCHAR'):
                param_rows.extend([('CLASS_MAP1', None, None, self.class_map1),
                                   ('CLASS_MAP0', None, None, self.class_map0)])
        try:
            call_pal_auto(conn,
                          proc_name,
                          data_,
                          ParameterTable().with_data(param_rows),
                          result_tbl,
                          pmml_tbl,
                          stat_tbl,
                          optim_tbl)
        except dbapi.Error as db_err:
            logger.exception(str(db_err))
            try_drop(conn, result_tbl)
            try_drop(conn, pmml_tbl)
            try_drop(conn, stat_tbl)
            try_drop(conn, optim_tbl)
            raise
        #pylint:disable=attribute-defined-outside-init
        self.result_ = conn.table(result_tbl)
        self.coef_ = self.result_.select(coef_list)
        self.pmml_ = conn.table(pmml_tbl) if self.pmml_export else None
        self.optim_param_ = conn.table(optim_tbl)
        self.stat_ = conn.table(stat_tbl)
        self.model_ = self.result_

    @trace_sql
    def predict(self,
                data,
                key,
                features=None,
                categorical_variable=None,
                thread_ratio=None,
                class_map0=None,
                class_map1=None,
                verbose=False):
        #pylint:disable=too-many-locals
        r"""
        Predict with the dataset using the trained model.

        Parameters
        ----------

        data : DataFrame

            DataFrame containing the data.

        key : str

            Name of the ID column.

        features : list of str, optional

            Names of the feature columns.

            If ``features`` is not provided, it defaults to all non-ID columns.

        verbose : bool, optional

            If true, output scoring probabilities for each class.

            It is only applicable for multi-class case.

            Defaults to False.

        categorical_variable : str or list of str, optional (deprecated)

            Specifies INTEGER column(s) that shoud be treated as categorical.

            Otherwise all integer columns are treated as numerical.

            Mandatory if training data of the prediction model contains such
            data columns.

        thread_ratio : float, optional

            Controls the proportion of available threads to use.

            The value range is from 0 to 1, where 0 means only using 1 thread, and 1 means using at most all the currently available threads.

            Values outside this range tell pal to heuristically determine the number of threads to use.

            Defaults to 0.

        class_map0 : str, optional

            Categorical label to map to 0.

            ``class_map0`` is mandatory when ``label`` column type is varchar or nvarchar during binary class fit and score.

            Only valid if ``multi_class`` is not set to True when initializing the class instance.

        class_map1 : str, optional

            Categorical label to map to 1.

            ``class_map1`` is mandatory when ``label`` column type is varchar or nvarchar during binary class fit and score.

            Only valid if ``multi_class`` is not set to True when initializing the class instance.

        Returns
        -------

        DataFrame
            Predicted result, structured as follows:

            - 1: ID column, with edicted class name.
            - 2: PROBABILITY, type DOUBLE

                - multi-class: probability of being predicted as the predicted class.
                - binary-class: probability of being predicted as the positive class.

           .. note ::

                predict() will pass the ``pmml_`` table to PAL as the model representation if there is a ``pmml_`` table, or the ``result_`` table otherwise.
        """
        conn = data.connection_context
        require_pal_usable(conn)
        if getattr(self, 'pmml_', None) is not None:
            model = self.pmml_
        elif hasattr(self, 'model_'):
            model = self.model_
        else:
            raise FitIncompleteError("Model not initialized. Perform a fit first.")
        key = self._arg('arg', key, str, True)
        features = self._arg('features', features, ListOfStrings)
        verbose = self._arg('verbose', verbose, bool)
        thread_ratio = self._arg('thread_ratio', thread_ratio, float)
        unique_id = str(uuid.uuid1()).replace('-', '_').upper()

        cols = data.columns
        cols.remove(key)
        if features is None:
            features = cols
        data_ = data[[key] + features]

        result_tbl = '#LR_PREDICT_RESULT_TBL_{}_{}'.format(self.id, unique_id)
        if self.multi_class:
            param_array = [('VERBOSE_OUTPUT', verbose, None, None)]
            proc_name = 'PAL_MULTICLASS_LOGISTIC_REGRESSION_PREDICT'
        else:
            param_array = [('THREAD_RATIO', None, thread_ratio, None)]
            if categorical_variable is not None:
                param_array.extend([('CATEGORICAL_VARIABLE', None, None, variable)
                                    for variable in categorical_variable])
            if self.categorical_variable is not None:
                param_array.extend([('CATEGORICAL_VARIABLE', None, None, variable)
                                    for variable in self.categorical_variable])
            if class_map0 is not None:
                param_array.extend([('CLASS_MAP0', None, None, class_map0),
                                    ('CLASS_MAP1', None, None, class_map1)])
            else:
                param_array.extend([('CLASS_MAP0', None, None, self.class_map0),
                                    ('CLASS_MAP1', None, None, self.class_map1)])
            proc_name = 'PAL_LOGISTIC_REGRESSION_PREDICT'
        try:
            call_pal_auto(conn,
                          proc_name,
                          data_,
                          model,
                          ParameterTable().with_data(param_array),
                          result_tbl)
        except dbapi.Error as db_err:
            logger.exception(str(db_err))
            try_drop(conn, result_tbl)
            raise
        result = conn.table(result_tbl)
        if result.has('SCORE'):
            result = result.rename_columns({'SCORE': 'PROBABILITY'})
        result = result[[key, 'CLASS', 'PROBABILITY']]
        return result

    def score(self,
              data,
              key,
              features=None,
              label=None,
              categorical_variable=None,
              thread_ratio=None,
              class_map0=None,
              class_map1=None):
        r"""
        Return the mean accuracy on the given test data and labels.

        Parameters
        ----------

        data : DataFrame

            DataFrame containing the data.

        key : str

            Name of the ID column.

        features : list of str, optional

            Names of the feature columns.

            If ``features`` is not provided, it defaults to all non-ID,
            non-label columns.

        label : str, optional

            Name of the label column.

            If ``label`` is not provided, it defaults to the last column.

        categorical_variable : str or list of str, optional (deprecated)

            Specifies INTEGER columns that shoud be treated as categorical, otherwise all integer columns are treated as numerical.

            Mandatory if training data of the prediction model contains such data columns.

        thread_ratio : float, optional

            Controls the proportion of available threads to use.

            The value range is from 0 to 1, where 0 means only using 1 thread, and 1 means using at most all the currently available threads.

            values outside this range tell pal to heuristically determine the number of threads to use.

            Defaults to 0.

        class_map0 : str, optional

            Categorical label to map to 0.

            ``class_map0`` is mandatory when ``label`` column type is varchar or nvarchar during binary class fit and score.

            Only valid if ``multi_class`` is not set to True when initializing the class instance.

        class_map1 : str, optional

            Categorical label to map to 1.

            ``class_map1`` is mandatory when ``label`` column type is varchar or nvarchar during binary class fit and score.

            Only valid if ``multi_class`` is not set to True when initializing the class instance.

        Returns
        -------
        float

            Scalar accuracy value after comparing the predicted label
            and original label.
        """

        key = self._arg('key', key, str, True)
        features = self._arg('features', features, ListOfStrings)
        label = self._arg('label', label, str)
        if isinstance(categorical_variable, str):
            categorical_variable = [categorical_variable]
        categorical_variable = self._arg('categorical_variable',
                                         categorical_variable,
                                         ListOfStrings)
        thread_ratio = self._arg('thread_ratio', thread_ratio, float)
        cols = data.columns
        cols.remove(key)
        if label is None:
            label = cols[-1]
        cols.remove(label)
        if features is None:
            features = cols
        if not self.multi_class:
            self._check_label_sql_type(data, label)
        prediction = self.predict(data=data, key=key,
                                  features=features,
                                  categorical_variable=categorical_variable,
                                  thread_ratio=thread_ratio,
                                  class_map0=class_map0,
                                  class_map1=class_map1)
        prediction = prediction.select(key, 'CLASS').rename_columns(['ID_P', 'PREDICTION'])
        actual = data.select(key, label).rename_columns(['ID_A', 'ACTUAL'])
        joined = actual.join(prediction, 'ID_P=ID_A').select('ACTUAL', 'PREDICTION')
        return metrics.accuracy_score(joined,
                                      label_true='ACTUAL',
                                      label_pred='PREDICTION')

    def _check_label_sql_type(self, data, label):
        label_sql_type = parse_one_dtype(data.dtypes([label])[0])[1]
        if label_sql_type.startswith("NVARCHAR") or label_sql_type.startswith("VARCHAR"):
            if self.class_map0 is None or self.class_map1 is None:
                msg = ("class_map0 and class_map1 are mandatory when `label` column type " +
                       "is VARCHAR or NVARCHAR.")
                logger.error(msg)
                raise ValueError(msg)

    def _check_params_for_binary(self, params):
        for name in params:
            msg = 'Parameter {} is only applicable for binary classification.'.format(name)
            if params[name] is not None and self.multi_class:
                logger.error(msg)
                raise ValueError(msg)
