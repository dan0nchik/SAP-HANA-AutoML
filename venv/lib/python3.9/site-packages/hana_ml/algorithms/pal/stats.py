"""
This module contains Python wrappers for statistics algorithms.

The following functions are available:

    * :func:`chi_squared_goodness_of_fit`
    * :func:`chi_squared_independence`
    * :func:`ttest_1samp`
    * :func:`ttest_ind`
    * :func:`ttest_paired`
    * :func:`f_oneway`
    * :func:`f_oneway_repeated`
    * :func:`univariate_analysis`
    * :func:`covariance_matrix`
    * :func:`pearsonr_matrix`
    * :func:`iqr`
    * :func:`wilcoxon`
    * :func:`median_test_1samp`
    * :func:`grubbs_test`
    * :func:`entropy`
    * :func:`condition_index`
    * :func:`cdf`
    * :func:`ftest_equal_var`
    * :func:`factor_analysis`
    * :func:`kaplan_meier_survival_analysis`
    * :func:`quantile`
    * :func:`distribution_fit`
"""

#pylint:disable=too-many-lines, line-too-long
import logging
import uuid

from hdbcli import dbapi
from .pal_base import (
    Table,
    NVARCHAR,
    create,
    ParameterTable,
    arg,
    try_drop,
    ListOfStrings,
    require_pal_usable,
    call_pal_auto
)

logger = logging.getLogger(__name__)#pylint: disable=invalid-name

def chi_squared_goodness_of_fit(data, key, observed_data=None, expected_freq=None):
    """
    Perform the chi-squared goodness-of fit test to tell whether or not an \
    observed distribution differs from an expected chi-squared distribution.

    Parameters
    ----------

    data : DataFrame
        DataFrame containing the data.

    key : str
        Name of the ID column.

    observed_data : str, optional
        Name of column for counts of actual observations belonging to each
        category.

        If not given, it defaults to the first non-ID column.

    expected_freq : str, optional
        Name of the expected frequency column.

        If not given, it defaults to the second non-ID column.

    Returns
    -------
    DataFrame

        Comparsion between the actual counts and the expected counts, structured as follows:

            - ID column, with same name and type as ``data``'s ID column.
            - Observed data column, with same name as ``data``'s observed_data
              column, but always with type DOUBLE.
            - EXPECTED, type DOUBLE, expected count in each category.
            - RESIDUAL, type DOUBLE, the difference between the observed
              counts and the expected counts.

        Statistical outputs, including the calculated chi-squared value, degrees of freedom and p-value, structured as follows:

            - STAT_NAME, type NVARCHAR(100), name of statistics.
            - STAT_VALUE, type DOUBLE, value of statistics.

    Examples
    --------
    Data to test:

    >>> df.collect()
       ID  OBSERVED    P
    0   0     519.0  0.3
    1   1     364.0  0.2
    2   2     363.0  0.2
    3   3     200.0  0.1
    4   4     212.0  0.1
    5   5     193.0  0.1

    Perform the function:

    >>> res, stat = chi_squared_goodness_of_fit(data=df, 'ID')
    >>> res.collect()
       ID  OBSERVED  EXPECTED  RESIDUAL
    0   0     519.0     555.3     -36.3
    1   1     364.0     370.2      -6.2
    2   2     363.0     370.2      -7.2
    3   3     200.0     185.1      14.9
    4   4     212.0     185.1      26.9
    5   5     193.0     185.1       7.9
    >>> stat.collect()
               STAT_NAME  STAT_VALUE
    0  Chi-squared Value    8.062669
    1  degree of freedom    5.000000
    2            p-value    0.152815
    """
    conn = data.connection_context
    require_pal_usable(conn)

    # SQLTRACE
    conn.sql_tracer.set_sql_trace(None, 'Stats', 'chi_squared_goodness_of_fit')

    unique_id = str(uuid.uuid1()).replace('-', '_').upper()

    key = arg('key', key, str, True)
    cols = data.columns
    cols.remove(key)

    observed_data = arg('observed_data', observed_data, str)
    if observed_data is None:
        observed_data = cols[0]
    cols.remove(observed_data)
    expected_freq = arg('expected_freq', expected_freq, str)
    if expected_freq is None:
        expected_freq = cols[0]
    data_ = data[[key, observed_data, expected_freq]]

    tables = result_tbl, stat_tbl = ["#CHI_GOODNESS_{}_{}".format(name, unique_id)
                                     for name in ['RESULT', 'STATISTICS']]

    try:
        call_pal_auto(conn,
                      'PAL_CHISQUARED_GOF_TEST',
                      data_,
                      *tables)
        return conn.table(result_tbl), conn.table(stat_tbl)
    except dbapi.Error as db_err:
        logger.exception(str(db_err))
        try_drop(conn, tables)
        raise

def chi_squared_independence(data, key, observed_data=None,#pylint: disable=too-many-locals,
                             correction=False):
    """
    Perform the chi-squared test of independence to tell whether observations of \
    two variables are independent from each other.

    Parameters
    ----------

    data : DataFrame
        DataFrame containing the data.

    key : str
        Name of the ID column.

    observed_data : list of str, optional
        Names of the observed data columns.

        If not given, it defaults to all non-ID columns.

    correction : bool, optional
        If True, and the degrees of freedom is 1, apply
        Yates's correction for continuity.

        The effect of the correction is to adjust each observed value by
        0.5 towards the corresponding expected value.

        Defaults to False.

    Returns
    -------
    DataFrame
        The expected count table, structured as follows:

            - ID column, with same name and type as ``data``'s ID column.
            - Expected count columns, named by prepending ``Expected_`` to
              each ``observed_data`` column name, type DOUBLE. There will be as
              many columns here as there are ``observed_data`` columns.

        Statistical outputs, including the calculated chi-squared value, degrees of freedom and p-value, structured as follows:

            - STAT_NAME, type NVARCHAR(100), name of statistics.
            - STAT_VALUE, type DOUBLE, value of statistics.

    Examples
    --------
    Data to test:

    >>> df.collect()
           ID  X1    X2  X3    X4
    0    male  25  23.0  11  14.0
    1  female  41  20.0  18   6.0

    Perform the function:

    >>> res, stats = chi_squared_independence(data=df, 'ID')
    >>> res.collect()
           ID  EXPECTED_X1  EXPECTED_X2  EXPECTED_X3  EXPECTED_X4
    0    male    30.493671    19.867089    13.398734     9.240506
    1  female    35.506329    23.132911    15.601266    10.759494
    >>> stats.collect()
               STAT_NAME  STAT_VALUE
    0  Chi-squared Value    8.113152
    1  degree of freedom    3.000000
    2            p-value    0.043730
    """
    conn = data.connection_context
    require_pal_usable(conn)

    # SQLTRACE
    conn.sql_tracer.set_sql_trace(None, 'Stats', 'chi_squared_independence')

    unique_id = str(uuid.uuid1()).replace('-', '_').upper()
    key = arg('key', key, str, True)
    observed_data = arg('observed_data', observed_data, ListOfStrings)
    cols = data.columns
    cols.remove(key)
    if observed_data is None:
        observed_data = cols

    data_ = data[[key] + observed_data]
    tables = ["#CHI_INDEPENDENCE_{}_{}".format(name, unique_id)
              for name in ['RESULT', 'STATS']]
    result_tbl, stat_tbl = tables
    param_array = [('CORRECTION_TYPE', correction, None, None)]

    try:
        call_pal_auto(conn,
                      'PAL_CHISQUARED_IND_TEST',
                      data_,
                      ParameterTable().with_data(param_array),
                      *tables)
        return conn.table(result_tbl), conn.table(stat_tbl)
    except dbapi.Error as db_err:
        logger.exception(str(db_err))
        try_drop(conn, tables)
        raise

def _ttest_base(data, mu=None, test_type=None, paired=None, var_equal=None,#pylint: disable=too-many-arguments, too-many-locals, invalid-name
                conf_level=None):
    conn = data.connection_context
    unique_id = str(uuid.uuid1()).replace('-', '_').upper()
    mu_final = arg('mu', mu, float)
    test_type_map = {'two_sides':0, 'less':1, 'greater':2}
    test_type_final = arg('test_type', test_type, test_type_map)
    paired_final = arg('paired', paired, bool)
    var_equal_final = arg('var_equal', var_equal, bool)
    conf_level_final = arg('conf_level', conf_level, float)
    stat_tbl = '#TTEST_STAT_TBL_{}'.format(unique_id)
    param_array = [('TEST_TYPE', test_type_final, None, None),
                   ('MU', None, mu_final, None),
                   ('PAIRED', paired_final, None, None),
                   ('VAR_EQUAL', var_equal_final, None, None),
                   ('CONF_LEVEL', None, conf_level_final, None)]
    try:
        call_pal_auto(conn,
                      'PAL_T_TEST',
                      data,
                      ParameterTable().with_data(param_array),
                      stat_tbl)
    except dbapi.Error as db_err:
        logger.exception(str(db_err))
        try_drop(conn, stat_tbl)
        raise
    return conn.table(stat_tbl)

def ttest_1samp(data, col=None, mu=0, test_type='two_sides', conf_level=0.95):#pylint: disable=too-many-arguments, invalid-name
    """
    Perform the t-test to determine whether a sample of observations
    could have been generated by a process with a specific mean.

    Parameters
    ----------

    data : DataFrame
        DataFrame containing the data.

    col : str, optional
        Name of the column for sample.

        If not given, it defaults to the first column.

    mu : float, optional
        Hypothesized mean of the population underlying the sample.

        Defaults to 0.

    test_type : {'two_sides', 'less', 'greater'}, optional
        The alternative hypothesis type.

        Defaults to 'two_sides'.

    conf_level : float, optional
        Confidence level for alternative hypothesis confidence interval.

        Defaults to 0.95.

    Returns
    -------

    DataFrame
        Statistics results.

    Examples
    --------
    Original data:

    >>> df.collect()
        X1
    0  1.0
    1  2.0
    2  4.0
    3  7.0
    4  3.0

    Perform One Sample T-Test:

    >>> ttest_1samp(data=df).collect()
               STAT_NAME  STAT_VALUE
    0            t-value    3.302372
    1  degree of freedom    4.000000
    2            p-value    0.029867
    3      _PAL_MEAN_X1_    3.400000
    4   confidence level    0.950000
    5         lowerLimit    0.541475
    6         upperLimit    6.258525
    """
    conn = data.connection_context
    require_pal_usable(conn)
    col = arg('col', col, str)
    if col is None:
        col = data.columns[0]
    data_ = data.select(col)

    # SQLTRACE
    conn.sql_tracer.set_sql_trace(None, 'Stats', 'ttest_1samp')

    return _ttest_base(data_, mu=mu, test_type=test_type, conf_level=conf_level)

def ttest_ind(data, col1=None, col2=None, mu=0, test_type='two_sides',#pylint: disable=too-many-arguments, invalid-name
              var_equal=False, conf_level=0.95):
    """
    Perform the T-test for the mean difference of two independent samples.

    Parameters
    ----------

    data : DataFrame
        DataFrame containing the data.

    col1 : str, optional
        Name of the column for sample1.

        If not given, it defaults to the first columns.

    col2 : str, optional
        Name of the column for sample2.

        If not given, it defaults to the second columns.

    mu : float, optional
        Hypothesized difference between the two underlying population means.

        Defaults to 0.

    test_type : {'two_sides', 'less', 'greater'}, optional
        The alternative hypothesis type.

        Defaults to 'two_sides'.

    var_equal : bool, optional
        Controls whether to assume that the two samples have equal variance.

        Defaults to False.

    conf_level : float, optional
        Confidence level for alternative hypothesis confidence interval.

        Defaults to 0.95.

    Returns
    -------

    DataFrame
        Statistics results.

    Examples
    --------

    Original data:

    >>> df.collect()
        X1    X2
    0  1.0  10.0
    1  2.0  12.0
    2  4.0  11.0
    3  7.0  15.0
    4  NaN  10.0

    Perform Independent Sample T-Test:

    >>> ttest_ind(data=df).collect()
               STAT_NAME  STAT_VALUE
    0            t-value   -5.013774
    1  degree of freedom    5.649757
    2            p-value    0.002875
    3      _PAL_MEAN_X1_    3.500000
    4      _PAL_MEAN_X2_   11.600000
    5   confidence level    0.950000
    6         lowerLimit  -12.113278
    7         upperLimit   -4.086722
    """
    conn = data.connection_context
    require_pal_usable(conn)
    col1 = arg('col1', col1, str)
    col2 = arg('col2', col2, str)

    if col1 is None:
        col1 = data.columns[0]
    if col2 is None:
        col2 = data.columns[1]
    data_ = data[[col1, col2]]

    # SQLTRACE
    conn.sql_tracer.set_sql_trace(None, 'Stats', 'ttest_ind')

    return _ttest_base(data_, mu=mu, test_type=test_type, paired=False,
                       var_equal=var_equal, conf_level=conf_level)

def ttest_paired(data, col1=None, col2=None, mu=0, test_type='two_sides',#pylint: disable=too-many-arguments, invalid-name
                 conf_level=0.95):
    """
    Perform the t-test for the mean difference of two sets of paired samples.

    Parameters
    ----------

    data : DataFrame
        DataFrame containing the data.

    col1 : str, optional
        Name of the column for sample1.

        If not given, defaults to the first columns.

    col2 : str, optional
        Name of the column for sample2.

        If not given, defaults to the second columns.

    mu : float, optional
        Hypothesized difference between two underlying population means.

        Defaults to 0.

    test_type : {'two_sides', 'less', 'greater'}, optional
        The alternative hypothesis type.

        Defaults to 'two_sides'.

    conf_level : float, optional
        Confidence level for alternative hypothesis confidence interval.

        Defaults to 0.95.

    Returns
    -------

    DataFrame
        Statistics results.

    Examples
    --------

    Original data:

    >>> df.collect()
        X1    X2
    0  1.0  10.0
    1  2.0  12.0
    2  4.0  11.0
    3  7.0  15.0
    4  3.0  10.0

    perform Paired Sample T-Test

    >>> ttest_paired(data=df).collect()
                    STAT_NAME  STAT_VALUE
    0                 t-value  -14.062884
    1       degree of freedom    4.000000
    2                 p-value    0.000148
    3  _PAL_MEAN_DIFFERENCES_   -8.200000
    4        confidence level    0.950000
    5              lowerLimit   -9.818932
    6              upperLimit   -6.581068
    """
    conn = data.connection_context
    require_pal_usable(conn)
    col1 = arg('col1', col1, str)
    col2 = arg('col2', col2, str)

    if col1 is None:
        col1 = data.columns[0]
    if col2 is None:
        col2 = data.columns[1]
    data_ = data[[col1, col2]]

    # SQLTRACE
    conn.sql_tracer.set_sql_trace(None, 'Stats', 'ttest_paired')

    return _ttest_base(data_, mu=mu, test_type=test_type, paired=True,
                       conf_level=conf_level)

#pylint: disable=too-many-arguments, too-many-locals
def f_oneway(data, group=None, sample=None,
             multcomp_method=None, significance_level=None):
    r"""
    Performs a 1-way ANOVA.

    The purpose of one-way ANOVA is to determine whether there is any \
    statistically significant difference between the means of three \
    or more independent groups.

    Parameters
    ----------

    data : DataFrame
        DataFrame containing the data.

    group : str
        Name of the group column.

        If not provided, it defaults to the first column.

    sample : str, optional
        Name of the sample measurement column.

        If not provided, it defaults to the first non-group column.

    multcomp_method : {'tukey-kramer', 'bonferroni', 'dunn-sidak', 'scheffe', 'fisher-lsd'},  str, optional
        Method used to perform multiple comparison tests.

        Defaults to 'tukey-kramer'.

    significance_level : float, optional
        The significance level when the function calculates the confidence
        interval in multiple comparison tests.

        Values must be greater than 0 and less than 1.

        Defaults to 0.05.

    Returns
    -------

    DataFrame

        Statistics for each group, structured as follows:

            - GROUP, type NVARCHAR(256), group name.
            - VALID_SAMPLES, type INTEGER, number of valid samples.
            - MEAN, type DOUBLE, group mean.
            - SD, type DOUBLE, group standard deviation.

        Computed results for ANOVA, structured as follows:

            - VARIABILITY_SOURCE, type NVARCHAR(100), source of variability, including between groups, within groups (error) and total.
            - SUM_OF_SQUARES, type DOUBLE, sum of squares.
            - DEGREES_OF_FREEDOM, type DOUBLE, degrees of freedom.
            - MEAN_SQUARES, type DOUBLE, mean squares.
            - F_RATIO, type DOUBLE, calculated as mean square between groups divided by mean square of error.
            - P_VALUE, type DOUBLE, associated p-value from the F-distribution.

        Multiple comparison results, structured as follows:

            - FIRST_GROUP, type NVARCHAR(256), the name of the first group to conduct pairwise test on.
            - SECOND_GROUP, type NVARCHAR(256), the name of the second group to conduct pairwise test on.
            - MEAN_DIFFERENCE, type DOUBLE, mean difference between the two groups.
            - SE, type DOUBLE, standard error computed from all data.
            - P_VALUE, type DOUBLE, p-value.
            - CI_LOWER, type DOUBLE, the lower limit of the confidence interval.
            - CI_UPPER, type DOUBLE, the upper limit of the confidence interval.

    Examples
    --------
    Data df:

    >>> df.collect()
       GROUP  DATA
    0      A   4.0
    1      A   5.0
    2      A   4.0
    3      A   3.0
    4      A   2.0
    5      A   4.0
    6      A   3.0
    7      A   4.0
    8      B   6.0
    9      B   8.0
    10     B   4.0
    11     B   5.0
    12     B   4.0
    13     B   6.0
    14     B   5.0
    15     B   8.0
    16     C   6.0
    17     C   7.0
    18     C   6.0
    19     C   6.0
    20     C   7.0
    21     C   5.0

    Perform the function:

    >>> stats, anova, mult_comp = f_oneway(data=df,
    ...                                    multcomp_method='Tukey-Kramer',
    ...                                    significance_level=0.05)

    Outputs:

    >>> stats.collect()
       GROUP  VALID_SAMPLES      MEAN        SD
    0      A              8  3.625000  0.916125
    1      B              8  5.750000  1.581139
    2      C              6  6.166667  0.752773
    3  Total             22  5.090909  1.600866
    >>> anova.collect()
      VARIABILITY_SOURCE  SUM_OF_SQUARES  DEGREES_OF_FREEDOM  MEAN_SQUARES
    0              Group       27.609848                 2.0     13.804924
    1              Error       26.208333                19.0      1.379386
    2              Total       53.818182                21.0           NaN
         F_RATIO   P_VALUE
    0  10.008021  0.001075
    1        NaN       NaN
    2        NaN       NaN
    >>> mult_comp.collect()
      FIRST_GROUP SECOND_GROUP  MEAN_DIFFERENCE        SE   P_VALUE  CI_LOWER
    0           A            B        -2.125000  0.587236  0.004960 -3.616845
    1           A            C        -2.541667  0.634288  0.002077 -4.153043
    2           B            C        -0.416667  0.634288  0.790765 -2.028043
       CI_UPPER
    0 -0.633155
    1 -0.930290
    2  1.194710
    """
    conn = data.connection_context
    require_pal_usable(conn)

    # SQLTRACE
    conn.sql_tracer.set_sql_trace(None, 'Stats', 'f_oneway')
    unique_id = str(uuid.uuid1()).replace('-', '_').upper()

    multcomp_method_map = {'tukey-kramer':0, 'bonferroni':1,
                           'dunn-sidak':2, 'scheffe':3, 'fisher-lsd':4}

    group = arg('group', group, str)
    sample = arg('sample', sample, str)
    multcomp_method = arg('multcomp_method', multcomp_method, multcomp_method_map)
    significance_level = arg('significance_level', significance_level, float)
    if significance_level is not None and not 0 < significance_level < 1:
        msg = "significance_level {!r} is out of bounds.".format(significance_level)
        logger.error(msg)
        raise ValueError(msg)

    cols = data.columns
    if group is None:
        group = cols[0]
    cols.remove(group)
    if sample is None:
        sample = cols[0]

    data_ = data[[group] + [sample]]

    tables = ['STATISTICS', 'ANOVA', 'MULTIPLE_COMPARISON']
    tables = ['#PAL_ANOVA_{}_TBL_{}'.format(name, unique_id) for name in tables]

    stats_tbl, anova_tbl, multi_comparison_tbl = tables

    param_rows = [
        ('MULTCOMP_METHOD', multcomp_method, None, None),
        ('SIGNIFICANCE_LEVEL', None, significance_level, None)]

    try:
        call_pal_auto(conn,
                      'PAL_ONEWAY_ANOVA',
                      data_,
                      ParameterTable().with_data(param_rows),
                      *tables)
    except dbapi.Error as db_err:
        logger.exception(str(db_err))
        try_drop(conn, tables)
        raise

    return (conn.table(stats_tbl),
            conn.table(anova_tbl),
            conn.table(multi_comparison_tbl))

def f_oneway_repeated(data, subject_id, measures=None,
                      multcomp_method=None, significance_level=None, se_type=None):
    """
    Performs one-way repeated measures analysis of variance, along with \
    Mauchly's Test of Sphericity and post hoc multiple comparison tests.

    Parameters
    ----------

    data : DataFrame
        DataFrame containing the data.

    subject_id : str
        Name of the subject ID column.
        The algorithm treats each row of the data table as a different subject.
        Hence there should be no duplicate subject IDs in this column.

    measures : list of str, optional
        Names of the groups (measures).

        If not provided, defaults to all non-subject_id columns.

    multcomp_method : {'tukey-kramer', 'bonferroni', 'dunn-sidak', 'scheffe', 'fisher-lsd'}, optional
        Method used to perform multiple comparison tests.

        Defaults to 'bonferroni'.

    significance_level : float, optional
        The significance level when the function calculates the confidence
        interval in multiple comparison tests.

        Values must be greater than 0 and less than 1.

        Defaults to 0.05.
    se_type : {'all-data', 'two-group'}
        Type of standard error used in multiple comparison tests.
            - 'all-data': computes the standard error from all data. It has
              more power if the assumption of sphericity is true, especially
              with small data sets.
            - 'two-group': computes the standard error from only the two groups
              being compared. It doesn't assume sphericity.

        Defaults to 'two-group'.

    Returns
    -------

    DataFrame

        Statistics for each group, structured as follows:

            - GROUP, type NVARCHAR(256), group name.
            - VALID_SAMPLES, type INTEGER, number of valid samples.
            - MEAN, type DOUBLE, group mean.
            - SD, type DOUBLE, group standard deviation.

        Mauchly test results, structured as follows:

            - STAT_NAME, type NVARCHAR(100), names of test result quantities.
            - STAT_VALUE, type DOUBLE, values of test result quantities.

        Computed results, structured as follows:

            - VARIABILITY_SOURCE, type NVARCHAR(100), source of variability, \
              divided into group, error and subject portions.
            - SUM_OF_SQUARES, type DOUBLE, sum of squares.
            - DEGREES_OF_FREEDOM, type DOUBLE, degrees of freedom.
            - MEAN_SQUARES, type DOUBLE, mean squares.
            - F_RATIO, type DOUBLE, calculated as mean square between groups \
              divided by mean square of error.
            - P_VALUE, type DOUBLE, associated p-value from the F-distribution.
            - P_VALUE_GG, type DOUBLE, p-value of Greehouse-Geisser correction.
            - P_VALUE_HF, type DOUBLE, p-value of Huynh-Feldt correction.
            - P_VALUE_LB, type DOUBLE, p-value of lower bound correction.

        Multiple comparison results, structured as follows:

          - FIRST_GROUP, type NVARCHAR(256), the name of the first group to \
            conduct pairwise test on.
          - SECOND_GROUP, type NVARCHAR(256), the name of the second group \
            to conduct pairwise test on.
          - MEAN_DIFFERENCE, type DOUBLE, mean difference between the two \
            groups.
          - SE, type DOUBLE, standard error computed from all data or \
            compared two groups, depending on ``se_type``.
          - P_VALUE, type DOUBLE, p-value.
          - CI_LOWER, type DOUBLE, the lower limit of the confidence interval.
          - CI_UPPER, type DOUBLE, the upper limit of the confidence interval.

    Examples
    --------
    Data df:

    >>> df.collect()
      ID  MEASURE1  MEASURE2  MEASURE3  MEASURE4
    0  1       8.0       7.0       1.0       6.0
    1  2       9.0       5.0       2.0       5.0
    2  3       6.0       2.0       3.0       8.0
    3  4       5.0       3.0       1.0       9.0
    4  5       8.0       4.0       5.0       8.0
    5  6       7.0       5.0       6.0       7.0
    6  7      10.0       2.0       7.0       2.0
    7  8      12.0       6.0       8.0       1.0


    Perform the function:

    >>> stats, mtest, anova, mult_comp = f_oneway_repeated(
    ...     data=df,
    ...     subject_id='ID',
    ...     multcomp_method='bonferroni',
    ...     significance_level=0.05,
    ...     se_type='two-group')

    Outputs:

    >>> stats.collect()
          GROUP  VALID_SAMPLES   MEAN        SD
    0  MEASURE1              8  8.125  2.232071
    1  MEASURE2              8  4.250  1.832251
    2  MEASURE3              8  4.125  2.748376
    3  MEASURE4              8  5.750  2.915476
    >>> mtest.collect()
                        STAT_NAME  STAT_VALUE
    0                 Mauchly's W    0.136248
    1                  Chi-Square   11.405981
    2                          df    5.000000
    3                      pValue    0.046773
    4  Greenhouse-Geisser Epsilon    0.532846
    5         Huynh-Feldt Epsilon    0.665764
    6         Lower bound Epsilon    0.333333
    >>> anova.collect()
      VARIABILITY_SOURCE  SUM_OF_SQUARES  DEGREES_OF_FREEDOM  MEAN_SQUARES  \
    0              Group          83.125                 3.0     27.708333
    1            Subject          17.375                 7.0      2.482143
    2              Error         153.375                21.0      7.303571
        F_RATIO  P_VALUE  P_VALUE_GG  P_VALUE_HF  P_VALUE_LB
    0  3.793806  0.02557    0.062584    0.048331    0.092471
    1       NaN      NaN         NaN         NaN         NaN
    2       NaN      NaN         NaN         NaN         NaN
    >>> mult_comp.collect()
      FIRST_GROUP SECOND_GROUP  MEAN_DIFFERENCE        SE   P_VALUE  CI_LOWER  \
    0    MEASURE1     MEASURE2            3.875  0.811469  0.012140  0.924655
    1    MEASURE1     MEASURE3            4.000  0.731925  0.005645  1.338861
    2    MEASURE1     MEASURE4            2.375  1.792220  1.000000 -4.141168
    3    MEASURE2     MEASURE3            0.125  1.201747  1.000000 -4.244322
    4    MEASURE2     MEASURE4           -1.500  1.336306  1.000000 -6.358552
    5    MEASURE3     MEASURE4           -1.625  1.821866  1.000000 -8.248955
       CI_UPPER
    0  6.825345
    1  6.661139
    2  8.891168
    3  4.494322
    4  3.358552
    5  4.998955
    """
    conn = data.connection_context
    require_pal_usable(conn)

    # SQLTRACE
    conn.sql_tracer.set_sql_trace(None, 'Stats', 'f_oneway_repeated')

    unique_id = str(uuid.uuid1()).replace('-', '_').upper()

    multcomp_method_map = {'tukey-kramer':0, 'bonferroni':1,
                           'dunn-sidak':2, 'scheffe':3, 'fisher-lsd':4}
    se_type_map = {'all-data':0, 'two-group':1}

    subject_id = arg('subject_id', subject_id, str, required=True)
    measures = arg('measures', measures, ListOfStrings)
    multcomp_method = arg('multcomp_method', multcomp_method, multcomp_method_map)
    significance_level = arg('significance_level', significance_level, float)
    if significance_level is not None and not 0 < significance_level < 1:
        msg = "significance_level {!r} is out of bounds.".format(significance_level)
        logger.error(msg)
        raise ValueError(msg)
    se_type = arg('se_type', se_type, se_type_map)

    cols = data.columns
    cols.remove(subject_id)
    if measures is None:
        measures = cols

    data_ = data[[subject_id] + measures]

    tables = ['STATISTICS', 'MAUCHLY_TEST', 'ANOVA', 'MULTIPLE_COMPARISON']
    tables = ['#PAL_ANOVA_REPEATED_{}_TBL_{}'.format(name, unique_id) for name in tables]

    stats_tbl, mauchly_test_tbl, anova_tbl, multi_comparison_tbl = tables

    param_rows = [
        ('MULTCOMP_METHOD', multcomp_method, None, None),
        ('SIGNIFICANCE_LEVEL', None, significance_level, None),
        ('SE_TYPE', se_type, None, None)]

    try:
        call_pal_auto(conn,
                      'PAL_ONEWAY_REPEATED_MEASURES_ANOVA',
                      data_,
                      ParameterTable().with_data(param_rows),
                      *tables)
    except dbapi.Error as db_err:
        logger.exception(str(db_err))
        try_drop(conn, tables)
        raise

    return (conn.table(stats_tbl),
            conn.table(mauchly_test_tbl),
            conn.table(anova_tbl),
            conn.table(multi_comparison_tbl))

def univariate_analysis(data,
                        key=None,
                        cols=None,
                        categorical_variable=None,
                        significance_level=None,
                        trimmed_percentage=None):
    """
    Provides an overview of the dataset. For continuous columns, it provides \
    the count of valid observations, min, lower quartile, median, upper \
    quartile, max, mean, confidence interval for the mean (lower and upper \
    bound), trimmed mean, variance, standard deviation, skewness, and kurtosis. \
    For discrete columns, it provides the number of occurrences and the \
    percentage of the total data in each category.

    Parameters
    ----------

    data : DataFrame
        DataFrame containing the data.

    key : str, optional
        Name of the ID column.

        If not provided, it is assumed that the input data has no ID column.

    cols : list of str, optional
        List of column names to analyze.

        If not provided, it defaults to all non-ID columns.

    categorical_variable : list of str, optional
        INTEGER columns specified in this list will be treated as categorical
        data.

        By default, INTEGER columns are treated as continuous.

        No default value.

    significance_level : float, optional
        The significance level when the function calculates the confidence
        interval of the sample mean.

        Values must be greater than 0 and less than 1.

        Defaults to 0.05.

    trimmed_percentage : float, optional
        The ratio of data at both head and tail that will be dropped in the
        process of calculating the trimmed mean.

        Value range is from 0 to 0.5.

        Defaults to 0.05.

    Returns
    -------
    DataFrame

        Statistics for continuous variables, structured as follows:

            - VARIABLE_NAME, type NVARCHAR(256), variable names.
            - STAT_NAME, type NVARCHAR(100), names of statistical quantities, \
              including the count of valid observations, min, lower quartile, \
              median, upper quartile, max, mean, confidence interval for the \
              mean (lower and upper bound), trimmed mean, variance, standard \
              deviation, skewness, and kurtosis (14 quantities in total).
            - STAT_VALUE, type DOUBLE, values for the corresponding \
              statistical quantities.

        Statistics for categorical variables, structured as follows:

            - VARIABLE_NAME, type NVARCHAR(256), variable names.
            - CATEGORY, type NVARCHAR(256), category names of the corresponding \
              variables. Null is also treated as a category.
            - STAT_NAME, type NVARCHAR(100), names of statistical quantities: \
              number of observations, percentage of total data points falling \
              in the current category for a variable (including null).
            - STAT_VALUE, type DOUBLE, values for the corresponding \
              statistical quantities.

    Examples
    --------
    Dataset to be analyzed:

    >>> df.collect()
          X1    X2  X3 X4
    0    1.2  None   1  A
    1    2.5  None   2  C
    2    5.2  None   3  A
    3  -10.2  None   2  A
    4    8.5  None   2  C
    5  100.0  None   3  B

    Perform univariate analysis:

    >>> continuous, categorical = univariate_analysis(
    ...     data=df,
    ...     categorical_variable=['X3'],
    ...     significance_level=0.05,
    ...     trimmed_percentage=0.2)

    Outputs:

    >>> continuous.collect()
       VARIABLE_NAME                 STAT_NAME   STAT_VALUE
    0             X1        valid observations     6.000000
    1             X1                       min   -10.200000
    2             X1            lower quartile     1.200000
    3             X1                    median     3.850000
    4             X1            upper quartile     8.500000
    5             X1                       max   100.000000
    6             X1                      mean    17.866667
    7             X1  CI for mean, lower bound   -24.879549
    8             X1  CI for mean, upper bound    60.612883
    9             X1              trimmed mean     4.350000
    10            X1                  variance  1659.142667
    11            X1        standard deviation    40.732575
    12            X1                  skewness     1.688495
    13            X1                  kurtosis     1.036148
    14            X2        valid observations     0.000000
    >>> categorical.collect()
       VARIABLE_NAME      CATEGORY      STAT_NAME  STAT_VALUE
    0             X3  __PAL_NULL__          count    0.000000
    1             X3  __PAL_NULL__  percentage(%)    0.000000
    2             X3             1          count    1.000000
    3             X3             1  percentage(%)   16.666667
    4             X3             2          count    3.000000
    5             X3             2  percentage(%)   50.000000
    6             X3             3          count    2.000000
    7             X3             3  percentage(%)   33.333333
    8             X4  __PAL_NULL__          count    0.000000
    9             X4  __PAL_NULL__  percentage(%)    0.000000
    10            X4             A          count    3.000000
    11            X4             A  percentage(%)   50.000000
    12            X4             B          count    1.000000
    13            X4             B  percentage(%)   16.666667
    14            X4             C          count    2.000000
    15            X4             C  percentage(%)   33.333333
    """
    conn = data.connection_context
    require_pal_usable(conn)

    # SQLTRACE
    conn.sql_tracer.set_sql_trace(None, 'Stats', 'univariate_analysis')

    unique_id = str(uuid.uuid1()).replace('-', '_').upper()

    key = arg('key', key, str)
    cols = arg('cols', cols, ListOfStrings)
    categorical_variable = arg('categorical_variable',
                               categorical_variable, ListOfStrings)
    significance_level = arg('significance_level', significance_level, float)
    if significance_level is not None and not 0 < significance_level < 1:
        msg = "significance_level {!r} is out of bounds.".format(significance_level)
        logger.error(msg)
        raise ValueError(msg)
    trimmed_percentage = arg('trimmed_percentage', trimmed_percentage, float)
    if trimmed_percentage is not None and not 0 <= trimmed_percentage < 0.5:
        msg = "trimmed_percentage {!r} is out of bounds.".format(trimmed_percentage)
        logger.error(msg)
        raise ValueError(msg)
    all_cols = data.columns
    if key is not None:
        id_col = [key]
        all_cols.remove(key)
    else:
        id_col = []
    if cols is None:
        cols = all_cols

    data_ = data[id_col + cols]

    tables = ['CONTINUOUS', 'CATEGORICAL']
    tables = ['#PAL_UNIVARIATE_{}_TBL_{}'.format(name, unique_id) for name in tables]

    continuous_tbl, categorical_tbl = tables

    param_rows = [
        ('SIGNIFICANCE_LEVEL', None, significance_level, None),
        ('TRIMMED_PERCENTAGE', None, trimmed_percentage, None),
        ('HAS_ID', key is not None, None, None)
        ]
    #PAL documentation is inconsistent with example, tests confirmed that the following
    #parameter works as expected
    if categorical_variable is not None:
        param_rows.extend(('CATEGORICAL_VARIABLE', None, None, variable)
                          for variable in categorical_variable)

    try:
        call_pal_auto(conn,
                      'PAL_UNIVARIATE_ANALYSIS',
                      data_,
                      ParameterTable().with_data(param_rows),
                      *tables)
    except dbapi.Error as db_err:
        logger.exception(str(db_err))
        try_drop(conn, tables)
        raise

    return (conn.table(continuous_tbl),
            conn.table(categorical_tbl))

def _multivariate_analysis(data,
                           cols=None,
                           result_type=None):
    conn = data.connection_context
    result_type_map = {'covariance': 0, 'pearsonr': 1}

    unique_id = str(uuid.uuid1()).replace('-', '_').upper()

    cols = arg('cols', cols, ListOfStrings)
    result_type = arg('result_type', result_type, result_type_map)
    data_ = data
    if cols is not None:
        data_ = data[cols]

    result_tbl = '#PAL_MULTIVARIATE_RESULT_TBL_{}'.format(unique_id)
    param_rows = [
        ('RESULT_TYPE', result_type, None, None)
        ]

    try:
        call_pal_auto(conn,
                      'PAL_MULTIVARIATE_ANALYSIS',
                      data_,
                      ParameterTable().with_data(param_rows),
                      result_tbl)
    except dbapi.Error as db_err:
        logger.exception(str(db_err))
        try_drop(conn, result_tbl)
        raise

    return conn.table(result_tbl)

def covariance_matrix(data, cols=None):
    """
    Computes the covariance matrix.

    Parameters
    ----------

    data : DataFrame
        DataFrame containing the data.

    cols : list of str, optional
        List of column names to analyze.

        If not provided, it defaults to all columns.

    Returns
    -------
    DataFrame

        Covariance between any two data samples (columns).

          - ID, type NVARCHAR. The values of this column are the column names from ``cols``.
          - Covariance columns, type DOUBLE, named after the columns in ``cols``.
            The covariance between variables X and Y is in column X, in the row with ID value Y.

    Examples
    --------
    Dataset to be analyzed:

    >>> df.collect()
        X     Y
    0   1   2.4
    1   5   3.5
    2   3   8.9
    3  10  -1.4
    4  -4  -3.5
    5  11  32.8

    Compute the covariance matrix:

    >>> result = covariance_matrix(data=df)

    Outputs:

    >>> result.collect()
      ID          X           Y
    0  X  31.866667   44.473333
    1  Y  44.473333  176.677667
    """
    # SQLTRACE
    conn = data.connection_context
    require_pal_usable(conn)
    conn.sql_tracer.set_sql_trace(None, 'Stats', 'covariance_matrix')

    return _multivariate_analysis(data, cols, result_type='covariance')

def pearsonr_matrix(data, cols=None):
    """
    Computes a correlation matrix using Pearson's correlation coefficient.

    Parameters
    ----------

    data : DataFrame
        DataFrame containing the data.

    cols : list of str, optional
        List of column names to analyze.

        If not provided, it defaults to all columns.

    Returns
    -------
    DataFrame

        Pearson's correlation coefficient between any two data samples
        (columns).

          - ID, type NVARCHAR. The values of this column are the column names from ``cols``.
          - Correlation coefficient columns, type DOUBLE, named after the columns in ``cols``. The correlation coefficient between variables \
            X and Y is in column X, in the row with ID value Y.

    Examples
    --------
    Dataset to be analyzed:

    >>> df.collect()
        X     Y
    0   1   2.4
    1   5   3.5
    2   3   8.9
    3  10  -1.4
    4  -4  -3.5
    5  11  32.8

    Compute the Pearson's correlation coefficient matrix:

    >>> result = pearsonr_matrix(data=df)
    >>> result.collect()
      ID               X               Y
    0  X               1  0.592707653621
    1  Y  0.592707653621               1
    """
    # SQLTRACE
    conn = data.connection_context
    require_pal_usable(conn)
    conn.sql_tracer.set_sql_trace(None, 'Stats', 'pearsonr_matrix')

    return _multivariate_analysis(data, cols, result_type='pearsonr')

def iqr(data, key, col=None, multiplier=None):
    """
    Perform the inter-quartile range (IQR) test to find the outliers of the
    data. The inter-quartile range (IQR) is the difference between the third
    quartile (Q3) and the first quartile (Q1) of the data. Data points will be
    marked as outliers if they fall outside the range from
    Q1 - ``multiplier`` * IQR to Q3 + ``multiplier`` * IQR.


    Parameters
    ----------

    data : DataFrame
        DataFrame containing the data.

    key : str
        Name of the ID column.

    col : str, optional
        Name of the data column that needs to be tested.

        If not given, it defaults to the first non-ID column.

    multiplier : float, optional
        The multiplier used to calculate the value range during the IQR test.

        Upper-bound = Q3 + ``multiplier`` * IQR.

        Lower-bound = Q1 - ``multiplier`` * IQR.

        Q1 is equal to 25th percentile and Q3 is equal to 75th percentile.

        Defaults to 1.5.

    Returns
    -------

    DataFrame

        Test results, structured as follows:

            - ID column, with same name and type as ``data``'s ID column.
            - IS_OUT_OF_RANGE, type INTEGER, containing the test results from
            the IQR test that determine whether each data sample is in the
            range or not:
                - 0: a value is in the range.
                - 1: a value is out of range.

        Statistical outputs, including Upper-bound and Lower-bound from the
        IQR test, structured as follows:

            - STAT_NAME, type NVARCHAR(256), statistics name.
            - STAT_VALUE, type DOUBLE, statistics value.

    Examples
    --------
    Original data:

    >>> df.collect()
         ID   VAL
    0    P1  10.0
    1    P2  11.0
    2    P3  10.0
    3    P4   9.0
    4    P5  10.0
    5    P6  24.0
    6    P7  11.0
    7    P8  12.0
    8    P9  10.0
    9   P10   9.0
    10  P11   1.0
    11  P12  11.0
    12  P13  12.0
    13  P14  13.0
    14  P15  12.0

    Perform the IQR test:

    >>> res, stat = iqr(data=df, key='ID', col='VAL', multiplier=1.5)
    >>> res.collect()
         ID  IS_OUT_OF_RANGE
    0    P1                0
    1    P2                0
    2    P3                0
    3    P4                0
    4    P5                0
    5    P6                1
    6    P7                0
    7    P8                0
    8    P9                0
    9   P10                0
    10  P11                1
    11  P12                0
    12  P13                0
    13  P14                0
    14  P15                0
    >>> stat.collect()
            STAT_NAME  STAT_VALUE
    0  lower quartile        10.0
    1  upper quartile        12.0
    """
    conn = data.connection_context
    require_pal_usable(conn)

    # SQLTRACE
    conn.sql_tracer.set_sql_trace(None, 'Stats', 'iqr')

    multiplier = arg('multiplier', multiplier, float)
    if multiplier is not None and multiplier < 0:
        msg = 'Parameter multiplier should be greater than or equal to 0.'
        logger.error(msg)
        raise ValueError(msg)

    unique_id = str(uuid.uuid1()).replace('-', '_').upper()
    key = arg('key', key, str)
    cols = data.columns
    cols.remove(key)
    col = arg('col', col, str)
    if col is None:
        col = cols[0]
    data_ = data.select(key, col)
    tables = ['#IQR_{}_TBL_{}'.format(name, unique_id) for name in ['RESULT',
                                                                    'STAT']]
    result_tbl, stat_tbl = tables
    param_rows = [('MULTIPLIER', None, multiplier, None)]

    try:
        call_pal_auto(conn,
                      'PAL_IQR',
                      data_,
                      ParameterTable().with_data(param_rows),
                      result_tbl,
                      stat_tbl)
    except dbapi.Error as db_err:
        try_drop(conn, tables)
        logger.exception(str(db_err))
        raise
    return conn.table(result_tbl), conn.table(stat_tbl)

def wilcoxon(data, col=None, mu=None, test_type=None, correction=None):#pylint: disable=invalid-name
    '''
    Perform a one-sample or paired two-sample non-parametric test to check whether
    the median of the data is different from a specific value.

    Parameters
    ----------

    data : DataFrame
        DataFrame containing the data.

    col : str/ListofStrings, optional
        Name of the data column that needs to be tested.

        If not given, the input dataframe must only have one or two columns.

    mu : float, optional
        The location mu0 for the one sample test. It does not affect the two-sample test.

        Defaults to 0.
    test_type : {'two_sides', 'less', 'greater'}, optional
        Specifies the alternative hypothesis type:

        Default to "two_sides".
    corrction : bool, optional
        Controls whether or not to include the continuity correction for
        the p value calculation.

        Default to true.

    Returns
    -------
    DataFrame
        Test results, structured as follows:

          - STAT_NAME column, name of statistics.
          - STAT_VALUE column, value of statistics.


    Examples
    --------
    Original data:

    >>> df.collect()
          X
    0    85
    1    65
    2    20
    3    56
    4    30
    5    46
    6    83
    7    33
    8    89
    9    72
    10   51
    11   76
    12   68
    13   82
    14   27
    15   59
    16   69
    17   40
    18   64
    19   8

    Perform the wilcox signed rank test:

    >>> res = wilcoxon(df, mu=40, test_type='two_sides', correction=true)

    Result:

    >>> res.collect()
         STAT_NAME  STAT_VALUE
    0    statistic  158.5
    1    p-value    0.011228240845317039
    '''
    conn = data.connection_context
    require_pal_usable(conn)
    test_type_map = {'two_sides':0, 'less':1, 'greater':2}
    mu = arg('mu', mu, float)
    test_type = arg('test_type', test_type, test_type_map)
    correction = arg('correction', correction, bool)
    if col is not None:
        if isinstance(col, str):
            col = [col]
        try:
            col = arg('col', col, ListOfStrings)#pylint: disable=undefined-variable
        except:
            msg = ("'col' must be list of string or string.")
            logger.error(msg)
            raise TypeError(msg)
    if col is None:
        col = data.columns
        if len(data.columns) not in (1, 2):
            msg = ("If 'col' is not specified, the input dataframe " +
                   "must only have one or two columns.")
            logger.error(msg)
            raise ValueError(msg)
    else:
        if len(col) not in (1, 2):
            msg = ("'col' can only specify one or two columns of data to be tested.")
            logger.error(msg)
            raise ValueError(msg)
    data_ = data[col]

    unique_id = str(uuid.uuid1()).replace('-', '_').upper()
    result_tbl = '#WILCOX_TEST_RESULT_TBL_{}'.format(unique_id)
    param_rows = [('MU', None, mu, None),
                  ('TEST_TYPE', test_type, None, None),
                  ('CORRECTION', correction, None, None)
                 ]
    try:
        call_pal_auto(conn,
                      'PAL_WILCOX_TEST',
                      data_,
                      ParameterTable().with_data(param_rows),
                      result_tbl)
    except dbapi.Error as db_err:
        logger.exception(str(db_err))
        try_drop(conn, result_tbl)
        raise
    return conn.table(result_tbl)

def median_test_1samp(data, col=None, mu=None, test_type=None,#pylint: disable=invalid-name
                      confidence_interval=None, thread_ratio=None):
    '''
    Perform one-sample non-parametric test to check whether the median of the data is
    different from a user specified one.

    Parameters
    ----------

    data : DataFrame
        DataFrame containing the data.

    col : str, optional
        Name of the data column that needs to be tested.

        If not given, it defauls to the first column.

    mu : float, optional
        The median of data. It only matters in the one sample test.

        Defaults to 0.

    test_type : {'two_sides', 'less', 'greater'}, optional
        Specifies the alternative hypothesis type.

        Default to "two_sides".

    confidence_interval : float, optional
        Confidence interval for the estimated median.

        Default to 0.95.

    thread_ratio : float, optional
        Specifies the ratio of total number of threads that can be used by
        this function.

        The value range is from 0 to 1, where 0 means only using 1 thread, and 1 means using at most all the currently available
        threads.

        Values outside the range will be ignored and this function heuristically determines the number of threads to use.

        Default to 0.

    Returns
    -------
    DataFrame
        Test results, structured as follows:

          - STAT_NAME column, name of statistics.
          - STAT_VALUE column, value of statistics.


    Examples
    --------
    Original data:

    >>> df.collect()
             X
        0    85
        1    65
        2    20
        3    56
        4    30
        5    46
        6    83
        7    33
        8    89
        9    72
        10   51
        11   76
        12   68
        13   82
        14   27
        15   59
        16   69
        17   40
        18   64
        19   8

    Perform the one-sample median test:

    >>> res = onesample_median_test(df, mu=40, test_type='two_sides')

    Result:

    >>> res.collect()
                                      STAT_NAME  STAT_VALUE
        0                          total number   20.000000
        1                number smaller than m0    5.000000
        2                 number larger than m0   14.000000
        3                      estimated median   61.500000
        4  CI for estimated median, lower bound   27.000000
        5  CI for estimated median, upper bound   83.000000
        6                     sign test p-value    0.066457
    '''
    conn = data.connection_context
    require_pal_usable(conn)
    test_type_map = {'two_sides':0, 'less':1, 'greater':2}
    mu = arg('mu', mu, float)
    test_type = arg('test_type', test_type, test_type_map)
    confidence_interval = arg('confidence_interval', confidence_interval, float)
    thread_ratio = arg('thread_ratio', thread_ratio, float)
    col = arg('col', col, str)
    if col is None:
        col = data.columns[0]

    data_ = data[[col]]
    unique_id = str(uuid.uuid1()).replace('-', '_').upper()
    result_tbl = '#1SAMP_MEDIAN_RESULT_TBL_{}'.format(unique_id)
    param_rows = [('M0', None, mu, None),
                  ('TEST_TYPE', test_type, None, None),
                  ('CONFIDENCE_INTERVAL', None, confidence_interval, None),
                  ('THREAD_RATIO', None, thread_ratio, None)]
    try:
        call_pal_auto(conn,
                      'PAL_SIGN_TEST',
                      data_,
                      ParameterTable().with_data(param_rows),
                      result_tbl)
    except dbapi.Error as db_err:
        logger.exception(str(db_err))
        try_drop(conn, result_tbl)
        raise
    return conn.table(result_tbl)

def grubbs_test(data, key, col=None, method=None, alpha=None):
    '''
    Perform grubbs' test to detect outliers from a given univariate data set.
    The algorithm assumes that Y comes from Gaussian distribution.

    Parameters
    ----------

    data : DataFrame
        DataFrame containing the data.

    key : str
        Name of the ID column.

    col : str, optional
        Name of the data column that needs to be tested.

        If not given, defauls to the first non-ID column.

    method : {'two_sides', 'one_side_min', 'one_side_max', 'repeat_two_sides'}, optional
        Specifies the alternative  type.

        Default to "one_side_min".

    alpha : flaot, optional
        Significance level.

        Default to 0.05.

    Returns
    -------
    DataFrame
        Test results, structured as follows:

          - SOURCE_ID column name, ID of outlier data.
          - RAW_DATA column name, value of original data.

        Statistics, structured as follows:

          - SOUCE_ID column name, ID of outlier data.
          - STAT_NAME column, name of statistics.
          - STAT_VALUE column, value of statistics.

    Examples
    --------
    Original data:

    >>> df.collect()
             ID        VAL
        0   100   4.254843
        1   200   0.135000
        2   300  11.072257
        3   400  14.797838
        4   500  12.125133
        5   600  14.265839
        6   700   7.731352
        7   800   6.856739
        8   900  15.094403
        9   101   8.149382
        10  201   9.160144

    Perform the grubb's test:

    >>> res, stats = grubbs_test(data, key='ID', method='one_side_max', alpha=0.2)

    Results:

    >>> res.collect()
           ID    VAL
        0  200  0.135
    >>> stats.collect()
            ID                 STAT_NAME  STAT_VALUE
        0  200                      MEAN    9.422085
        1  200  STANDARD_SAMPLE_VARIANCE    4.675935
        2  200                         T    1.910219
        3  200                         G    1.986145
        4  200                         U    0.566075
    '''
    conn = data.connection_context
    require_pal_usable(conn)
    method_map = {'two_sides':0, 'one_side_min':1, 'one_side_max':2, 'repeat_two_sides':3}
    method = arg('method', method, method_map)
    alpha = arg('alpha', alpha, float)
    col = arg('col', col, str)
    key = arg('key', key, str)
    cols = data.columns
    cols.remove(key)
    if col is None:
        col = cols[0]
    data_ = data[[key] + [col]]
    unique_id = str(uuid.uuid1()).replace('-', '_').upper()
    tables = ['RESULT', 'STATS']
    tables = ["#GRUBBS_TEST_{}_TBL_{}".format(tbl, unique_id) for tbl in tables]
    res_tbl, stats_tbl = tables
    param_rows = [('METHOD', method, None, None),
                  ('ALPHA', None, alpha, None)]
    try:
        call_pal_auto(conn,
                      'PAL_GRUBBS_TEST',
                      data_,
                      ParameterTable().with_data(param_rows),
                      *tables)
    except dbapi.Error as db_err:
        logger.exception(str(db_err))
        try_drop(conn, tables)
        raise
    return conn.table(res_tbl), conn.table(stats_tbl)

def entropy(data, col=None,
            distinct_value_count_detail=True, thread_ratio=None):
    '''
    This function is used to calculate the information entropy of attributes.

    Parameters
    ----------

    data : DataFrame
        DataFrame containing the data.

    col : str/ListofStrings, optional
        Name of the data column that needs to be processed.

        If not given, it defaults to all columns.

    distinct_value_count_detail : bool, optional
        Indicates whether to output the details of distinct value counts:

            - False: Does not output detailed distinct value count.
            - True: Outputs detailed distinct value count.

        Default to True.

    thread_ratio : flaot, optional
        Specifies the ratio of total number of threads that can be used by this function.

        The value range is from 0 to 1, where 0 means only using 1 thread, and 1 means using
        at most all the currently available threads.

        Values outside the range are ignored and this function heuristically determines the number of threads to use.

        Default to 0.

    Returns
    -------
    DataFrame
        Entropy results, structured as follows:
          - COLUMN_NAME, name of columns.
          - ENTROPY, entropy of columns.
          - COUNT_OF_DISTINCT_VALUES, count of distinct values.

        Distinct values results, structured as follows:
          - COLUMN_NAME, name of columns.
          - DISTINCT_VALUE, distinct values of columns.
          - COUNT, count of distinct values.

    Examples
    --------
    Original data:

    >>> df.collect()
              OUTLOOK  TEMP  HUMIDITY WINDY        CLASS
        0      Sunny  75.0      70.0   Yes         Play
        1      Sunny   NaN      90.0   Yes  Do not Play
        2      Sunny  85.0       NaN    No  Do not Play
        3      Sunny  72.0      95.0    No  Do not Play
        4       None   NaN      70.0  None         Play
        5   Overcast  72.0      90.0   Yes         Play
        6   Overcast  83.0      78.0    No         Play
        7   Overcast  64.0      65.0   Yes         Play
        8   Overcast  81.0      75.0    No         Play
        9       None  71.0      80.0   Yes  Do not Play
        10      Rain  65.0      70.0   Yes  Do not Play
        11      Rain  75.0      80.0    No         Play
        12      Rain  68.0      80.0    No         Play
        13      Rain  70.0      96.0    No         Play

    Calculate the entropy:

    >>> res1, res2 = entropy(data, col=['TEMP','WINDY'],
                             distinct_value_count_detail=False)
    >>> res1.collect()
          COLUMN_NAME   ENTROPY  COUNT_OF_DISTINCT_VALUES
        0        TEMP  2.253858                        10
        1       WINDY  0.690186                         2
    >>> res2.collect()
        Empty DataFrame
        Columns: [COLUMN_NAME, DISTINCT_VALUE, COUNT]
        Index: []
    '''
    conn = data.connection_context
    require_pal_usable(conn)
    distinct_value_count_detail = arg('distinct_value_count_detail', distinct_value_count_detail, bool)#pylint: disable=line-too-long
    thread_ratio = arg('thread_ratio', thread_ratio, float)
    data_ = data
    if col is not None:
        if isinstance(col, str):
            col = [col]
        try:
            col = arg('col', col, ListOfStrings)#pylint: disable=undefined-variable
            data_ = data[col]
        except:
            msg = ("'col' must be list of string or string.")
            logger.error(msg)
            raise TypeError(msg)
    unique_id = str(uuid.uuid1()).replace('-', '_').upper()

    tables = ['RESULT_1', 'RESULT_2']
    tables = ["#ENTROPY_{}_TBL_{}".format(tbl, unique_id) for tbl in tables]
    res1_tbl, res2_tbl = tables
    param_rows = [('DISTINCT_VALUE_COUNT_DETAIL', distinct_value_count_detail, None, None),
                  ('THREAD_RATIO', None, thread_ratio, None)]
    try:
        call_pal_auto(conn,
                      'PAL_ENTROPY',
                      data_,
                      ParameterTable().with_data(param_rows),
                      *tables)
    except dbapi.Error as db_err:
        logger.exception(str(db_err))
        try_drop(conn, tables)
        raise
    return conn.table(res1_tbl), conn.table(res2_tbl)

def condition_index(data, key=None, col=None, scaling=True,
                    include_intercept=True, thread_ratio=None):
    '''
    Condition index is used to detect collinearity problem between independent
    variables which are later used as predictors in a multiple linear regression model.

    Parameters
    ----------

    data : DataFrame
        DataFrame containing the data.

    key : str, optional
        Name of the ID column.

    col : str/ListofStrings, optional
        Name of the feature column that needs to be processed.

        If not given, it defaults to all non-ID columns.

    scaling : bool, optional
        Specifies whether the input data are scaled to have unit variance before the analysis.

            - False: No
            - True: Yes

        Default to True.

    include_intercept : bool, optional
        Specifies whether the algorithm considers intercept during the calculation.

            - False: No
            - True: Yes

        Default to True.

    thread_ratio : float, optional
        Specifies the ratio of total number of threads that can be used by this function.

        The value range is from 0 to 1, where 0 means only using 1 thread, and 1 means using
        at most all the currently available threads.

        Values outside the range are ignored and this function heuristically determines the number of threads to use.

        Default to 0.

    Returns
    -------
    DataFrame
        Condition index results, structured as follows:

          - COMPONENT_ID, principal component ID.
          - EIGENVALUE, eigenvalue.
          - CONDITION_INDEX, Condition index.
          - FEATURES, variance decomposition proportion for each variable.
          - INTERCEPT, variance decomposition proportion for the intercept term.

        Second DataFrame is empty if collinearity problem has not been detected.
        Distinct values results, structured as follows:

          - STAT_NAME, Name for the values, including condition number,
            and the name of variables which are involved in collinearity problem.
          - STAT_VALUE, values of the corresponding name.

    Examples
    --------
    Original data:

    >>> df.collect()
           ID    X1    X2    X3    X4
        0   1  12.0  52.0  20.0  44.0
        1   2  12.0  57.0  25.0  45.0
        2   3  12.0  54.0  21.0  45.0
        3   4  13.0  52.0  21.0  46.0
        4   5  14.0  54.0  24.0  46.0
    Apply the condition index function:

    >>> res, stats = condition_index(data,  key='ID', scaling=True,
                                     include_intercept=True, thread_ratio=0.1)
    >>> res.collect()
            COMPONENT_ID  EIGENVALUE  CONDITION_INDEX        X1        X2        X3        X4  INTERCEPT      #pylint: disable=line-too-long
        0       Comp_1   19.966688         1.000000  0.000012  0.000002  0.000010  0.000003   0.000002        #pylint: disable=line-too-long
        1       Comp_2    0.020736        31.030738  0.008776  0.000210  0.031063  0.001251   0.000907        #pylint: disable=line-too-long
        2       Comp_3    0.012260        40.355748  0.053472  0.002571  0.005315  0.000639   0.002710        #pylint: disable=line-too-long
        3       Comp_4    0.000230       294.940696  0.205666  0.015224  0.006579  0.931121   0.246862        #pylint: disable=line-too-long
        4       Comp_5    0.000086       480.735654  0.732074  0.981993  0.957034  0.066986   0.749518        #pylint: disable=line-too-long
    >>> stats.collect()
            STAT_NAME  STAT_VALUE
        0  CONDITION_NUMBER  480.735654
        1                X1    0.732074
        2                X2    0.981993
        3                X3    0.957034
        4         INTERCEPT    0.749518
    '''
    conn = data.connection_context
    require_pal_usable(conn)
    include_intercept = arg('include_intercept', include_intercept, bool)
    scaling = arg('scaling', scaling, bool)
    thread_ratio = arg('thread_ratio', thread_ratio, float)
    key = arg('key', key, str)
    cols = data.columns
    data_ = None
    if key is not None:
        cols.remove([key])
    col = arg('col', col, str)
    if col is not None and key is not None:
        data_ = data.select(key, col)
    elif col is not None and key is None:
        data_ = data.select(col)
    elif col is None and key is not None:
        data_ = data.select(key, cols)
    else:
        data_ = data

    unique_id = str(uuid.uuid1()).replace('-', '_').upper()

    tables = ['RESULT', 'HELPER']
    tables = ["#CONDITION_INDEX_{}_TBL_{}".format(tbl, unique_id) for tbl in tables]
    res_tbl, helper_tbl = tables
    param_rows = [('INCLUDE_INTERCEPT', include_intercept, None, None),
                  ('SCALING', scaling, None, None),
                  ('THREAD_RATIO', None, thread_ratio, None)]
    try:
        call_pal_auto(conn,
                      'PAL_CONDITION_INDEX',
                      data_,
                      ParameterTable().with_data(param_rows),
                      *tables)
    except dbapi.Error as db_err:
        logger.exception(str(db_err))
        try_drop(conn, tables)
        raise
    return conn.table(res_tbl), conn.table(helper_tbl)

def ftest_equal_var(data_x, data_y, test_type=None):
    '''
    This function is used to test the equality of two random
    variances using F-test. The null hypothesis is that two
    independent normal variances are equal. The observed sums
    of some selected squares are then examined to see whether
    their ratio is significantly incompatible with this null hypothesis.

    Parameters
    ----------

    data_x : DataFrame
        DataFrame containing the first column data.

    data_y : DataFrame
        DataFrame containing the second column data.

    test_type : {'two_sides', 'less', 'greater'}, optional
        Specifies the alternative hypothesis type.

        Default to "two_sides".

    Returns
    -------
    stats_tbl : DataFrame
        Test results, structured as follows:

          - STAT_NAME, name of statistics.
          - STAT_VALUE, value of statistics.

    Examples
    --------
    Original data:

    >>> df_x.collect()
           X
        0  1
        1  2
        2  4
        3  7
        4  3
    >>> df_y.collect()
              Y
        0  10.0
        1  15.0
        2  12.0

    Apply the ftest_equal_var function:

    >>> res = ftest_equal_var(data_x, data_y, test_type='two_sides')
    >>> res.collect()
                               STAT_NAME  STAT_VALUE
        0                        F Value    0.836842
        1    numerator degree of freedom    4.000000
        2  denominator degree of freedom    2.000000
        3                        p-value    0.783713
    '''
    conn = data_x.connection_context
    require_pal_usable(conn)
    test_type_map = {'two_sides':0, 'less':1, 'greater':2}
    test_type = arg('test_type', test_type, test_type_map)
    if len(data_x.columns) * len(data_y.columns) != 1:
        msg = ("Both input dataframe must only have one column.")
        logger.error(msg)
        raise ValueError(msg)
    if not all(var.dtypes(var.columns[0])[0][1] in ('INT', 'DOUBLE') for var in (data_x, data_y)):
        msg = ("Data type in both input dataframe must be numerical.")
        logger.error(msg)
        raise TypeError(msg)
    unique_id = str(uuid.uuid1()).replace('-', '_').upper()
    res_tbl = "#FTEST_EQUAL_VAR_STATS_TBL_{}".format(unique_id)
    param_rows = [('TEST_TYPE', test_type, None, None)]
    try:
        call_pal_auto(conn,
                      'PAL_EQUAL_VARIANCE_TEST',
                      data_x,
                      data_y,
                      ParameterTable().with_data(param_rows),
                      res_tbl)
    except dbapi.Error as db_err:
        logger.exception(str(db_err))
        try_drop(conn, res_tbl)
        raise
    return conn.table(res_tbl)

def factor_analysis(data, key, factor_num, col=None,
                    method=None, rotation=None,
                    score=None, matrix=None, kappa=None):
    '''
    Factor analysis is a statistical method that tries to extract a low
    number of unobserved variables, i.e. factors, that can best describe
    the covariance pattern of a larger set of observed variables.

    Parameters
    ----------

    data : DataFrame
        DataFrame containing the data.

    key : str
        Name of the ID column.

    factor_num : int
        Number of factors.

    col : str/ListofStrings, optional
        Name of the feature columns.

    method : {'principle_component'}, optional
        Specifies the method used for factor analysis.

        Currently SAP HANA PAL only supports the principal component method.

    rotation : {'non', 'varimax', 'promax'}, optional
        Specifies the rotation to be performed on loadings.

        Default to 'varimax'.

    score : {'non', 'regression'}, optional
        Specifies the method to compute factor scores.

        Default to 'regression'.

    matrix : {'covariance', 'correlation'}, optional
        Uses cor matrix to perform factor analysis.

        Default to 'correlation'.

    kappa : float, optional
        Power of promax rotation. (Only valid when rotation is promax.)

        Default to 4.

    Returns
    -------
    DataFrame
     DataFrame 1:
     Eigenvalues, structured as follows:

        - FACTOR_ID: factor id.
        - EIGENVALUE: Eigenvalue (i.e. variance explained).
        - VAR_PROP: Variance proportion to the total variance explained.
        - CUM_VAR_PROP: Cumulative variance proportion to the total variance explained.

     DataFrame 2:
     Variance explanation, structured as follows:

        - FACTOR_ID: factor id.
        - VAR: Variance explained without rotation .
        - VAR_PROP: Variance proportion to the total variance explained without rotation.
        - CUM_VAR_PROP: Cumulative variance proportion to the total variance explained without rotation.
        - ROT_VAR: Variance explained with rotation.
        - ROT_VAR_PROP: Variance proportion to the total variance explained withrotation.
          Note that there is no rotated variance proportion when performing oblique rotation since
          the rotated factors are correlated.
        - ROT_CUM_VAR_PROP: Cumulative variance proportion to the total variance explained with rotation.

     DataFrame 3:
     Communalities, structured as follows:

         - NAME:  variable name.
         - OBERVED_VARS: Communalities of observed variable.

     DataFrame 4:
     Loadings, structured as follows:

         - FACTOR_ID:  Factor id.
         - LOADINGs_+OBSERVED_VARs: loadings.

     DataFrame 5:
     Rotated loadings, structured as follows:

         - FACTOR_ID:  Factor id.
         - ROT_LOADINGS_+OBSERVED_VARs: rotated loadings.

     DataFrame 6:
     Structure, structured as follows:

         - FACTOR_ID:  Factor id.
         - STRUCTURE+OBSERVED_VARS: Structure matrix. It is empty when rotation is not oblique.

     DataFrame 7:
     Rotation, structured as follows:

         - ROTATION:  rotation
         - ROTATION_ + i (i sequences from 1 to number of columns in OBSERVED_VARS (in input table) : Rotation matrix.

     DataFrame 8:
     Factor correlation, structured as follows:

         - FACTOR_ID:  Factor id.
         - FACTOR_ + i (i sequences from 1 to number of columns in OBSERVED_VARS (in input table): Factor correlation matrix.
           It is empty when rotation is not oblique.

     DataFrame 9:
     Score model, structured as follows:

         - NAME:  Factor id, MEAN, SD.
         - OBSERVED_VARS (in input table) column name: Score coefficients, means and standard deviations of observed variables.

     DataFrame 10:
     Scores, structured as follows:

         - FACTOR_ID:  Factor id.
         - FACTOR_ + i (i sequences from 1 to number of columns in OBSERVED_VARS(in input table)): scores.

     DataFrame 11:
     Statistics, placeholder for future features, structured as follows:

         - STAT_NAME:  statistic name.
         - STAT_VALUE: statistic value.


    Examples
    --------
    Original data:

    >>> df.collect()
            ID   X1   X2   X3   X4   X5   X6
        0    1  1.0  1.0  3.0  3.0  1.0  1.0
        1    2  1.0  2.0  3.0  3.0  1.0  1.0
        2    3  1.0  1.0  3.0  4.0  1.0  1.0
        3    4  1.0  1.0  3.0  3.0  1.0  2.0
        4    5  1.0  1.0  3.0  3.0  1.0  1.0
        5    6  1.0  1.0  1.0  1.0  3.0  3.0
        6    7  1.0  2.0  1.0  1.0  3.0  3.0
        7    8  1.0  1.0  1.0  2.0  3.0  3.0
        8    9  1.0  2.0  1.0  1.0  3.0  4.0
        9   10  1.0  1.0  1.0  1.0  3.0  3.0
        10  11  3.0  3.0  1.0  1.0  1.0  1.0
        11  12  3.0  4.0  1.0  1.0  1.0  1.0
        12  13  3.0  3.0  1.0  2.0  1.0  1.0
        13  14  3.0  3.0  1.0  1.0  1.0  2.0
        14  15  3.0  3.0  1.0  1.0  1.0  1.0
        15  16  4.0  4.0  5.0  5.0  6.0  6.0
        16  17  5.0  6.0  4.0  6.0  4.0  5.0
        17  18  6.0  5.0  6.0  4.0  5.0  4.0

    Apply the factor_analysis function:

    >>> res = factor_analysis(data, key='ID', factor_num=2,
                              rotation='promax',
                              matrix='correlation')
    >>> res[0].collect()
          FACTOR_ID  EIGENVALUE  VAR_PROP  CUM_VAR_PROP
        0  FACTOR_1    3.696031  0.616005      0.616005
        1  FACTOR_2    1.073114  0.178852      0.794858
        2  FACTOR_3    1.000774  0.166796      0.961653
        3  FACTOR_4    0.161003  0.026834      0.988487
        4  FACTOR_5    0.040961  0.006827      0.995314
        5  FACTOR_6    0.028116  0.004686      1.000000
    '''
    conn = data.connection_context
    require_pal_usable(conn)
    method_map = {'principal_component': 0}
    rotation_map = {'non': 0, 'varimax': 1, 'promax': 2}
    score_map = {'non': 0, 'regression': 1}
    matrix_map = {'covariance': 0, 'correlation': 1}

    factor_num = arg('factor_num', factor_num, int)
    method = arg('method', method, method_map)
    rotation = arg('rotation', rotation, rotation_map)
    score = arg('score', score, score_map)
    matrix = arg('matrix', matrix, matrix_map)
    kappa = arg('kappa', kappa, float)
    columns = data.columns
    key = arg('key', key, str)
    columns.remove(key)
    data_ = data
    if col is None:
        data_ = data[[key]+columns]
    if col is not None:
        if isinstance(col, str):
            col = [col]
        try:
            col = arg('col', col, ListOfStrings)#pylint: disable=undefined-variable
            data_ = data[[key]+col]
        except:
            msg = ("'col' must be list of string or string.")
            logger.error(msg)
            raise TypeError(msg)
    unique_id = str(uuid.uuid1()).replace('-', '_').upper()
    tables = ['EIGENVALUES', 'VARIANCE_EXPLANATION', 'COMMUNALITIES',
              'LOADINGS', 'ROTATED_LOADINGS', 'STRUCTURE', 'ROTATION',
              'FACTOR_CORRELATION', 'SCORE_MODEL', 'SCORES', 'STATISTICS']
    tables = ["#CDF_{}_TBL_{}".format(tbl, unique_id) for tbl in tables]
    eigen_tbl, var_explanation_tbl, communalities_tbl, loadings_tbl, rotated_loadings_tbl, structure_tbl, rotation_tbl, factor_correlation_tbl, score_model_tbl, scores_tbl, stats_tbl = tables#pylint: disable=line-too-long
    param_rows = [('FACTOR_NUMBER', factor_num, None, None),
                  ('METHOD', method, None, None),
                  ('ROTATION', rotation, None, None),
                  ('SCORE', score, None, None),
                  ('COR', matrix, None, None),
                  ('KAPPA', None, kappa, None)
                 ]
    try:
        call_pal_auto(conn,
                      'PAL_FACTOR_ANALYSIS',
                      data_,
                      ParameterTable().with_data(param_rows),
                      *tables)
    except dbapi.Error as db_err:
        logger.exception(str(db_err))
        try_drop(conn, tables)
        raise
    return conn.table(eigen_tbl), conn.table(var_explanation_tbl), conn.table(communalities_tbl), conn.table(loadings_tbl), conn.table(rotated_loadings_tbl), conn.table(structure_tbl), conn.table(rotation_tbl), conn.table(factor_correlation_tbl), conn.table(score_model_tbl), conn.table(scores_tbl), conn.table(stats_tbl)#pylint: disable=line-too-long

def kaplan_meier_survival_analysis(data, event_indicator=None, conf_level=None):
    '''
    The Kaplan-Meier estimator is a non-parametric statistic used to estimate
    the survival function from lifetime data. It is often used to measure the
    time-to-death of patients after treatment or time-to-failure of machine parts.

    Parameters
    ----------

    data : DataFrame
        DataFrame containing the data.

    event_indicator : int, optional
        Specifies one value to indicate an event has occurred.

        Default to 1.

    conf_level : float, optional
        Specifies confidence level for a two-sided confidence interval on the survival estimate.

        Default to 0.95.

    Returns
    -------
    DataFrame
        Survival estimates, structured as follows:

            - GROUP, group.
            - TIME, event occurrence time. Survival estimates at all event times are output.
            - RISK_NUMBER, number at risk (total number of survivors at the beginning of each period).
            - EVENT_NUMBER, number of event occurrences.
            - PROBABILITY, probability of surviving beyond event occurrence time.
            - SE, standard error for the survivor estimate.
            - CI_LOWER, lower bound of confidence interval.
            - CI_UPPER, upper bound of confidence interval.

        Log rank test statistics result 1, structured as follows:

            - GROUP, group.
            - TOTAL_RISK, all individuals in the lifetime study.
            - OBSERVED, observed event number.
            - EXPECTED, expected event number.
            - LOGRANK_STAT, log rank test statistics.

        Log rank test statistics result 2, structured as follows:

            - STAT_NAME, name of statistics.
            - STAT_VALUE, value of statistics.


    Examples
    --------
    Original data:

    >>> df.collect()
              TIME  STATUS  OCCURRENCES  GROUP
        0      9       1            1      2
        1     10       1            1      1
        2      1       1            2      0
        3     31       0            1      1
        4      2       1            1      0
        5     25       1            3      1
        6    255       0            1      0
        7     90       1            1      0
        8     22       1            1      1
        9    100       0            1      1
        10    28       0            1      0
        11     5       1            1      1
        12     7       1            1      1
        13    11       0            1      0
        14    20       0            1      0
        15    30       1            2      2
        16   101       0            1      2
        17     8       0            1      1

    Perform the function:

    >>> survival_estimates, res, stats = kaplan_meier_survival_analysis(data)
    >>> res.collect()
              GROUP  TOTAL_RISK  OBSERVED  EXPECTED  LOGRANK_STAT
        0     0           8         4  4.353652      0.045712
        1     1          10         7  6.024638      0.307951
        2     2           4         3  3.621710      0.161786
    '''
    conn = data.connection_context
    require_pal_usable(conn)
    if len(data.columns) != 4:
        msg = ("Input dataframe should include follow-up time, status "
               +"indicator, occurrence number of events, and group.")
        logger.error(msg)
        raise ValueError(msg)
    event_indicator = arg('event_indicator', event_indicator, int)
    conf_level = arg('conf_level', conf_level, float)
    unique_id = str(uuid.uuid1()).replace('-', '_').upper()

    tables = ['SURVIVAL_ESTIMATES', 'LOG_RANK_TEST_STATISTICS_1', 'LOG_RANK_TEST_STATISTICS_2']
    tables = ['#KMSURV_{}_TBL_{}'.format(name, unique_id) for name in tables]
    survival_estimates_tbl, res_tbl, stats_tbl = tables

    param_rows = [('EVENT_INDICATOR', event_indicator, None, None),
                  ('CONF_LEVEL', None, conf_level, None)]
    try:
        call_pal_auto(conn,
                      'PAL_KMSURV',
                      data,
                      ParameterTable().with_data(param_rows),
                      *tables)
    except dbapi.Error as db_err:
        logger.exception(str(db_err))
        try_drop(conn, tables)
        raise
    return conn.table(survival_estimates_tbl), conn.table(res_tbl), conn.table(stats_tbl)#pylint: disable=line-too-long

def _distr_table(conn, distr_info, unique_id):
    """
    This is a helper function to turn input dict distr_info to a
    hana DataFrame.
    """
    distr_info = arg('distri_info', distr_info, dict)
    distr_fields = list(distr_info.keys())
    distr_fields = arg('distr_fields', distr_fields, ListOfStrings)
    distr_fields = [field.lower() for field in distr_fields]
    if distr_fields not in (['name', 'min', 'max'],
                            ['name', 'mean', 'variance'],
                            ['name', 'shape', 'scale']):
        msg = ('Unrecognized distribution parameter settings.')
        logger.error(msg)
        raise ValueError(msg)
    distr_type = arg('distr_type', list(distr_info.values())[0], str).lower()
    if distr_type not in ('uniform', 'normal', 'weibull', 'gamma'):
        msg = ("Unrecognized distribution name {}.".format(distr_type))
        logger.error(msg)
        raise ValueError(msg)
    #pylint:disable=too-many-boolean-expressions
    if (distr_type == 'uniform' and distr_fields[1:] != ['min', 'max']) or\
       (distr_type == 'normal' and distr_fields[1:] != ['mean', 'variance']) or\
       (distr_type in ('weibull', 'gamma') and distr_fields[1:] != ['shape', 'scale']):
        msg = ("Distribution name '{}' is mismatched with parameters '{}' and '{}'.".format(distr_type, distr_fields[1], distr_fields[2]))#pylint:disable=line-too-long
        raise ValueError(msg)
    distr_params = list(distr_info.values())[1:]
    if not all(isinstance(par, (int, float)) for par in distr_params):
        msg = "Distribution parameters must be of numerical types."
        logger.error(msg)
        raise ValueError(msg)
    distr_data = [("DistributionName", distr_type.capitalize()),
                  (distr_fields[1].capitalize(), str(distr_params[0])),
                  (distr_fields[2].capitalize(), str(distr_params[1]))]
    distr_tbl = "#DISTR_TBL_{}".format(unique_id)
    distr_spec = [("NAME", NVARCHAR(50)), ("VALUE", NVARCHAR(50))]
    create(conn, Table(distr_tbl, distr_spec).with_data(distr_data))
    return distr_tbl

def cdf(data, distr_info, col=None, complementary=False):
    '''
    This algorithm evaluates the probability of a variable x from the cumulative distribution
    function (CDF) or complementary cumulative distribution function (CCDF) for a given
    probability distribution.

    Parameters
    ----------

    data : DataFrame
        DataFrame containing the data.

    distr_info : dict
        A python dictionary object that contains the distribution name and parameter.
        Supported distributions include: uniform, normal, weibull and gamma.
        Examples for illustration:

            - {'name':'normal', 'mean':0, 'variance':1.0}.
            - {'name':'uniform', 'min':0.0, 'max':1.0}.
            - {'name':'weibull', 'shape':1.0, 'scale':1.0}.
            - {'name':'gamma', 'shape':1.0, 'scale':1.0}.
        You may change the parameter values followed by any of the supported distribution name
        listed as above.

    col : str, optional
        Name of the column in the data frame that needs to be processed.
        If not given, the input dataframe data should only have one column.

    complementary : bool, optional

        - False: 'cdf'.
        - True: 'ccdf'.

        Default to False.

    Returns
    -------
    DataFrame
        CDF results.

    Examples
    --------
    Original data:

    >>> df.collect()
            DATACOL
        0     37.4
        1    277.9
        2    463.2
    >>> df_distri.collect()
                        NAME    VALUE
        0  DistributionName  Weibull
        1             Shape  2.11995
        2             Scale  277.698

    Apply the cdf function:

    >>> res = cdf(data, distri)
    >>> res.collect()
           DATACOL  PROBABILITY
        0     37.4     0.014160
        1    277.9     0.632688
        2    463.2     0.948094
    '''
    conn = data.connection_context
    require_pal_usable(conn)
    complementary = arg('complementary', complementary, bool)

    col = arg('col', col, str)
    if col is None:
        if len(data.columns) != 1:
            msg = "If 'col' is not given, the input dataframe must only have one column."
            logger.error(msg)
            raise ValueError(msg)
        col = data.columns[0]
    data_ = data[[col]]
    unique_id = str(uuid.uuid1()).replace('-', '_').upper()
    distr_tbl = _distr_table(conn, distr_info, unique_id)
    tables = ['RESULT']
    tables = ["#CDF_{}_TBL_{}".format(tbl, unique_id) for tbl in tables]
    res_tbl = tables[0]
    param_rows = [('LOWER_UPPER', complementary, None, None)]
    try:
        call_pal_auto(conn,
                      'PAL_DISTRIBUTION_FUNCTION',
                      data_,
                      conn.table(distr_tbl),
                      ParameterTable().with_data(param_rows),
                      *tables)
    except dbapi.Error as db_err:
        logger.exception(str(db_err))
        try_drop(conn, [distr_tbl, res_tbl])
        raise
    return conn.table(res_tbl)

def quantile(data, distr_info, col=None, complementary=False):
    '''
    This algorithm evaluates the inverse of the cumulative distribution function (CDF)
    or the inverse  of the complementary cumulative distribution function (CCDF) for a
    given probability p and probability distribution.

    Parameters
    ----------

    data : DataFrame
        DataFrame containing the data.

    distr_info : dict
        A python dictionary object that contains the distribution name and parameter.
        Supported distributions include: uniform, normal, weibull and gamma.
        Examples for illustration:

            - {'name':'normal', 'mean':0, 'variance':1.0}.
            - {'name':'uniform', 'min':0.0, 'max':1.0}.
            - {'name':'weibull', 'shape':1.0, 'scale':1.0}.
            - {'name':'gamma', 'shape':1.0, 'scale':1.0}.
        You may change the parameter values followed by any of the supported distribution name
        listed as above.

    col : str, optional
        Name of the column in the data frame that needs to be processed.

        If not given, it defaults to the first column.

    complementary : bool, optional

        - False: 'cdf'
        - True: 'ccdf'

        Default to False.

    Returns
    -------
    DataFrame
        CDF results.

    Examples
    --------
    Original data:

    >>> df.collect()
            DATACOL
        0    0.3
        1    0.5
        2    0.632
        3    0.8
    >>> df_distr.collect()
                        NAME    VALUE
        0  DistributionName  Weibull
        1             Shape  2.11995
        2             Scale  277.698

    Apply the quantile function:

    >>> res = quantile(data, distr)
    >>> res.collect()
           DATACOL  QUANTILE
        0     0.3     170.755854
        1     0.5     233.608506
        2     0.632   277.655075
        3     0.8     347.586495
    '''
    conn = data.connection_context
    require_pal_usable(conn)
    complementary = arg('complementary', complementary, bool)

    col = arg('col', col, str)
    if col is None:
        col = data.columns[0]
    data_ = data[[col]]
    unique_id = str(uuid.uuid1()).replace('-', '_').upper()
    distr_tbl = _distr_table(conn, distr_info, unique_id)
    tables = ['RESULT']
    tables = ["#DISTRIBUTION_QUANTILE_{}_TBL_{}".format(tbl, unique_id) for tbl in tables]
    res_tbl = tables[0]
    param_rows = [('LOWER_UPPER', complementary, None, None)]
    try:
        call_pal_auto(conn,
                      'PAL_DISTRIBUTION_QUANTILE',
                      data_,
                      conn.table(distr_tbl),
                      ParameterTable().with_data(param_rows),
                      *tables)
    except dbapi.Error as db_err:
        logger.exception(str(db_err))
        try_drop(conn, tables)
        raise
    return conn.table(res_tbl)

def distribution_fit(data, distr_type, optimal_method=None, censored=False):
    '''
    This algorithm aims to fit a probability distribution for a variable according
    to a series of measurements to the variable. There are many probability distributions
    of which some can be fitted more closely to the observed variable than others.

    Parameters
    ----------

    data : DataFrame
        DataFrame containing the data.

    distr_type : {'exponential', 'gamma', 'normal', 'poisson', 'uniform', 'weibull'}
        Specify the type of distribution to fit.

    optimal_method : {'maximum_likelihood', 'median_rank'}, optional
        Specifies the estimation method.

        Defaults to 'median_rank' when ``distr_type`` is 'weibull', 'maximum_likelihood' otherwise.

    censored : bool, optional
        Specify if ``data`` is censored of not.

        Only valid when ``distr_type`` is 'weibull'.

        Default to False.

    Returns
    -------
    DataFrame
        Fitting results, structured as follows:

          - NAME: name of distribution parameters.
          - VALUE: value of distribution parameters.

        Fitting statistics, structured as follows:

          - STAT_NAME: name of statistics.
          - STAT_VALUE: value of statistics.

    Examples
    --------
    Original data:

    >>> df.collect()
             DATA
        0    71.0
        1    83.0
        2    92.0
        3   104.0
        4   120.0
        5   134.0
        6   138.0
        7   146.0
        8   181.0
        9   191.0
        10  206.0
        11  226.0
        12  276.0
        13  283.0
        14  291.0
        15  332.0
        16  351.0
        17  401.0
        18  466.0

    Perform the function:

    >>> res, stats = distribution_fit(data, distr_type, optimal_method='maximum_likelihood')
    >>> res.collect()
                       NAME    VALUE
        0  DISTRIBUTIONNAME  WEIBULL
        1             SCALE    244.4
        2             SHAPE  2.06698
    >>> stats.collect()
        Empty DataFrame
        Columns: [STAT_NAME, STAT_VALUE]
        Index: []
    '''
    conn = data.connection_context
    require_pal_usable(conn)
    optimal_method_map = {'maximum_likelihood': 0, 'median_rank': 1}
    distribution_map = {'exponential': 'EXPONENTIAL', 'gamma': 'GAMMA', 'normal': 'NORMAL',
                        'poisson': 'POISSON', 'uniform': 'UNIFORM', 'weibull': 'WEIBULL'}
    distr_type = arg('distr_type', distr_type, distribution_map)
    optimal_method = arg('optimal_method', optimal_method, optimal_method_map)
    censored = arg('censored', censored, bool)
    if not censored and len(data.columns) != 1:
        msg = "The input dataframe must have and only have one column when censored is not set."
        logger.error(msg)
        raise ValueError(msg)
    if censored is True and len(data.columns) != 2:
        msg = "The input censored dataframe must have two columns."
        logger.error(msg)
        raise ValueError(msg)
    if distr_type != 'WEIBULL' and censored is True:
        msg = ('Censored data is only acceptable when distribution is set as weibull.')
        logger.error(msg)
        raise ValueError(msg)
    func_name = 'PAL_DISTRIBUTION_FIT' if not censored else 'PAL_DISTRIBUTION_FIT_CENSORED'
    unique_id = str(uuid.uuid1()).replace('-', '_').upper()

    tables = ['RESULT', 'STATISTICS']
    tables = ["#DISTRIBUTION_FIT_{}_TBL_{}".format(tbl, unique_id) for tbl in tables]
    res_tbl, stats_tbl = tables
    param_rows = [('DISTRIBUTIONNAME', None, None, distr_type),
                  ('OPTIMAL_METHOD', optimal_method, None, None)]
    try:
        call_pal_auto(conn,
                      func_name,
                      data,
                      ParameterTable().with_data(param_rows),
                      *tables)
    except dbapi.Error as db_err:
        logger.exception(str(db_err))
        try_drop(conn, tables)
        raise
    return conn.table(res_tbl), conn.table(stats_tbl)
