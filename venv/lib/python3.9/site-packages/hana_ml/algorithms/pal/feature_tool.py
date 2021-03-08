"""
This module supports to generate additional features for HANA DataFrame.
"""
from hana_ml import dataframe
from hana_ml.dataframe import quotename

#pylint: disable=invalid-name
#pylint: disable=eval-used
#pylint: disable=unused-variable
#pylint: disable=line-too-long
#pylint: disable=too-many-arguments

def generate_feature(data,
                     target,
                     group_by=None,
                     agg_func=None,
                     trans_func=None,
                     order_by=None,
                     trans_param=None):
    """
    Add additional features to the existing dataframe using agg_func and trans_func.

    Parameters
    ----------
    data : DataFrame
        SAP HANA DataFrame.
    target : str
        The column in data to be feature engineered.
    group_by : str
        The column in data for group by when performing agg_func.
    agg_func : str
        HANA aggeration operations. SUM, COUNT, MIN, MAX, ...
    trans_func : str
        HANA transformation operations. MONTH, YEAR, LAG, ...
    order_by : str
        LEAD, LAG function requires an OVER(ORDER_BY) window specification.
    trans_param : int or tuple
        Parameters for transformation operations.
    Returns
    -------

    DataFrame
        SAP HANA DataFrame with new features.

    Examples
    --------

    >>> feat_df = generate_feature(data, "ds", agg_func="max", group_by="y")
    """
    view_sql = data.select_statement
    if agg_func is not None:
        if group_by is None:
            raise Exception("group_by cannot be None!")
        agg_keyword = '"{}({})"'.format(agg_func, target)
        agg_sql = "SELECT {}, {}({}) {} FROM ({}) GROUP BY {}".format(quotename(group_by),\
             agg_func, quotename(target), agg_keyword, view_sql, quotename(group_by))
        view_sql = "SELECT T1.*, T2.{} FROM ({}) T1 INNER JOIN ({}) T2 ON T1.{}=T2.{}"\
            .format(agg_keyword, view_sql, agg_sql, quotename(group_by), quotename(group_by))
    trans_params = ''
    if trans_param is not None:
        if isinstance(trans_param, tuple or list):
            trans_params = ", " + ", ".join(trans_param)
        else:
            trans_params = ", " + str(trans_param)
    if trans_func is not None:
        trans_keyword = '"{}({})"'.format(trans_func, target)
        if order_by is not None:
            view_sql = "SELECT *, {0}({1}{5}) OVER(ORDER BY {2}) AS {3} FROM ({4})".format(trans_func,
                                                                                           quotename(target),
                                                                                           quotename(order_by),
                                                                                           trans_keyword,
                                                                                           view_sql,
                                                                                           trans_params)
        else:
            view_sql = "SELECT *, {0}({1}) AS {2} FROM ({3})".format(trans_func, quotename(target), trans_keyword, view_sql)
    return dataframe.DataFrame(data.connection_context, view_sql)
