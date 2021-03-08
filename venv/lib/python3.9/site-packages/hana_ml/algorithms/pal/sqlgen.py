"""
Utilities for generating SQL to cal PAL procedures.
"""

#pylint: disable=invalid-name, line-too-long, no-else-return
import itertools
import operator
import textwrap
import sys

from functools import wraps
from hana_ml.dataframe import DataFrame, quotename
from hana_ml.ml_base import colspec_from_df, Table, INTEGER, DOUBLE, NVARCHAR

if sys.version_info.major == 2:
    #pylint: disable=undefined-variable
    _INT_TYPES = (int, long)
    _TEXT_TYPES = (str, unicode)
else:
    _INT_TYPES = (int,)
    _TEXT_TYPES = (str,)

DECLARE_PARAM_ARRAYS = textwrap.dedent('''\
    DECLARE param_name VARCHAR(5000) ARRAY;
    DECLARE int_value INTEGER ARRAY;
    DECLARE double_value DOUBLE ARRAY;
    DECLARE string_value VARCHAR(5000) ARRAY;
    ''')

ONE_PARAM_ROW_TEMPLATE = textwrap.dedent('''\
    param_name[{i}] := {name};
    int_value[{i}] := {ival};
    double_value[{i}] := {dval};
    string_value[{i}] := {sval};
    ''')

UNNEST = 'params = UNNEST(:param_name, :int_value, :double_value, :string_value);\n'

def literal(value):
    """
    Return a SQL literal representing the given value.

    Parameters
    ----------
    value : int, float, string, or None
        Python equivalent of the desired SQL value.

    Returns
    -------
    str
        String representing the SQL equivalent of the given value.
    """
    # Eventually, we'll probably need something like this functionality
    # in the public API. We're leaving this private for now to try to
    # avoid locking ourselves into design decisions we might regret, like
    # whether a string becomes a VARCHAR or NVARCHAR literal (or even a
    # VARBINARY expression), or how we handle the types of numeric values.
    if value is None:
        return 'NULL'
    elif isinstance(value, _INT_TYPES):
        return str(value)
    elif isinstance(value, float):
        return repr(value)
    elif isinstance(value, _TEXT_TYPES):
        # This will need better unicode handling eventually. I'm not sure
        # how to do that best, given that SAP HANA has VARCHAR, NVARCHAR, and
        # VARBINARY as 3 distinct types while Python (2 or 3) has only 2
        # corresponding types, bytestrings and unicode strings.
        # For now, for how we're using this function, NVARCHAR is fine.
        return "N'{}'".format(value.replace("'", "''"))
    else:
        raise TypeError("Unexpected value of type {}".format(type(value)))

def create_params(param_rows):
    """
    Return SQL to build a parameter table (variable) from the given rows.

    Parameters
    ----------

    param_rows : list of tuple
        Data rows for a parameter table.

    Returns
    -------

    str
        SQL text that would generate a table variable named "params"
        with the given rows.
    """
    populate_params = []
    for i, (name, ival, dval, sval) in enumerate(param_rows, 1):
        # int(operator.index(ival)) works around the fact that booleans don't
        # count as ints in SQL. Passing booleans as ints in query parameters
        # works fine in execute(), but using True in SQL where an int
        # is needed doesn't work. This is awkward, and there may be
        # a better option. Possibly query parameters.
        # (The operator.index call rejects things that don't "behave like"
        # ints, and the int call converts int subclasses (like bool) to int.)
        name, ival, dval, sval = map(literal, [
            name,
            ival if ival is None else int(operator.index(ival)),
            dval,
            sval
        ])
        populate_params.append(ONE_PARAM_ROW_TEMPLATE.format(
            i=i, name=name, ival=ival, dval=dval, sval=sval))
    return ''.join([DECLARE_PARAM_ARRAYS] + populate_params + [UNNEST])

def tabletype(df):
    """
    Express a DataFrame's type in SQL.

    Parameters
    ----------

    df : DataFrame
        DataFrame to take the type of.

    Returns
    -------

    str
        "TABLE (...)" SQL string representing the table type.
    """
    spec = colspec_from_df(df)
    return 'TABLE ({})'.format(', '.join(
        quotename(name)+' '+sqltype for name, sqltype in spec))

SAFETY_TEST_TEMPLATE = textwrap.dedent('''\
    DO BEGIN
    IF 0=1 THEN
        {};
    END IF;
    END''')

def safety_test(df):
    """
    Return SQL to test a dataframe's temporary-table-safety.

    If the dataframe needs local temporary tables, executing this SQL
    will throw an error.

    Parameters
    ----------

    df : DataFrame
        DataFrame to test.

    Returns
    -------

    str
        A do-nothing anonymous block that includes the argument
        dataframe's select statement inside a never-executed IF.
    """
    return SAFETY_TEST_TEMPLATE.format(df.select_statement)

def call_pal_tabvar(name, sql_tracer, *args): # SQLTRACE added sql_tracer
    """
    Return SQL to call a PAL function, using table variables.

    Parameters
    ----------

    name : str
        Name of the PAL function to execute. Should not include the "_SYS_AFL."
        part.
    arg1, arg2, ..., argN : DataFrame, ParameterTable, or str
        Arguments for the PAL function, in the same order the PAL function
        takes them.
        DataFrames represent input tables, a ParameterTable object represents
        the parameter table, and strings represent names for output
        tables. Output names with a leading "#" will produce local temporary
        tables.

    Returns
    -------

    str
        SQL string that would call the PAL function with the given inputs
        and place output in tables with the given output names.
    """
    # pylint:disable=too-many-locals,too-many-branches
    if not isinstance(name, _TEXT_TYPES):
        raise TypeError("name argument not a string - it may have been omitted")

    in_palarg_gen = ('in_'+str(i) for i in itertools.count())
    out_palarg_gen = ('out_'+str(i) for i in itertools.count())

    pal_args = []
    assign_dfs = []
    arg_dfs = []
    output_names = []
    param_rows = None

    # pylint:disable=protected-access
    for arg in args:
        if isinstance(arg, DataFrame):
            in_name = next(in_palarg_gen)
            record = (in_name, arg)
            if arg._ttab_handling in ('safe', 'unknown'):
                assign_dfs.append(record)
            elif arg._ttab_handling == 'ttab':
                arg_dfs.append(record)
            else:
                raise ValueError("call_pal_tabvar can't handle input DataFrame.")
            pal_args.append(':'+in_name)
            # SQLTRACE
            sql_tracer.trace_object({
                'name': in_name, # Generated name is unknown
                'auto_name': in_name,
                'table_type': arg.generate_table_type(),
                'select': arg.select_statement
            }, sub_cat='auto')
        elif isinstance(arg, ParameterTable):
            if param_rows is not None:
                # I know there are a few PAL functions with no parameter table,
                # such as PAL_CHISQUARED_GOF_TEST, but I don't know any with
                # multiple parameter tables. We can adjust this if we need to.
                raise TypeError('Multiple parameter tables not supported')
            param_rows = arg.data if arg.data is not None else []
            pal_args.append(':params')
        elif isinstance(arg, _TEXT_TYPES):
            output_names.append(arg)
            pal_args.append(next(out_palarg_gen))
            # SQLTRACE Getting mapping of out variable to internal table name
            sql_tracer.trace_object({
                'auto_name': 'out_'+str(len(output_names)-1),
                'name': arg,
                'table_type': 'auto',
                'select': None
            }, sub_cat='auto')
        else:
            raise TypeError('Unexpected argument type {}'.format(type(arg)))

    if arg_dfs:
        header = 'DO ({})\nBEGIN\n'.format(',\n    '.join(
            'IN {} {} => {}'.format(argname, tabletype(df), df._ttab_reference)
            for argname, df in arg_dfs))
    else:
        header = 'DO\nBEGIN\n'

    if param_rows is not None:
        param_table_creation = create_params(param_rows)
    else:
        param_table_creation = ''
    input_assignments = ''.join('{} = {};\n'.format(argname, df_in.select_statement)
                                for argname, df_in in assign_dfs)
    invocation = 'CALL _SYS_AFL.{}({});\n'.format(name, ', '.join(pal_args))
    extract_outputs = ''.join(
        'CREATE {}COLUMN TABLE {} AS (SELECT * FROM :out_{});\n'.format(
            'LOCAL TEMPORARY ' if output_name.startswith('#') else '',
            quotename(output_name),
            i
        ) for i, output_name in enumerate(output_names)
    )

    return (
        header
        + param_table_creation
        + input_assignments
        + invocation
        + extract_outputs
        + 'END\n'
    )

PARAMETER_TABLE_SPEC = [
    ("PARAM_NAME", NVARCHAR(5000)),
    ("INT_VALUE", INTEGER),
    ("DOUBLE_VALUE", DOUBLE),
    ("STRING_VALUE", NVARCHAR(5000))
]

class ParameterTable(Table):
    """
    Represents a PAL parameter table to be created on SAP HANA.
    """
    def __init__(self, name=None):
        super(ParameterTable, self).__init__(name, PARAMETER_TABLE_SPEC)
    def with_data(self, data):
        """
        Like Table.with_data, but filters out rows with no parameter value.

        Parameters
        ----------

        data : list of tuple
            PAL parameter table rows. Rows where the only non-None element
            is the parameter name will be automatically removed.

        Returns
        -------

        ParameterTable
            New ParameterTable with data.
        """
        filtered_data = [param for param in data
                         if any(val is not None for val in param[1:])]
        return super(ParameterTable, self).with_data(filtered_data)

def trace_sql(func):
    """
    SQL tracer for PAL functions.
    """
    @wraps(func)
    def function_with_sql_tracing(*args, **kwargs):
        # SQLTRACE
        if len(args) > 1:
            conn = args[1].connection_context
        else:
            conn = None
            if kwargs.get('data') is None:
                conn = args[0].conn_context
            else:
                conn = kwargs.get('data').connection_context
        conn.sql_tracer.set_sql_trace(args[0], args[0].__class__.__name__, func.__name__.lower().replace('_', ''))
        return func(*args, **kwargs)
    return function_with_sql_tracing
