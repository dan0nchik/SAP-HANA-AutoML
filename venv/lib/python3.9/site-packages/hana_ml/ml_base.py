"""
This module contains the base class for ML classes, as well as
a number of helper functions and classes to support MLBase.
"""

import copy
import logging
import sys
import threading

from hdbcli import dbapi

from .dataframe import quotename

#pylint: disable=too-many-return-statements
#pylint: disable=line-too-long
#pylint: disable=unused-variable
__all__ = [
    'MLBase',
    'Table',
    'INTEGER',
    'DOUBLE',
    'NVARCHAR',
    'NCLOB',
    'arg',
    'create',
    'materialize',
    'try_drop',
]

logger = logging.getLogger(__name__) #pylint: disable=invalid-name

if sys.version_info.major == 2:
    #pylint: disable=undefined-variable
    _TEXT_TYPES = (str, unicode)
    _INT_TYPES = (int, long)
else:
    _TEXT_TYPES = (str,)
    _INT_TYPES = (int,)

ListOfStrings = object() #pylint: disable=invalid-name
ListOfTuples = object() #pylint: disable=invalid-name
TupleOfIntegers = object() #pylint: disable=invalid-name

class MLBase(object): #pylint: disable=useless-object-inheritance
    """
    Base class for HANA ML classes.

    Attributes
    ----------
    conn_context : ConnectionContext
        The connection to the HANA system.
    id : int
        Number unique to each instance, used in table names to prevent
        name collision.
    """

    # pylint: disable=too-few-public-methods

    _count = 0
    _count_lock = threading.Lock()

    def __init__(self, conn_context=None):
        self.conn_context = conn_context
        with MLBase._count_lock:
            # Most of our objects don't support concurrent access from
            # multiple threads, but it seems worthwhile to support
            # concurrent creation and use of *separate* objects.
            self.id = MLBase._count  # pylint: disable=invalid-name
            MLBase._count += 1

    @staticmethod
    def _arg(attr, value, constraint, required=False):
        """
        Validate and possibly transform an argument.

        See ``hana_ml.arg`` for full documentation.
        """
        return arg(attr, value, constraint, required)

    # Usage examples
    # self._create(ParameterTable("#WHATEVER_NAME").with_data(list_of_tuples))
    # output_spec = [
    #     ("COL1", INTEGER),
    #     ("COL2", DOUBLE),
    #     ("COL3", INTEGER),
    #     ("COL4", NVARCHAR(50))
    # ]
    # self._create(Table("#TABLE_NAME", output_spec))
    def _create(self, table, force=True):
        """
        Create a database table from a Table object.

        Local temporary column tables only.

        Parameters
        ----------
        table : Table
            Table definition.
        force : boolean, optional
            Whether to delete any existing table with the same name.
        """
        create(self.conn_context, table, force)

    def _try_drop(self, names):
        """
        Drop the given table or tables.

        Parameters
        ----------
        names : str or list of str
            Name(s) of the table or tables to drop.
        """
        try_drop(self.conn_context, names)

    def _materialize(self, name, df, force=True): # pylint: disable=invalid-name
        """
        Materialize a DataFrame into a table.

        Local temporary column tables only.

        Parameters
        ----------
        name : str
            Name of the new table.
        df : DataFrame
            DataFrame to copy into the new table.
        force : boolean, optional
            Whether to delete any existing table with the same name.
        """
        materialize(self.conn_context, name, df, force)

def arg(name, value, constraint, required=False): #pylint: disable=too-many-branches
    """
    Validate and possibly transform an argument.

    If `value` is None and `required` is False, _arg returns None.
    Otherwise, `value` must be an instance of the given type
    or a key of the given dict.
    If `constraint` is a dict, a non-None `value` must be a string.
    `value` is lowercased and replaced by constraint[value].
    As a special case, ints are accepted in place of floats, and on
    Python 2, longs and unicode are accepted in place of float and str.

    Parameters
    ----------
    name : str
        Parameter name. Used for error messages.
    value : object
        Argument value.
    constraint : type or dict
        Required type of the argument, or mapping from allowed values
        to corresponding HANA-side values.
    required : boolean, optional
        Whether the argument is a required argument. Defaults to False.

    Raises
    ------
    TypeError
        If `value` has the wrong type.
    ValueError
        If `value` is a string, but not a valid string for the
        constraint dict.

    Returns
    -------
    new_arg
        `arg`, or the result of transforming `arg`
    """
    if value is None and not required:
        return None

    if isinstance(constraint, type): #pylint: disable=no-else-return
        if isinstance(value, constraint):
            okay = True
        elif constraint is float and isinstance(value, _INT_TYPES):
            okay = True
        elif constraint is str and isinstance(value, _TEXT_TYPES):
            okay = True
        else:
            okay = False
        if not okay:
            template = ('Parameter {!r} must be of type {} - ' +
                        'actual value was {!r}')
            raise TypeError(template.format(
                name, constraint.__name__, value))
        return value
    elif isinstance(constraint, dict):
        if not isinstance(value, _TEXT_TYPES):
            template = ('Parameter {!r} must be a string - ' +
                        'actual value was {!r}')
            raise TypeError(template.format(name, value))
        value = value.lower()
        if value not in constraint:
            template = ('Parameter {!r} must be one of the following: ' +
                        '{!r} - actual value was {!r}')
            raise ValueError(template.format(name, sorted(constraint), value))
        return constraint[value]
    elif constraint is ListOfStrings:
        if isinstance(value, list) and all(isinstance(x, _TEXT_TYPES) for x in value): #pylint: disable=no-else-return
            return value
        else:
            raise TypeError('Parameter {!r} must be a list of strings.'.format(name))
    elif constraint is ListOfTuples:
        if isinstance(value, list) and all(isinstance(x, tuple) for x in value): #pylint: disable=no-else-return
            return value
        else:
            raise TypeError('Parameter {!r} must be a list of tuples.'.format(name))
    elif constraint is TupleOfIntegers:
        if isinstance(value, tuple) and all(isinstance(x, int) for x in value): #pylint: disable=no-else-return
            return value
        else:
            raise TypeError('Parameter {!r} must be a tuple of integers'.format(name))
    elif isinstance(constraint, tuple):
        for valid_type in constraint:
            if isinstance(value, valid_type):
                return value
        template = ('Parameter {!r} type must be one of the following: ' +
                    '{!r} - actual type was {!r}')
        raise ValueError(template.format(name, sorted(constraint),
                                         type(value)))
    else:
        raise TypeError('Bad constraint: {!r}'.format(constraint))

def create(conn, table, force=True):
    """
    Create a database table from a Table object.

    Local temporary column tables only.

    Parameters
    ----------
    conn : ConnectionContext
        HANA connection.
    table : Table
        Table definition.
    force : boolean, optional
        Whether to delete any existing table with the same name.
    """
    name, spec, data = table.name, table.spec, table.data
    with conn.connection.cursor() as cur:
        if force:
            try:
                 # SQLTRACE
                conn.sql_tracer.trace_object({
                    'name': name,
                    'table_type': spec
                }, sub_cat='internal_tables')
                # SQLTRACE added sql_tracer
                execute_logged(cur, sql_for_drop_table(name), conn.sql_tracer)
            except dbapi.Error:
                pass
        # SQLTRACE added sql_tracer
        execute_logged(cur, sql_for_create_table(name, spec), conn.sql_tracer)
        if data:
            # non-None, non-empty
            statement = sql_for_insert_values(name, data)
            logger.info("Calling executemany with SQL: %s\nand values: %s",
                        statement,
                        data)
            # SQLTRACE
            conn.sql_tracer.trace_sql_many(statement, data)

            cur.executemany(statement, data)

def try_drop(conn, names):
    """
    Drop the given table or tables.

    Parameters
    ----------
    conn : ConnectionContext
        HANA connection.
    names : str or list of str
        Name(s) of the table or tables to drop.
    """
    if isinstance(names, _TEXT_TYPES):
        names = [names]
    with conn.connection.cursor() as cur:
        for name in names:
            try:
                # SQLTRACE added sql_tracer
                execute_logged(cur, sql_for_drop_table(name), conn.sql_tracer)
            except dbapi.Error:
                pass

def materialize(conn, name, df, force=True): # pylint: disable=invalid-name
    """
    Materialize a DataFrame into a table.

    Local temporary column tables only.

    Parameters
    ----------
    conn : ConnectionContext
        HANA connection.
    name : str
        Name of the new table
    df : DataFrame
        DataFrame to copy into the new table
    force : boolean, optional
        Whether to delete any existing table with the same name.
    """
    statement = 'CREATE LOCAL TEMPORARY COLUMN TABLE {} AS ({})'.format(
        quotename(name), df.select_statement)
    with conn.connection.cursor() as cur:
        # SQLTRACE
        conn.sql_tracer.trace_object({
            'name': name,
            'table_type': df.generate_table_type(),
            'select': df.select_statement
        }, sub_cat='input_tables')

        if force:
            try:
                # SQLTRACE added sql_tracer
                execute_logged(cur, sql_for_drop_table(name), conn.sql_tracer)
            except dbapi.Error:
                pass
        execute_logged(cur, statement, conn.sql_tracer) # SQLTRACE added sql_tracer

def execute_logged(cursor, statement, sql_tracer=None): # SQLTRACE added sql_tracer
    """
    Execute a SQL statement and log that we did it.

    Parameters
    ----------
    cursor : hdbcli.dbapi.Cursor
        Database cursor to execute the statement through.
    statement : str
        SQL statement to execute.
    """
    logger.info("Executing SQL: %s", statement)

    # SQLTRACE
    if sql_tracer:
        sql_tracer.trace_sql(statement)

    cursor.execute(statement)

INTEGER = "INTEGER"
DOUBLE = "DOUBLE"
NCLOB = "NCLOB"

def nvarchar(size):
    """
    Returns a string representing an NVARCHAR type of the given size.

    Parameters
    ----------
    size : int
        Maximum length of the strings.

    Returns
    -------
    str
        String representation of the given NVARCHAR type.

    Examples
    --------
    >>> NVARCHAR(50)
    'NVARCHAR(50)'
    """
    return 'NVARCHAR({})'.format(size)

NVARCHAR = nvarchar

# Deliberately no VARCHAR for now - we can put it in if we find we're using it,
# but until then, leaving it out prevents accidentally writing VARCHAR instead
# of NVARCHAR.

def sql_for_create_table(name, spec):
    """
    Returns an SQL string to create the described table.

    Local temporary column tables only.

    Parameters
    ----------
    name : str
        Name of the table to create.
    spec : list of (str, str) tuples
        Each tuple represents the name and type of a column in the
        resulting table.

    Returns
    -------
    str
        SQL for creating the table.
    """
    header = "CREATE LOCAL TEMPORARY COLUMN TABLE {} (".format(quotename(name))
    columnlines_nosep = [
        '    {} {}'.format(quotename(colname), coltype)
        for (colname, coltype) in spec
    ]
    footer = ')'
    column_defs_string = ',\n'.join(columnlines_nosep)
    return '{}\n{}\n{}'.format(header, column_defs_string, footer)

def sql_for_insert_values(name, data):
    """
    Returns an SQL string for inserting data into a table.

    Uses query parameters rather than filling the data into the SQL.
    Requires at least one element in data, to determine the number of columns.

    Parameters
    ----------
    name : str
        Name of the table to insert to.
    data : list of tuple
        Data rows to insert into the table.

    Returns
    -------
    str
        SQL for inserting the data.
    """
    # Could be changed to take a Table instead of a name
    # and determine column information from that.
    return "INSERT INTO {} VALUES ({})".format(
        quotename(name),
        ', '.join(['?']*len(data[0]))
    )

def sql_for_drop_table(name):
    """
    Returns an SQL string that would drop the given table.

    Parameters
    ----------
    name : str
        The name of the table.

    Returns
    -------
    str
        SQL to perform the drop.
    """
    return "DROP TABLE {}".format(quotename(name))

class Table(object): #pylint: disable=useless-object-inheritance
    """
    Represents a table to be created on HANA.

    Attributes
    ----------
    name : str
        Table name.
    spec : list of tuple
        Column names and types.
    data : list of tuple, or None
        Initial data to populate the table with.
    """
    def __init__(self, name, spec):
        self.name = name
        self.spec = spec

        self.data = None

    def with_data(self, data):
        """
        Return a Table like this table, but with the given initial data.

        Parameters
        ----------
        data : list of tuple
            Initial data rows.

        Returns
        -------
        Table
            New Table with data.
        """
        newobj = copy.copy(self)
        newobj.data = data
        return newobj

    @staticmethod
    def like(name, df): # pylint: disable=invalid-name
        """
        Return a Table with column names and types based on a DataFrame.

        Parameters
        ----------
        name : str
            Table name
        df : DataFrame
            DataFrame to base the table structure on.

        Returns
        -------
        Table
            Table based on the DataFrame.
        """
        return Table(name, colspec_from_df(df))

    def with_leading_column(self, name, sqltype):
        """
        Copy this Table with an extra leading column.

        Parameters
        ----------
        name : str
            Column name.
        sqltype : str
            Column type.

        Returns
        -------
        Table
            Table with new leading column.
        """
        newobj = copy.copy(self)
        newobj.spec = [(name, sqltype)] + newobj.spec
        return newobj

    def with_trailing_column(self, name, sqltype):
        """
        Copy this Table with an extra trailing column.

        Parameters
        ----------
        name : str
            Column name.
        sqltype : str
            Column type.

        Returns
        -------
        Table
            Table with new trailing column.
        """
        newobj = copy.copy(self)
        newobj.spec = newobj.spec + [(name, sqltype)]
        return newobj

ONE_PARAM_TYPES = 'VARBINARY VARCHAR NVARCHAR ALPHANUM SHORTTEXT'.split()

def colspec_from_df(df): # pylint: disable=invalid-name
    """
    Create a column spec for a new Table, based on a DataFrame.

    Parameters
    ----------
    df : DataFrame
        DataFrame to base the spec on.

    Returns
    -------
    list of tuples
        Column name-type pairs.
    """
    # I'm not comfortable with relying on dtypes(). I don't trust the
    # type names in dtypes() to be the same as the type names we need
    # to use to create tables. For example, a column of type BOOLEAN
    # comes back as TINYINT, and I can't rule out the possibility of
    # getting type codes like NVARCHAR3. Also, INTEGER comes back as INT,
    # which is syntactically okay but inconsistent with our use of
    # INTEGER.

    # That said, I don't see a better option for now.
    return [parse_one_dtype(dtype) for dtype in df.dtypes()]

def parse_one_dtype(dtype):
    """
    Parse a name-type pair from one element of a DataFrame's dtypes().

    Parameters
    ----------
    dtype : tuple
        Column name, type, and length.

    Returns
    -------
    tuple
        Column name and type, with length incorporated into the type.
    """
    name, sqltype, size, d_arg1, d_arg2, d_arg3 = dtype
    if sqltype == 'DECIMAL':
        return (name, '{}({}, {})'.format(sqltype, d_arg2, d_arg3))
    if sqltype in ONE_PARAM_TYPES:
        return (name, '{}({})'.format(sqltype, size))
    return (name, sqltype)
