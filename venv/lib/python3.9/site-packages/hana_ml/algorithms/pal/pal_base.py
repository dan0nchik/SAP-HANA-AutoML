"""
PAL-specific helper functionality.
"""

import uuid
import inspect
import json

from hdbcli import dbapi
import hana_ml.ml_base
import hana_ml.ml_exceptions

from hana_ml.dataframe import quotename, DataFrame
from hana_ml.algorithms.pal import sqlgen

from hana_ml.model_storage_services import ModelSavingServices

# Expose most contents of ml_base in pal_base for import convenience.
# pylint: disable=unused-import
# pylint: disable=attribute-defined-outside-init
# pylint: disable=bare-except
# pylint: disable=too-many-locals
# pylint: disable=too-many-arguments
from hana_ml.ml_base import (
    Table,
    INTEGER,
    DOUBLE,
    NVARCHAR,
    NCLOB,
    arg,
    create,
    materialize,
    try_drop,
    parse_one_dtype,
    execute_logged,
    colspec_from_df,
    ListOfStrings,
    ListOfTuples,
    TupleOfIntegers,
    _TEXT_TYPES,
    _INT_TYPES,
)
from hana_ml.algorithms.pal.sqlgen import ParameterTable

MINIMUM_HANA_VERSION_PREFIX = '2.00.030'

_SELECT_HANA_VERSION = ("SELECT VALUE FROM SYS.M_SYSTEM_OVERVIEW " +
                        "WHERE NAME='Version'")
_SELECT_PAL = "SELECT * FROM SYS.AFL_PACKAGES WHERE PACKAGE_NAME='PAL'"
_SELECT_PAL_PRIVILEGE = (
    "SELECT * FROM SYS.EFFECTIVE_ROLES " +
    "WHERE USER_NAME=CURRENT_USER AND " +
    "ROLE_SCHEMA_NAME IS NULL AND "
    "ROLE_NAME IN ('AFL__SYS_AFL_AFLPAL_EXECUTE', " +
    "'AFL__SYS_AFL_AFLPAL_EXECUTE_WITH_GRANT_OPTION')"
)

def pal_param_register():
    """
    Register PAL parameters after PAL object has been initialized.
    """
    frame = inspect.currentframe()
    params = frame.f_back.f_locals
    try:
        params.pop('self')
    except KeyError:
        pass
    try:
        params.pop('functionality')
    except KeyError:
        pass
    serializable_params = {}
    for param_key, param_value in params.items():
        try:
            json.dumps(param_value)
            serializable_params[param_key] = param_value
        except:
            pass
    return serializable_params

class PALBase(hana_ml.ml_base.MLBase, ModelSavingServices):
    """
    Subclass for PAL-specific functionality.
    """
    # pylint: disable=too-few-public-methods
    def __init__(self, conn_context=None):
        super(PALBase, self).__init__(conn_context)
        ModelSavingServices.__init__(self)
        self.execute_statement = None

    def _call_pal(self, funcname, *tablenames):
        call_pal(self.conn_context, funcname, *tablenames)

    def _call_pal_auto(self, funcname, *args):
        self.execute_statement = call_pal_auto(self.conn_context, funcname, *args)

    def load_model(self, model):
        """
        Function to load fitted model.

        Parameters
        ----------
        model : DataFrame
            HANA DataFrame for fitted model.
        """
        self.model_ = model

    def _load_model_tables(self, schema_name, model_table_names, name, version, conn_context=None):
        """
        Function to load models.
        """
        if conn_context is None:
            conn_context = self.conn_context
        if isinstance(model_table_names, str):
            self.model_ = conn_context.table(model_table_names, schema=schema_name)
        elif isinstance(model_table_names, list):
            self.model_ = []
            for model_name in model_table_names:
                self.model_.append(conn_context.table(model_name, schema=schema_name))
        else:
            raise ValueError('Cannot load the model table. Unknwon values ({}, \
            {})'.format(schema_name, str(model_table_names)))

    def add_attribute(self, attr_key, attr_val):
        """
        Function to add attribute.
        """
        setattr(self, attr_key, attr_val)

def attempt_version_comparison(minimum, actual):
    """
    Make our best guess at checking whether we have a high-enough version.

    This may not be a reliable comparison. The version number format has
    changed before, and it may change again. It is unclear what comparison,
    if any, would be reliable.

    Parameters
    ----------
    minimum : str
        (The first three components of) the version string for the
        minimum acceptable HANA version.
    actual : str
        The actual HANA version string.

    Returns
    -------
    bool
        True if (we think) the version is okay.
    """
    truncated_actual = actual.split()[0]
    min_as_ints = [int(x) for x in minimum.split('.')]
    actual_as_ints = [int(x) for x in truncated_actual.split('.')]
    return min_as_ints <= actual_as_ints

def require_pal_usable(conn):
    """
    Raises an error if no compatible PAL version is usable.

    To pass this check, HANA must be version 2 SPS 03,
    PAL must be installed, and the user must have one of the roles
    required to execute PAL procedures (AFL__SYS_AFL_AFLPAL_EXECUTE
    or AFL__SYS_AFL_AFLPAL_EXECUTE_WITH_GRANT_OPTION).

    A successful result is cached, to avoid redundant checks.

    Parameters
    ----------
    conn : ConnectionContext
        ConnectionContext on which PAL must be available.

    Raises
    ------
    hana_ml.ml_exceptions.PALUnusableError
        If the wrong HANA version is installed, PAL is uninstalled,
        or PAL execution permission is unavailable.
    """
    # pylint: disable=protected-access
    if not conn._pal_check_passed:
        with conn.connection.cursor() as cur:
            # Check HANA version. (According to SAP note 1898497, this
            # should match the PAL version.)
            cur.execute(_SELECT_HANA_VERSION)
            hana_version_string = cur.fetchone()[0]

            if not attempt_version_comparison(
                    minimum=MINIMUM_HANA_VERSION_PREFIX,
                    actual=hana_version_string):
                template = ('hana_ml version {} PAL support is not ' +
                            'compatible with this version of HANA. ' +
                            'HANA version must be at least {!r}, ' +
                            'but actual version string was {!r}.')
                msg = template.format(hana_ml.__version__,
                                      MINIMUM_HANA_VERSION_PREFIX,
                                      hana_version_string)
                raise hana_ml.ml_exceptions.PALUnusableError(msg)

            # Check PAL installation.
            cur.execute(_SELECT_PAL)
            if cur.fetchone() is None:
                raise hana_ml.ml_exceptions.PALUnusableError('PAL is not installed.')

            # Check required role.
            cur.execute(_SELECT_PAL_PRIVILEGE)
            if cur.fetchone() is None:
                msg = ('Missing needed role - PAL procedure execution ' +
                       'needs role AFL__SYS_AFL_AFLPAL_EXECUTE or ' +
                       'AFL__SYS_AFL_AFLPAL_EXECUTE_WITH_GRANT_OPTION')
                raise hana_ml.ml_exceptions.PALUnusableError(msg)
        conn._pal_check_passed = True

def call_pal(conn, funcname, *tablenames):
    """
    Call a PAL function.

    Parameters
    ----------
    conn : ConnectionContext
        HANA connection.
    funcname : str
        PAL procedure name.
    tablenames : list of str
        Table names to pass to PAL.
    """
    # This currently takes function names as "PAL_KMEANS".
    # Should that just be "KMEANS"?

    # callproc doesn't seem to handle table parameters.
    # It looks like we have to use execute.

    # In theory, this function should only be called with function and
    # table names that are safe without quoting.
    # We quote them anyway, for a bit of extra safety in case things
    # change or someone makes a typo in a call site.
    header = 'CALL _SYS_AFL.{}('.format(quotename(funcname))
    arglines_nosep = ['    {}'.format(quotename(tabname))
                      for tabname in tablenames]
    arglines_string = ',\n'.join(arglines_nosep)
    footer = ') WITH OVERVIEW'
    call_string = '{}\n{}\n{}'.format(header, arglines_string, footer)

    # SQLTRACE
    conn.sql_tracer.trace_object({
        'name':funcname,
        'schema': '_SYS_AFL',
        'type': 'pal'
    }, sub_cat='function')

    with conn.connection.cursor() as cur:
        execute_logged(cur, call_string, conn.sql_tracer) # SQLTRACE added sql_tracer

def anon_block_safe(*dataframes):
    """
    Checks if these dataframes are compatible with call_pal_auto.

    Parameters
    ----------
    df1, df2, ... : DataFrame
        DataFrames to be fed to PAL.

    Returns
    -------
    bool
        True if call_pal_auto can be used.
    """
    # pylint:disable=protected-access
    return all(df._ttab_handling in ('safe', 'ttab') for df in dataframes)

def call_pal_auto(conn, funcname, *args):
    """
    Uses an anonymous block to call a PAL function.

    DataFrames that are not known to be safe in anonymous blocks will be
    temporarily materialized.

    Parameters
    ----------
    conn : ConnectionContext
        HANA connection to use.
    funcname : str
        Name of the PAL function to execute. Should not include the "_SYS_AFL."
        part.
    arg1, arg2, ..., argN : DataFrame, ParameterTable, or str
        Arguments for the PAL function, in the same order the PAL function
        takes them.
        DataFrames represent input tables, a ParameterTable object represents
        the parameter table, and strings represent names for output
        tables. Output names with a leading "#" will produce local temporary
        tables.
    """
    adjusted_args = list(args)
    temporaries = []
    unknown_indices = []

    def materialize_at(i):
        "Materialize the i'th element of adjusted_args."
        tag = str(uuid.uuid4()).upper().replace('-', '_')
        name = '#{}_MATERIALIZED_INPUT_{}'.format(funcname, tag)
        adjusted_args[i] = adjusted_args[i].save(name)
        temporaries.append(name)

    def try_exec(cur, sql):
        """
        Try to execute the given sql. Returns True on success, False if
        execution fails due to an anonymous block trying to read a local
        temporary table. Other exceptions are propagated.
        """
        try:
            execute_logged(cur, sql, conn.sql_tracer) # SQLTRACE added sql_tracer
            return True
        except dbapi.Error as err:
            if not err.errortext.startswith(
                    'feature not supported: Cannot use local temporary table'):
                raise
            return False

    try:
        for i, argument in enumerate(args):
            if isinstance(argument, DataFrame):
                # pylint: disable=protected-access
                if argument._ttab_handling == 'unknown':
                    unknown_indices.append(i)
                elif argument._ttab_handling == 'unsafe':
                    materialize_at(i)

        # SQLTRACE added sql_tracer
        sql = sqlgen.call_pal_tabvar(funcname, conn.sql_tracer, *adjusted_args)

        # SQLTRACE
        conn.sql_tracer.trace_object({
            'name':funcname,
            'schema': '_SYS_AFL',
            'type': 'pal'
        }, sub_cat='function')

        # Optimistic execution.
        with conn.connection.cursor() as cur:
            if try_exec(cur, sql):
                # Optimistic execution succeeded, meaning all arguments with
                # unknown ttab safety are safe.
                for i in unknown_indices:
                    adjusted_args[i].declare_lttab_usage(False)
                return sql

        # If we reach this point, optimistic execution failed.

        if len(unknown_indices) == 1:
            # Only one argument of unknown ttab safety, so that one needs
            # materialization.
            adjusted_args[unknown_indices[0]].declare_lttab_usage(True)
            materialize_at(unknown_indices[0])
        else:
            # Multiple arguments of unknown safety. Test which ones are safe.
            for i in unknown_indices:
                with conn.connection.cursor() as cur:
                    ttab_used = not try_exec(cur, sqlgen.safety_test(adjusted_args[i]))
                adjusted_args[i].declare_lttab_usage(ttab_used)
                if ttab_used:
                    materialize_at(i)

        # SQLTRACE added sql_tracer
        sql = sqlgen.call_pal_tabvar(funcname, conn.sql_tracer, *adjusted_args)
        with conn.connection.cursor() as cur:
            execute_logged(cur, sql, conn.sql_tracer) # SQLTRACE added sql_tracer
        return sql
    finally:
        try_drop(conn, temporaries)
