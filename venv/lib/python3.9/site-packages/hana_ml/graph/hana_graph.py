"""
This module represents a database set of HANA Dataframes that are
the edge and vertex tables of a HANA Graph.

The following classes and functions are available:

    * :class:`Graph`
    * :func:`create_hana_graph_from_existing_workspace`
    * :func:`create_hana_graph_from_vertex_and_edge_frames`
    * :func:`discover_graph_workspaces`

"""
#pylint: disable=line-too-long
#pylint: disable=too-many-locals
#pylint: disable=too-many-arguments
#pylint: disable=unnecessary-lambda
#pylint: disable=too-many-instance-attributes
#pylint: disable=useless-object-inheritance
#pylint: disable=invalid-name
#pylint: disable=no-else-raise
#pylint: disable=logging-format-interpolation
#pylint: disable=too-many-lines
#pylint: disable=too-many-branches
#pylint: disable=too-many-statements

import random
import string
import logging
import pandas as pd
import numpy as np
from hdbcli import dbapi
from hana_ml.dataframe import create_dataframe_from_pandas, DataFrame

# HANA Graph variables and reserved words
EDGE_ID = "EDGE_ID"
G_SCHEMA = "SCHEMA_NAME"
GWS_NAME = "WORKSPACE_NAME"
V_SCHEMA = "VERTEX_SCHEMA_NAME"
V_TABLE = "VERTEX_TABLE_NAME"
E_SCHEMA = "EDGE_SCHEMA_NAME"
E_TABLE = "EDGE_TABLE_NAME"
EDGE_SOURCE = "EDGE_SOURCE_COLUMN_NAME"
EDGE_TARGET = "EDGE_TARGET_COLUMN"
EDGE_KEY = "EDGE_KEY_COLUMN_NAME"
VERTEX_KEY = "VERTEX_KEY_COLUMN_NAME"
CREATE_TIME_STAMP = "CREATE_TIMESTAMP"
USER_NAME = "USER_NAME"
IS_VALID = "IS_VALID"
COLUMN_NAME = "COLUMN_NAME"
DIRECTIONS = ['OUTGOING', 'INCOMING', 'ANY']
DEF_DIR = DIRECTIONS[0]
logger = logging.getLogger(__name__)


def _shortest_path_procedure(
        cur, schema, proc_name, vertex_columns, edge_columns, vertex_dtype, workspace, weight, vertex_select, edge_select,
        start_vertex, end_vertex, direction):
    """
    Factored out of the class based function so that it can be used in public anonymous block functions. In this case it
    would be called with a signature based on shortest_path('Graph_name', source, target).

    Parameters
    ----------
    cur : ConnectionContext.connection.cursor
        Cursor based on the HANA data source used to execute sql in scope of shortest path.
    schema : str
        Name of the schema.
    proc_name : str
        Name of the procedure that can be used to clean it up.
    vertex_columns : str
        Key, value pairs of column names and data types representing nodes/vertices in an sql format.
    edge_columns : str
        Key, value pairs of column names and data types representing edges in an sql format.
    vertex_dtype : str
        Data type of the vertex key column.
    workspace : str
        Name of HANA graph workspace.
    weight : str
        Variable for column name to which to apply the weight.
    vertex_select : str
        Select statement for the nodes/vertices.
    edge_select : str
        Select statement for the edges.
    start_vertex : str
        ID of the source node/vertex from which the path is determined.
    end_vertex : str
        ID of the target node/vertex to which the path is determined.
    direction : str
            OUTGOING, INCOMING, or ANY which determines the algorithm results.

    Returns
    -------
    tuple
        Results of cursor running the procedure which returns the scalars.
    """
    # SQL for creating the result tables.
    sql = '''CREATE TYPE "{schema}"."TT_VERTICES_SP_{proc_name}" AS TABLE ({vertex_columns}, "VERTEX_ORDER" BIGINT);'''.format(
        schema=schema,
        proc_name=proc_name,
        vertex_columns=vertex_columns,
    )
    cur.execute(sql)
    sql = '''CREATE TYPE "{schema}"."TT_EDGES_SP_{proc_name}" AS TABLE ({edge_columns}, "EDGE_ORDER" BIGINT);'''.format(
        schema=schema,
        proc_name=proc_name,
        edge_columns=edge_columns,
    )
    cur.execute(sql)
    # SQL variable for weight in procedure.
    if weight:
        weighted_definition = '''WeightedPath<DOUBLE> p = Shortest_Path(:g, :v_start, :v_end, (Edge e) => DOUBLE{{ return :e."{weight}"; }}, :i_direction);'''.format(
            weight=weight)
    else:
        weighted_definition = '''WeightedPath<BIGINT> p = Shortest_Path(:g, :v_start, :v_end, :i_direction);'''
    sql = '''
    CREATE OR REPLACE PROCEDURE "{schema}"."GS_SP_{proc_name}"(
    IN i_startVertex {vertex_dtype}, IN i_endVertex {vertex_dtype}, IN i_direction NVARCHAR(10), 
    OUT o_path_weight DOUBLE, OUT o_vertices "{schema}"."TT_VERTICES_SP_{proc_name}", 
    OUT o_edges "{schema}"."TT_EDGES_SP_{proc_name}")
    LANGUAGE GRAPH READS SQL DATA AS
    BEGIN
    GRAPH g = Graph("{schema}", "{workspace}");
    VERTEX v_start = Vertex(:g, :i_startVertex);
    VERTEX v_end = Vertex(:g, :i_endVertex);
    {weighted_definition}
    o_path_weight = DOUBLE(WEIGHT(:p));
    o_vertices = SELECT {vertex_select}, :VERTEX_ORDER FOREACH v IN Vertices(:p) WITH ORDINALITY AS VERTEX_ORDER;
    o_edges = SELECT {edge_select}, :EDGE_ORDER FOREACH e IN Edges(:p) WITH ORDINALITY AS EDGE_ORDER;
    END;
    '''.format(
        schema=schema,
        proc_name=proc_name,
        vertex_dtype=vertex_dtype,
        workspace=workspace.upper(),
        weighted_definition=weighted_definition,
        vertex_select=vertex_select,
        edge_select=edge_select)
    cur.executemany(sql)
    return cur.callproc('"{schema}"."GS_SP_{proc_name}"'.format(schema=schema, proc_name=proc_name),
                        (start_vertex, end_vertex, direction, None, None, None))


def discover_graph_workspaces(connection_context):
    """
    Provide a view of the Graph Workspaces (GWS) on a given connection to SAP HANA. This provides the basis for creating
    a HANA graph from existing GWS instead of only creating them from vertex and edge tables. Use the SYS SQL
    provided for Graph Workspaces so a user can create a HANA graph from one of them. The SQL returns the following
    per GWS:
        SCHEMA_NAME, WORKSPACE_NAME, CREATE_TIMESTAMP, USER_NAME, EDGE_SCHEMA_NAME, EDGE_TABLE_NAME,
        EDGE_SOURCE_COLUMN_NAME, EDGE_TARGET_COLUMN_NAME, EDGE_KEY_COLUMN_NAME, VERTEX_SCHEMA_NAME, VERTEX_TABLE_NAME,
        VERTEX_KEY_COLUMN_NAME, IS_VALID.

    Due to the differences in Cloud and On-Prem Graph workspaces, the sql creation requires different methods to derive
    the same summary pattern for GWS as defined above. For this reason, 2 internal functions return the summary.

    Parameters
    ----------
    connection_context : ConnectionContext
        Connection to the given SAP HANA Database and implied Graph Workspace.

    Returns
    -------
    list
        The list of tuples returned by fetchall but with headers included and as a dict.

    """
    CLOUD = "CLOUD"
    ONPREM = "ONPREM"

    DISCOVER_CLOUD = '''
    SELECT GW.*, GWC.* 
    FROM SYS.GRAPH_WORKSPACES AS GW
    LEFT OUTER JOIN SYS.GRAPH_WORKSPACE_COLUMNS AS GWC
    ON GW.SCHEMA_NAME = GWC.SCHEMA_NAME AND GW.WORKSPACE_NAME = GWC.WORKSPACE_NAME
    ORDER BY GW.SCHEMA_NAME, GW.WORKSPACE_NAME, GWC.ENTITY_TYPE, GWC.ENTITY_ROLE_POSITION;
    '''

    DISCOVER_ON_PREM = '''
    SELECT * FROM SYS.GRAPH_WORKSPACES
    '''
    def _cloud_or_on_prem():
        v_str = connection_context.hana_version()
        if 'CE' in v_str[v_str.find('('):v_str.find(')')] or int(v_str[0]) > 3:
            return CLOUD
        return ONPREM

    if not connection_context:
        raise Exception('Cannot determine HANA and version information. Connection is not set.')
    with connection_context.connection.cursor() as cur:
        try:
            # Check if the current connected version is cloud or on-prem to decide the proper SQL to run
            if _cloud_or_on_prem() == CLOUD:
                # Gets G_SCHEMA(0), WORKSPACE(1), ENTIY_TYPE(2), E_SCH(3), E_TBL(4), ROLE(5), COLUMN(6), IS_VALID(7)
                cur.execute(DISCOVER_CLOUD)
                res = cur.fetchall()
                # Storage for aggregated results
                summary = []
                # Start with the first schema and graphworkspace
                if len(res) < 1:
                    raise ValueError("There are no graph workspaces in this schema")
                i_schema = res[0].column_names.index('SCHEMA_NAME')
                i_isvalid = res[0].column_names.index('IS_VALID')
                i_wrkspace = res[0].column_names.index('WORKSPACE_NAME')
                i_ent_type = res[0].column_names.index('ENTITY_TYPE')
                i_ent_role = res[0].column_names.index('ENTITY_ROLE')
                i_sch_name = res[0].column_names.index('ENTITY_SCHEMA_NAME')
                i_tbl_name = res[0].column_names.index('ENTITY_TABLE_NAME')
                i_col_name = res[0].column_names.index('ENTITY_COLUMN_NAME')
                cur_sch_wks = "{}{}".format(res[0][0], res[0][1])
                cur_gws = {G_SCHEMA: res[0][i_schema], GWS_NAME: res[0][i_wrkspace], IS_VALID: res[0][i_isvalid]}
                for row in res:
                    # Iterate results to combine the workspace descriptors into a single row for the summary
                    if "{}{}".format(row[i_schema], row[i_wrkspace]) == cur_sch_wks:
                        if row[i_ent_type] == "VERTEX" and row[i_ent_role] == "KEY":
                            cur_gws[V_SCHEMA] = row[i_sch_name]
                            cur_gws[V_TABLE] = row[i_tbl_name]
                            cur_gws[VERTEX_KEY] = row[i_col_name]
                        elif row[i_ent_type] == "EDGE" and row[i_ent_role] == "KEY":
                            cur_gws[E_SCHEMA] = row[i_sch_name]
                            cur_gws[E_TABLE] = row[i_tbl_name]
                            cur_gws[EDGE_KEY] = row[i_col_name]
                        elif row[i_ent_type] == "EDGE" and row[i_ent_role] == "SOURCE":
                            cur_gws[EDGE_SOURCE] = row[i_col_name]
                        elif row[i_ent_type] == "EDGE" and row[i_ent_role] == "TARGET":
                            cur_gws[EDGE_TARGET] = row[i_col_name]
                    else:
                        summary.append(cur_gws)
                        cur_sch_wks = "{}{}".format(row[i_schema], row[i_wrkspace])
                        cur_gws = {G_SCHEMA: row[i_schema], GWS_NAME: row[i_wrkspace]}
                # Append the last started since the append only occurs if the current is different than the last.
                summary.append(cur_gws)
            else:
                cur.execute(DISCOVER_ON_PREM)
                res = cur.fetchall()
                summary = [{
                    G_SCHEMA: gws[0],
                    GWS_NAME: gws[1],
                    CREATE_TIME_STAMP: gws[2],
                    USER_NAME: gws[3],
                    E_SCHEMA: gws[4],
                    E_TABLE: gws[5],
                    EDGE_SOURCE: gws[6],
                    EDGE_TARGET: gws[7],
                    EDGE_KEY: gws[8],
                    V_SCHEMA: gws[9],
                    V_TABLE: gws[10],
                    VERTEX_KEY: gws[11],
                    IS_VALID: gws[12]
                } for gws in res]
            return pd.DataFrame(summary)
        except dbapi.Error:
            raise Exception('Unable to get HANA version.'
                            ' please check with your database administrator')
        except dbapi.ProgrammingError:
            logger.error('No result set')


def create_hana_graph_from_vertex_and_edge_frames(
        connection_context, vertices_hdf, edges_hdf, workspace_name, schema=None, edge_source_column='from',
        edge_target_column='to', edge_key_column=None, vertex_key_column=None, object_type_as_bin=False, drop_exist_tab=True,
        force=True, replace=True, geo_cols=None, srid=None):
    """
    Expects either a hana dataframe or pandas dataframe as input for each vertex and edges table, thus the function
    expects an n_frame and e_frame. If it is pandas then it will be transformed into a hana_ml.DataFrame.

    If it is an hdf then the graph only needs to be created by combining existing tables. No collecting.
    If it is a pdf then the tables expect to need to be created and then the gws from those resulting tables.
    If it is just an edge list then the graph workspace needs to be created from a notional vertices table and the edge content.

    Parameters
    ----------
    connection_context : ConnectionContext
        The connection to the SAP HANA system.
    vertices_hdf : pandas.DataFrame or hana_ml.DataFrame
        Table of data containing vertices and their keys that correspond with the edge frame.
    edges_hdf : pandas.DataFrame or hana_ml.DataFrame
        Table of data containing edges that link keys within the vertex frame.
    workspace_name : str
        Name of the workspace expected in the SAP HANA Graph workspaces of the ConnectionContext.
    schema : str
        Schema name of the workspace.

        Defaults to ConnectionContext current_schema.
    edge_source_column : str
        Column name in the e_frame containing only source vertex keys that exist within the vertex_key_column of the n_frame.

        Defaults to 'from'.
    edge_target_column : str
        Column name in the e_frame containing the unique id of the edge.

        Defaults to 'to'.
    edge_key_column : str
        Column name in the n_frame containing the vertex key which uniquely identifies the vertex in the edge table.

        Defaults to None.
    vertex_key_column : str
        Column name in the n_frame containing the vertex key which uniquely identifies the vertex in the edge table.

        Defaults to None.
    object_type_as_bin : bool, optional
        If True, the object type will be considered CLOB in SAP HANA.

        Defaults to False.
    drop_exist_tab : bool, optional
        If force is True, drop the existing table when drop_exist_tab is True and truncate the existing table when it is
        False.

        Defaults to False.
    force : bool, optional
        If force is True, then the SAP HANA table with table_name is truncated or dropped.

        Defaults to False.
    replace : bool, optional
        If replace is True, then the SAP HANA table performs the missing value handling.

        Defaults to True.
    geo_cols : dict, (optional but required for Spatial functions)
        Tuples or strings as keys that denote columns with geometric data and srid as values.

        Defaults to None.

    srid : int, (optional for Spatial functions)
        Spatial reference system id.

        Defaults to None.

    Returns
    -------
    Graph
        A virtual HANA Graph with functions inherited from the individual vertex and edge HANA Dataframes.

    Examples
    --------
    >>> n_path = os.path.join(self.datasets, 'nodes.csv')
    >>> e_path = os.path.join(self.datasets, 'edges.csv')
    >>> vertex_key_column = 'guid'
    >>> edge_from_col = 'from'
    >>> edge_to_col = 'to'
    >>> edge_label = 'label'
    >>> nodes = pd.read_csv(n_path)
    >>> edges = pd.read_csv(e_path)
    >>> # Create the hana_graph based on the 2 csv
    >>> hana_graph = create_hana_graph_from_vertex_and_edge_frames(
    >>>  connection_context=ConnectionContext(), vertices_hdf=nodes, edges_hdf=edges, workspace_name=workspace_name,
    >>>  schema='SYSTEM', vertex_key_column=vertex_key_column, edge_source_column=edge_from_col, edge_target_column=edge_to_col,
    >>>  object_type_as_bin=False, drop_exist_tab=True)

    """
    if not schema:
        schema = connection_context.get_current_schema()
    if isinstance(vertices_hdf, pd.DataFrame) and isinstance(edges_hdf, pd.DataFrame):
        # Use the 2 pandas dataframes as vertex and edges tables to create a workspace and return the collect statements
        vertex_table_name = "{}_VERTICES".format(workspace_name)
        edge_table_name = "{}_EDGES".format(workspace_name)
        # Check geo_cols and whether it references a column in the node or edges table
        if isinstance(geo_cols, list):
            if srid:
                geo_cols = {col: srid for col in geo_cols}
            else:
                raise ValueError("SRID required if sending a list of columns")
        if isinstance(geo_cols, dict):
            v_geo_cols = {col: geo_cols[col] for col in geo_cols if col in vertices_hdf.columns}
            e_geo_cols = {col: geo_cols[col] for col in geo_cols if col in edges_hdf.columns}
        elif geo_cols is None:
            v_geo_cols = None
            e_geo_cols = None
        else:
            raise TypeError("Geometry provided was a {} when a dict or list is expected".format(type(geo_cols)))
        vertices = create_dataframe_from_pandas(
            connection_context, pandas_df=vertices_hdf, schema=schema, replace=replace, force=force,
            table_name=vertex_table_name, object_type_as_bin=object_type_as_bin, drop_exist_tab=drop_exist_tab,
            primary_key=vertex_key_column, geo_cols=v_geo_cols
        )
        # If there is no edge_col_key then assign one called EDGE_ID and base values on a row sequence for id
        if not edge_key_column:
            edge_key_column = EDGE_ID
            edges_hdf.insert(loc=0, column=EDGE_ID, value=np.arange(len(edges_hdf)))
        # Create the Edge table within the same schema but not as its own Dataframe
        edges = create_dataframe_from_pandas(
            connection_context, pandas_df=edges_hdf, schema=schema, table_name=edge_table_name, geo_cols=e_geo_cols,
            force=force, drop_exist_tab=drop_exist_tab, replace=replace, primary_key=edge_key_column,
            not_nulls=[edge_key_column, edge_source_column, edge_target_column])

    elif isinstance(edges_hdf, pd.DataFrame):
        logger.info("Creating graph only from edge list not available yet")
        raise ValueError("Creating graph only from edge list not available yet")
    elif isinstance(vertices_hdf, DataFrame) and isinstance(edges_hdf, DataFrame):
        # Create a view on the HANA Dataframes to prevent any unintended changes to source tables
        vertex_table_name = '{}_VIEW'.format(vertices_hdf.source_table['TABLE_NAME'])
        edge_table_name = '{}_VIEW'.format(edges_hdf.source_table['TABLE_NAME'])
        try:
            connection_context.connection.cursor().execute("DROP VIEW {};".format(vertex_table_name))
        except dbapi.Error:
            pass
        try:
            connection_context.connection.cursor().execute("DROP VIEW {};".format(edge_table_name))
        except dbapi.Error:
            pass
        vertices = vertices_hdf.save(where=vertex_table_name, table_type='VIEW', force=True)
        vertices.geo_cols = vertices_hdf.geo_cols
        edges = edges_hdf.save(where=edge_table_name, table_type='VIEW', force=True)
        edges.geo_cols = edges_hdf.geo_cols
    else:
        raise ValueError("An edges and vertices definition are required.")

    hana_graph = Graph(
        connection_context=connection_context,
        schema=schema,
        vertices_hdf=vertices,
        edges_hdf=edges,
        vertex_tbl_name=vertex_table_name,
        edge_tbl_name=edge_table_name,
        vertex_key_column=vertex_key_column,
        edge_key_column=edge_key_column,
        edge_source_column=edge_source_column,
        edge_target_column=edge_target_column,
        workspace_name=workspace_name
    )

    return hana_graph


def create_hana_graph_from_existing_workspace(connection_context, workspace_name, schema=None):
    """
    Given a workspace name that is assumed to exist within the connection_context, return a Graph as if created from
    external data sources.

    Parameters
    ----------
    connection_context : ConnectionContext
        The connection to the SAP HANA system.
    workspace_name : str
        Case sensitive name of the HANA Graph workspace.
    schema : str, optional
        Schema name of the workspace.

        Defaults to ConnectionContext current_schema.

    Returns
    -------
    Graph
        A virtual HANA Graph with functions inherited from the individual vertex and edge HANA Dataframes.

    """
    if not schema:
        schema = connection_context.get_current_schema()
    # Get the workspaces in the given connection context and ensure the named space is included.
    summary = discover_graph_workspaces(connection_context)
    meta = [graph for index, graph in summary.iterrows() if graph['WORKSPACE_NAME'] == workspace_name]
    if len(meta) < 1:
        raise ValueError("No graph workspace found with name {}".format(workspace_name))
    else:
        meta = meta[0]
        vertex_table_reference = '"{}"."{}"'.format(meta['VERTEX_SCHEMA_NAME'],
                                                    meta['VERTEX_TABLE_NAME'])
        edge_table_reference = '"{}"."{}"'.format(meta['EDGE_SCHEMA_NAME'],
                                                  meta['EDGE_TABLE_NAME'])
        return Graph(
            connection_context=connection_context,
            schema=schema,
            vertices_hdf=DataFrame(connection_context, select_statement='SELECT * FROM {}'.format(vertex_table_reference)),
            edges_hdf=DataFrame(connection_context, select_statement='SELECT * FROM {}'.format(edge_table_reference)),
            vertex_tbl_name=meta['VERTEX_TABLE_NAME'],
            edge_tbl_name=meta['EDGE_TABLE_NAME'],
            vertex_key_column=meta['VERTEX_KEY_COLUMN_NAME'],
            edge_key_column=meta['EDGE_KEY_COLUMN_NAME'],
            edge_source_column=meta['EDGE_SOURCE_COLUMN_NAME'],
            edge_target_column=meta['EDGE_TARGET_COLUMN'],
            workspace_name=meta['WORKSPACE_NAME']
        )


class _Path(object):
    """
    Represents a group of vertices and edges that define a path. Used by the Graph object when running functions that
    return more than a single set of values. Prevents the user from requiring verbose syntax to access results. For
    example, returning a dictionary from the functions would require a user to write path['edges'] instead of
    path.edges().

    Parameters
    ----------
    edges : pd.DataFrame
        Path defined by a collection of source and target columns with vertex keys values expected in those columns.

    vertices : pd.DataFrame
        Vertices that are included in the path.

    """

    def __init__(self, edges, vertices, weight=None):
        self.edge_table = edges
        self.vertex_table = vertices
        self._weight = weight

    def edges(self):
        """

        Returns
        -------
        pd.Dataframe

        """
        return self.edge_table

    def vertices(self):
        """

        Returns
        -------
        pd.Dataframe

        """
        return self.vertex_table

    def weight(self):
        """

        Returns
        -------
        float

        """
        return self._weight


class Graph(object):#pylint: disable=too-many-public-methods
    """
    Represents a graph consisting of a vertex and edges table that was created from a set of pandas dataframes,
    existing tables that are changed into a graph workspace, or through an existing graph workspace.

    Parameters
    ----------
    connection_context : ConnectionContext
        The connection to the SAP HANA system.
    schema : str
        Name of the schema.
    vertices_hdf : hana_ml.Dataframe
        Table object representing vertices derived from a set of tables.
    edges_hdf : hana_ml.Dataframe
        Table object representing edges derived from a set of tables.
    vertex_tbl_name : str
        Name that references the source table for vertices.
    edge_tbl_name : str
        Name that references the source table for edges.
    vertex_key_column : str
        Name of the column containing the unique identifier for nodes/vertices.
    edge_key_column : str
        Name of the column containing the unique identifier for edges/links.
    edge_source_column : str
        Name that references the column where the keys for the source in an edge are located.
    edge_target_column : str
        Name that references the column where the keys for the target in an edge are located.
    workspace_name : str
        Name that references the HANA Graph workspace.

    """

    def __init__(self, connection_context, schema, vertices_hdf, edges_hdf, vertex_tbl_name, edge_tbl_name, vertex_key_column,
                 edge_key_column, edge_source_column, edge_target_column, workspace_name):

        self.connection_context = connection_context
        self.schema = schema
        self.vertices_hdf = vertices_hdf
        self.edges_hdf = edges_hdf
        self.vertex_tbl_name = vertex_tbl_name
        self.edge_tbl_name = edge_tbl_name
        self.vertex_key_column = vertex_key_column
        vertex_dt = [dtype[1] for dtype in self.vertices_hdf.dtypes() if dtype[0] == self.vertex_key_column][0]
        if vertex_dt == 'NVARCHAR':
            vertex_dt = 'NVARCHAR(5000)'
        self.vertex_key_col_dtype = vertex_dt
        self.edge_key_column = edge_key_column
        self.edge_source_column = edge_source_column
        self.edge_target_column = edge_target_column
        self.workspace_name = workspace_name
        self._create_graph()

    @staticmethod
    def _get_random_string(k=8):
        return ''.join(random.choices(string.ascii_uppercase, k=k))

    def _vertex_columns(self):
        _cols = []
        for dtype in self.vertices_hdf.dtypes():
            if dtype[0] == self.vertex_key_column:
                if dtype[1].upper() == 'NVARCHAR':
                    _cols.append('"{}" {}(5000)'.format(dtype[0], dtype[1].upper()))
                else:
                    _cols.append('"{}" {}'.format(dtype[0], dtype[1].upper()))
        return ', '.join(_cols)

    def _edge_columns(self, valid_edge):
        _cols = []
        ', '.join(['"{}" {}'.format(dtype[0], dtype[1].upper()) for dtype in self.edges_hdf.dtypes() if
                   dtype[0] in valid_edge])
        for dtype in self.edges_hdf.dtypes():
            if dtype[0] in valid_edge:
                if dtype[1].upper() == 'NVARCHAR':
                    _cols.append('"{}" {}(5000)'.format(dtype[0], dtype[1].upper()))
                else:
                    _cols.append('"{}" {}'.format(dtype[0], dtype[1].upper()))
        return ', '.join(_cols)

    def _clean_up(self, include_vertices=True, include_edges=True):
        # Remove all the associated procedures and tables as required
        try:
            self.connection_context.connection.cursor().execute(
                '''DROP GRAPH WORKSPACE "{}"."{}" '''.format(self.schema, self.workspace_name))
        except dbapi.Error as error:
            if 'invalid graph workspace name:' in error.errortext:
                pass
            else:
                logger.error(error.errortext)
        if include_edges:
            try:
                self.connection_context.connection.cursor().execute("DROP TABLE {}".format(self.edge_tbl_name))
            except dbapi.Error as error:
                if 'invalid table' in error.errortext:
                    pass
                else:
                    logger.error(error.errortext)
        if include_vertices:
            try:
                self.connection_context.connection.cursor().execute("DROP TABLE {}".format(self.vertex_tbl_name))
            except dbapi.Error as error:
                if 'invalid table' in error.errortext:
                    pass
                else:
                    logger.error(error.errortext)

    def _clean_proc(self, artifacts):
        for art in artifacts:
            self.connection_context.connection.cursor().execute('''DROP {}'''.format(art))

    def _create_graph(self):
        """
        Explicitly create the graph workspace.

        Returns
        -------

        """
        sql = 'DROP GRAPH WORKSPACE "{}"."{}"'.format(self.schema, self.workspace_name)
        try:
            self.connection_context.connection.cursor().execute(sql)
        except dbapi.Error as error:
            if 'invalid graph workspace name:' in error.errortext:
                pass
            else:
                logger.error(error.errortext)
        sql = '''
        CREATE GRAPH WORKSPACE "{schema}"."{workspace_name}"
        EDGE TABLE "{schema}"."{edge_table}" 
        SOURCE COLUMN "{source_column}" 
        TARGET COLUMN "{target_column}" 
        KEY COLUMN "{edge_id}"
        VERTEX TABLE "{schema}"."{vertex_table}" 
        KEY COLUMN "{vertex_key_column}"
        '''.format(workspace_name=self.workspace_name,
                   schema=self.schema,
                   edge_table=self.edge_tbl_name,
                   edge_id=self.edge_key_column,
                   vertex_table=self.vertex_tbl_name,
                   source_column=self.edge_source_column,
                   target_column=self.edge_target_column,
                   vertex_key_column=self.vertex_key_column)
        try:
            self.connection_context.connection.cursor().execute(sql)
        except dbapi.Error as error:
            logger.error(error.errortext)

    @staticmethod
    def _spatial_transform(hdf, pdf):
        """
        If the hana dataframe has geo_cols, transform the cols into pandas ready format. Used internally when a pandas
        dataframe is returned without using collect().

        Parameters
        ----------
        hdf : HANA Dataframe
            Checked for geo_cols.
        pdf : Pandas Dataframe
            Transformed if containing spatial data.

        Returns
        -------
        pd.Dataframe

        """
        if hdf.geo_cols:
            for geo in list(hdf.geo_cols.keys()):
                # Use the HANA Dataframe method typically used by hdf.collect() if there are geo cols.
                pdf = hdf._transform_geo_column(pandas_df=pdf, col=geo)     #pylint: disable=protected-access
        return pdf

    def _check_vertex_exists_by_key(self, vertices):
        """
        Change a list of vertex keys into a string and filter on them to check if they are in the graph. Raise a
        ValueError if any of the keys are not recognized in the vertex table. Edge case is possible where source tables
        are not up to date of the workspace.

        Parameters
        ----------
        vertices : list
            Vertex keys expected to be in the graph.

        Returns
        -------
        bool : True if the vertices exist otherwise ValueError raised.

        """
        vertex_str = ', '.join(["'{}'".format(vertex) for vertex in vertices])
        cur = self.connection_context.connection.cursor()
        cur.execute('''
        SELECT "{key}" FROM "{sch}"."{tbl}" where "{tbl}"."{key}" IN ({vertex_str})
        '''.format(key=self.vertex_key_column, sch=self.schema, tbl=self.vertex_tbl_name, vertex_str=vertex_str))
        vertex_check = cur.fetchall()
        if len(vertex_check) < len(vertices):
            missing = ', '.join(list(filter(
                lambda vertex_key: vertex_key not in [key[0] for key in vertex_check], [str(key) for key in vertices])))
            logger.error("{} not recognized key(s) in {}".format(missing, self.vertex_tbl_name))
            raise ValueError("{} not recognized key(s) in {}".format(missing, self.vertex_tbl_name))
        return True

    def vertices(self, vertex_key=None):
        """
        Get the table representing vertices within a graph. If there is a vertex, check it.

        Parameters
        ----------
        vertex_key : str, optional
            Vertex keys expected to be in the graph.

        Returns
        -------
        pd.Dataframe

        """
        if not vertex_key:
            pdf = self.vertices_hdf.collect()
        elif self._check_vertex_exists_by_key([vertex_key]):
            cur = self.connection_context.connection.cursor()
            cur.execute('''SELECT * FROM "{sch}"."{vertex_tbl}" WHERE "{v_key_col}" = '{v_key}' '''.format(
                sch=self.schema, vertex_tbl=self.vertex_tbl_name, v_key_col=self.vertex_key_column, v_key=vertex_key))
            pdf = pd.DataFrame(cur.fetchall(), columns=self.vertices_hdf.columns)
            pdf = self._spatial_transform(hdf=self.vertices_hdf, pdf=pdf)
        else:
            raise ValueError("Vertex key not recognized in the graph {}.".format(self.workspace_name))
        return pdf

    def edges(self, vertex_key=None, edge_key=None, direction=DEF_DIR):
        """
        Get the table representing edges within a graph. If there is a vertex_key, then only get the edges respective
        to that vertex.

        Parameters
        ----------
        vertex_key : str, optional
            Vertex key from which to get edges.

            Defaults to None.

        edge_key : str, optional
            Edge key from which to get edges.

            Defaults to None.

        direction : str, optional
            OUTGOING, INCOMING, or ANY which determines the algorithm results. Only applicable if vertex_key is not None.

            Defaults to OUTGOING.

        Returns
        -------
        pd.Dataframe

        """
        pdf = None
        if vertex_key:
            if self._check_vertex_exists_by_key([vertex_key]):
                if direction == 'ANY':
                    cur = self.connection_context.connection.cursor()
                    cur.execute('''SELECT * FROM "{sch}"."{wks}" WHERE "{src}" = '{v_key}' OR "{tgt}" = '{v_key}' '''.format(
                        sch=self.schema, wks=self.edge_tbl_name, src=self.edge_source_column, tgt=self.edge_target_column,
                        v_key=vertex_key))
                    pdf = pd.DataFrame(cur.fetchall(), columns=self.edges_hdf.columns)
                elif direction == 'INCOMING':
                    pdf = self.in_edges(vertex_key=vertex_key)
                elif direction == 'OUTGOING':
                    pdf = self.out_edges(vertex_key=vertex_key)
            else:
                raise ValueError("Vertex key not recognized in the graph {}.".format(self.workspace_name))
            # Take care of the geo_cols that aren't processed unless with the hdf collect method
            pdf = self._spatial_transform(hdf=self.edges_hdf, pdf=pdf)
        elif edge_key:
            cur = self.connection_context.connection.cursor()
            cur.execute(
                '''SELECT * FROM "{sch}"."{wks}" WHERE "{key_col}" = '{e_key}' '''.format(
                    sch=self.schema, wks=self.edge_tbl_name, key_col=self.edge_key_column, e_key=edge_key))
            result = pd.DataFrame(cur.fetchall(), columns=self.edges_hdf.columns)
            if result.size == 0:
                raise KeyError("No edge with {} {}".format(self.edge_key_column, edge_key))
            else:
                pdf = result
                # Take care of the geo_cols that aren't processed unless with the hdf collect method
                pdf = self._spatial_transform(hdf=self.edges_hdf, pdf=pdf)
        elif not vertex_key and not edge_key:
            pdf = self.edges_hdf.collect()
        return pdf

    def out_edges(self, vertex_key):
        """
        Get the table representing edges within a graph filtered on a vertex_key and its outgoing edges.

        Parameters
        ----------
        vertex_key : str
            Vertex key from which to get edges.

        Returns
        -------
        pd.Dataframe

        """
        if self._check_vertex_exists_by_key([vertex_key]):
            cur = self.connection_context.connection.cursor()
            cur.execute('''SELECT * FROM "{sch}"."{wks}" WHERE "{src}" = '{v_key}' '''.format(
                sch=self.schema, wks=self.edge_tbl_name, src=self.edge_source_column, v_key=vertex_key))
            return pd.DataFrame(cur.fetchall(), columns=self.edges_hdf.columns)
        else:
            raise ValueError("Vertex key not recognized in the graph {}.".format(self.workspace_name))

    def source(self, edge_key):
        """
        Get the vertex that is the source/from/origin/start point of an edge.

        Parameters
        ----------
        edge_key : str
            Edge key from which to get source vertex.

        Returns
        -------
        pd.Dataframe

        """
        cur = self.connection_context.connection.cursor()
        cur.execute('''SELECT "{src}" FROM "{sch}"."{edge_tbl}" WHERE "{e_key_col}" = '{e_key}' '''.format(
            src=self.edge_source_column, sch=self.schema, edge_tbl=self.edge_tbl_name, e_key_col=self.edge_key_column,
            e_key=edge_key))
        return self.vertices(vertex_key=cur.fetchone()[0])

    def target(self, edge_key):
        """
        Get the vertex that is the source/from/origin/start point of an edge.

        Parameters
        ----------
        edge_key : str
            Edge key from which to get source vertex.

        Returns
        -------
        pd.Dataframe

        """
        cur = self.connection_context.connection.cursor()
        cur.execute('''SELECT "{tgt}" FROM "{sch}"."{edge_tbl}" WHERE "{e_key_col}" = '{e_key}' '''.format(
            tgt=self.edge_target_column, sch=self.schema, edge_tbl=self.edge_tbl_name, e_key_col=self.edge_key_column,
            e_key=edge_key))
        return self.vertices(vertex_key=cur.fetchone()[0])

    def in_edges(self, vertex_key):
        """
        Get the table representing edges within a graph filtered on a vertex_key and its incoming edges.

        Parameters
        ----------
        vertex_key : str
            Vertex key from which to get edges.

        Returns
        -------
        pd.Dataframe

        """
        if self._check_vertex_exists_by_key([vertex_key]):
            cur = self.connection_context.connection.cursor()
            cur.execute('''SELECT * FROM "{sch}"."{wks}" WHERE "{tgt}" = '{v_key}' '''.format(
                sch=self.schema, wks=self.edge_tbl_name, tgt=self.edge_target_column, v_key=vertex_key))
            return pd.DataFrame(cur.fetchall(), columns=self.edges_hdf.columns)
        else:
            raise ValueError("Vertex key not recognized in the graph {}.".format(self.workspace_name))

    def neighbors_sub_graph(self, start_vertex, direction=None, min_depth=1, max_depth=1):
        """
        Get a virtual subset of the graph based on a start_vertex and all vertices within a min->max count of degrees of
        separation. The result is similar to get_neighbors but includes edges which could be useful for visualization.
        Create the procedure based on a random name that can be deleted after running and fetching results.

        No default direction . If not remove line 1037

        Parameters
        ----------
        start_vertex : str
            Source from which the subset is based.
        direction : str, optional
            OUTGOING, INCOMING, or ANY which determines the algorithm results.

            Defaults to None.
        min_depth : int, optional
            The count of degrees of separation from which to start considering neighbors.

            Defaults to 1.
        max_depth : int, optional
            The count of degrees of separation at which to end considering neighbors.

            Defaults to 1.

        Returns
        -------
        Path
            Class representing the Pandas Dataframes that resulted from the function.

        """
        # Check if direction is required, default to OUTGOING in graph script direction sql which doesn't require :i_dir
        if direction:
            dir_sql = '''MULTISET<Vertex> m_neighbors = Neighbors(:g, :v_start, :min_depth, :max_depth, :i_dir);'''
        else:
            direction = 'OUTGOING'
            dir_sql = '''MULTISET<Vertex> m_neighbors = Neighbors(:g, :v_start, :min_depth, :max_depth);'''
        if max_depth and min_depth:
            if max_depth < min_depth:
                raise ValueError("Max depth {} is greater than min depth {}".format(max_depth, min_depth))
        # Set procedure name and run validation/existence checks
        proc_name = "{}".format(self._get_random_string())
        self._check_vertex_exists_by_key([start_vertex])
        # Get the column data types for each table and create the vertices / edges graph_script tables if dtypes are valid for graph
        valid_edge = [self.edge_key_column, self.edge_target_column, self.edge_source_column]  # filter only required cols
        vertex_columns = self._vertex_columns()
        edge_columns = self._edge_columns(valid_edge=valid_edge)
        vertex_select = ', '.join([':v."{}"'.format(col) for col in self.vertices_hdf.columns if col == self.vertex_key_column])
        edge_select = ', '.join([':e."{}"'.format(dtype[0]) for dtype in self.edges_hdf.dtypes() if dtype[0] in valid_edge])
        if direction.upper() not in DIRECTIONS:
            raise KeyError("Direction needs to be one of {}".format(', '.join(DIRECTIONS)))
        cur = self.connection_context.connection.cursor()
        sql = '''CREATE TYPE "{schema}"."TT_VERTICES_NEI_{proc_name}" AS TABLE ({vertex_columns});'''.format(
            schema=self.schema, proc_name=proc_name, vertex_columns=vertex_columns)
        cur.execute(sql)
        sql = '''CREATE TYPE "{schema}"."TT_EDGES_NEI_{proc_name}" AS TABLE ({edge_columns});'''.format(
            schema=self.schema, proc_name=proc_name, edge_columns=edge_columns)
        cur.execute(sql)
        sql = '''
        CREATE OR REPLACE PROCEDURE "{schema}"."GS_NEIGHBORS_{proc_name}"(
        IN i_startVertex {vertex_key_col_dtype}, 
        IN min_depth BIGINT, 
        IN max_depth BIGINT, 
        IN i_dir VARCHAR(10), 
        OUT o_vertices "{schema}"."TT_VERTICES_NEI_{proc_name}", 
        OUT o_edges "{schema}"."TT_EDGES_NEI_{proc_name}")
        LANGUAGE GRAPH READS SQL DATA AS 
        BEGIN 
        GRAPH g = Graph("{schema}", "{workspace}"); 
        VERTEX v_start = Vertex(:g, :i_startVertex); 
        {dir_sql}
        o_vertices = SELECT {vertex_select} FOREACH v IN :m_neighbors; 
        MULTISET<Edge> m_edges = Edges(:g, :m_neighbors, :m_neighbors); 
        o_edges = SELECT {edge_select} FOREACH e IN :m_edges; 
        END;
        '''.format(
            schema=self.schema,
            proc_name=proc_name,
            vertex_key_col_dtype=self.vertex_key_col_dtype,
            workspace=self.workspace_name.upper(),
            dir_sql=dir_sql,
            vertex_select=vertex_select,
            edge_select=edge_select
        )
        cur.execute(sql)
        sql = '''CALL "{schema}"."GS_NEIGHBORS_{proc_name}"(i_startVertex => '{start_vertex}', min_depth => {min_depth}, max_depth => {max_depth}, i_dir => '{direction}', o_vertices => ?, o_edges => ?);
        '''.format(
            schema=self.schema,
            proc_name=proc_name,
            start_vertex=start_vertex,
            min_depth=min_depth,
            max_depth=max_depth,
            direction=direction
        )
        # set the artifacts which are cleaned afer the procedure is run here in the case it fails
        artifacts = ['PROCEDURE "{}"."GS_NEIGHBORS_{}"'.format(self.schema, proc_name),
                     'TYPE "{}"."TT_VERTICES_NEI_{}"'.format(self.schema, proc_name),
                     'TYPE "{}"."TT_EDGES_NEI_{}"'.format(self.schema, proc_name)
                     ]
        try:
            cur.executemany(sql)
        except dbapi.Error as err:
            self._clean_proc(artifacts=artifacts)
            raise RuntimeError(err.errortext)
        # Build the response from the result sets
        vertices = pd.DataFrame(
            cur.fetchall(), columns=[col for col in self.vertices_hdf.columns if col == self.vertex_key_column])
        cur.nextset()
        edges = pd.DataFrame(
            cur.fetchall(), columns=[col for col in self.edges_hdf.columns if col in valid_edge])
        self._clean_proc(artifacts=artifacts)
        return _Path(vertices=vertices, edges=edges)

    def neighbors(self, start_vertex, direction=None, min_depth=1, max_depth=1, include_edges=False):
        """
        Get a virtual subset of the graph based on a start_vertex and all vertices within a min->max count of degrees of
        separation. The result is similar to get_neighbors but only includes nodes/vertices not the edges.
        Create the procedure based on a random name that can be deleted after running and fetching results.

        Parameters
        ----------
        start_vertex : str
            Source from which the subset is based.
        direction: str, optional
            OUTGOING, INCOMING, or ANY which determines the algorithm results.

            Defaults to None.
        min_depth : int, optional
            The count of degrees of separation from which to start considering neighbors.

            Defaults to 1.
        max_depth : int, optional
            The count of degrees of separation at which to end considering neighbors.

            Defaults to 1.
        include_edges : bool, optional
            If the user requires edges then it is a sub_graph and that method is called. Shortcut for repeat usage.

            Defaults to False.

        Returns
        -------
        pd.Dataframe
            A Pandas Dataframe that contains the nodes/vertices representing the result of the algorithm.

        """
        if include_edges:
            return self.neighbors_sub_graph(start_vertex=start_vertex, direction=direction, min_depth=min_depth, max_depth=max_depth)
        # Check if direction is required, default to OUTGOING in graph script direction sql which doesn't require :i_dir
        if direction:
            dir_sql = '''MULTISET<Vertex> m_neighbors = Neighbors(:g, :v_start, :min_depth, :max_depth, :i_dir);'''
        else:
            direction = 'OUTGOING'
            dir_sql = '''MULTISET<Vertex> m_neighbors = Neighbors(:g, :v_start, :min_depth, :max_depth);'''
        # Set procedure name and run validation/existence checks
        proc_name = "{}".format(self._get_random_string())
        self._check_vertex_exists_by_key([start_vertex])
        vertex_columns = self._vertex_columns()
        vertex_select = ', '.join([':v."{}"'.format(col) for col in self.vertices_hdf.columns if col == self.vertex_key_column])
        if direction.upper() not in DIRECTIONS:
            raise KeyError("Direction needs to be one of {}".format(', '.join(DIRECTIONS)))
        cur = self.connection_context.connection.cursor()
        sql = '''CREATE TYPE "{schema}"."TT_VERTICES_NEI_{proc_name}" AS TABLE ({vertex_columns});'''.format(
            schema=self.schema, proc_name=proc_name, vertex_columns=vertex_columns)
        cur.execute(sql)
        sql = '''
        CREATE OR REPLACE PROCEDURE "{schema}"."GS_NEIGHBORS_{proc_name}"(
        IN i_startVertex {vertex_key_col_dtype}, 
        IN min_depth BIGINT, 
        IN max_depth BIGINT, 
        IN i_dir VARCHAR(10), 
        OUT o_vertices "{schema}"."TT_VERTICES_NEI_{proc_name}")
        LANGUAGE GRAPH READS SQL DATA AS 
        BEGIN 
        GRAPH g = Graph("{schema}", "{workspace}"); 
        VERTEX v_start = Vertex(:g, :i_startVertex); 
        {dir_sql}
        o_vertices = SELECT {vertex_select} FOREACH v IN :m_neighbors;
        END;
        '''.format(
            schema=self.schema,
            proc_name=proc_name,
            vertex_key_col_dtype=self.vertex_key_col_dtype,
            workspace=self.workspace_name.upper(),
            dir_sql=dir_sql,
            vertex_select=vertex_select
        )
        cur.execute(sql)
        sql = '''CALL "{schema}"."GS_NEIGHBORS_{proc_name}"(i_startVertex => '{start_vertex}', min_depth => {min_depth}, max_depth => {max_depth}, i_dir => '{direction}', o_vertices => ?);
        '''.format(
            schema=self.schema,
            proc_name=proc_name,
            start_vertex=start_vertex,
            min_depth=min_depth,
            max_depth=max_depth,
            direction=direction
        )
        # set the artifacts which are cleaned afer the procedure is run here in the case it fails
        artifacts = ['PROCEDURE "{}"."GS_NEIGHBORS_{}"'.format(self.schema, proc_name),
                     'TYPE "{}"."TT_VERTICES_NEI_{}"'.format(self.schema, proc_name)]
        try:
            cur.executemany(sql)
        except dbapi.Error as err:
            self._clean_proc(artifacts=artifacts)
            raise RuntimeError(err.errortext)
        results = cur.fetchall()
        self._clean_proc(artifacts=artifacts)

        return pd.DataFrame(
            results, columns=[col for col in self.vertices_hdf.columns if col == self.vertex_key_column])

    def shortest_path(self, source, target, weight=None, direction=DEF_DIR):
        """
        Given a source and target vertex_key with optional weight and direction, get the shortest path between them.
        Create the procedure, get the results, and then delete the procedure and types to avoid the need of a clean up.
        The procedure may fail for HANA versions prior to SP05 therefore a switch to determine the version is provided.
        The user can take the results and visualize them with libraries such as networkX using the result['edges'].

        Parameters
        ----------
        source : str
            vertex key from which the shortest path will start.
        target : str
            vertex key from which the shortest path will end.
        weight : str, optional
            Variable for column name to which to apply the weight.

            Defaults to None.
        direction : str, optional
            OUTGOING, INCOMING, or ANY which determines the algorithm results.

            Defaults to OUTGOING.

        Returns
        -------
        Path
            Class representing the Pandas Dataframes that resulted from the function.

        Examples
        --------
        >>> path1 = hana_graph.shortest_path(source='3', target='5', weight='rating')
        >>> nx_graph1 = nx.from_pandas_edgelist(path1.edges(), source=edge_source_column, target=edge_target_column)
        >>> fig3, ax3 = plt.subplots(1, 1, figsize=(80, 80))
        >>> nx.draw_networkx(nx_graph1, ax=ax3)

        """
        # Version check
        if int(self.connection_context.hana_major_version()) < 4:
            raise EnvironmentError("SAP HANA version is not compatible with this method")
        # Set procedure name and run validation/existence checks
        proc_name = "{}".format(self._get_random_string())
        self._check_vertex_exists_by_key([source, target])
        if direction.upper() not in DIRECTIONS:
            raise KeyError("Direction needs to be one of {}".format(', '.join(DIRECTIONS)))
        cur = self.connection_context.connection.cursor()

        # Get the column data types for each table and create the vertices / edges graph_script tables if dtypes are valid for graph
        valid_edge = [self.edge_key_column, self.edge_target_column, self.edge_source_column]  # filter only required cols
        vertex_columns = self._vertex_columns()
        edge_columns = self._edge_columns(valid_edge=valid_edge)
        vertex_select = ', '.join([':v."{}"'.format(dtype[0]) for dtype in self.vertices_hdf.dtypes() if dtype[0] == self.vertex_key_column])
        edge_select = ', '.join([':e."{}"'.format(dtype[0]) for dtype in self.edges_hdf.dtypes() if dtype[0] in valid_edge])

        artifacts = ['PROCEDURE "{}"."GS_SP_{}"'.format(self.schema, proc_name),
                     'TYPE "{}"."TT_EDGES_SP_{}"'.format(self.schema, proc_name),
                     'TYPE "{}"."TT_VERTICES_SP_{}"'.format(self.schema, proc_name)]
        # SQL to create and run procedure
        try:
            res = _shortest_path_procedure(
                cur=cur,
                schema=self.schema,
                proc_name=proc_name,
                vertex_columns=vertex_columns,
                edge_columns=edge_columns,
                vertex_dtype=self.vertex_key_col_dtype,
                workspace=self.workspace_name.upper(),
                weight=weight,
                vertex_select=vertex_select,
                edge_select=edge_select,
                start_vertex=source,
                end_vertex=target,
                direction=direction
            )
        except dbapi.Error as err:
            self._clean_proc(artifacts=artifacts)
            raise RuntimeError(err.errortext)

        # Build the response from the result sets
        vertex_cols = [col for col in self.vertices_hdf.columns if col == self.vertex_key_column]
        vertex_cols.append('VERTEX_ORDER')
        vertices = pd.DataFrame(
            cur.fetchall(), columns=vertex_cols)
        cur.nextset()
        edge_cols = [col for col in self.edges_hdf.columns if col in valid_edge]
        edge_cols.append('EDGE_ORDER')
        edges = pd.DataFrame(
            cur.fetchall(), columns=edge_cols)
        self._clean_proc(artifacts=artifacts)

        return _Path(vertices=vertices, edges=edges, weight=res[3])
