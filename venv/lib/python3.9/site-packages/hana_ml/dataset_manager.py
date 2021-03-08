"""
This module provides the features of **dataset manager**.

All these features are accessible to the end user via a single class:

    * :class:`DatasetManager`
    * :func:`is_mem_loaded`
"""
#pylint: disable=line-too-long
#pylint: disable=too-many-arguments
import logging
import time
import uuid
from hana_ml.dataframe import quotename, DataFrame, data_manipulation

logger = logging.getLogger(__name__) #pylint: disable=invalid-name

def is_mem_loaded(connection_context, table, schema):
    """
    Return load/unload status.

    Parameters
    ----------
    connection_context : ConnectionContext
        The connection object to an SAP HANA database.
    table : str
        Table name.
    schema : str
        Schema name.
    """
    return DataFrame(connection_context, "SELECT * FROM M_CS_TABLES")\
                .select('SCHEMA_NAME', 'TABLE_NAME', 'LOADED')\
                .filter("TABLE_NAME='{}'".format(table))\
                .filter("SCHEMA_NAME='{}'".format(schema))\
                .collect().iat[0, 2]

def is_load_unit_page(connection_context, table, schema):
    """
    Check the load unit whether it is page or column.

    Parameters
    ----------
    connection_context : ConnectionContext
        The connection object to an SAP HANA database.
    table : str
        Table name.
    schema : str
        Schema name.
    """
    return 'NSE' if (DataFrame(connection_context, "SELECT * FROM M_CS_TABLES")\
                    .select('SCHEMA_NAME', 'TABLE_NAME', 'LOAD_UNIT')\
                    .filter("TABLE_NAME='{}'".format(table))\
                    .filter("SCHEMA_NAME='{}'".format(schema))\
                    .collect().iat[0, 2]).upper() == 'PAGE' else 'MEM'

class DatasetManager:
    """
    Dataset manager.

    Parameters
    ----------
    connection_context : ConnectionContext
        The connection object to an SAP HANA database.
    schema : str, optional
        The schema name where the dataset meta table locates.
        Defaults to the current schema.
    meta : str, optional
        The meta table name.
        Defaults to "HANAMLAPI_DATASET_MANAGER_META_TBL".
    force : str, optional
        False to drop the existing meta table.
        Defaults to False.
    """
    def __init__(self, connection_context, schema=None, meta=None, force=False):
        self.connection_context = connection_context
        if schema is None:
            schema = connection_context.get_current_schema()
        self.schema = schema
        if meta is None:
            meta = "HANAMLAPI_DATASET_MANAGER_META_TBL"
        self.meta = meta
        self._create_meta(force)

    def _create_meta(self, force=False):
        """
        Creates the meta table.
        """
        exist_meta = self.connection_context.has_table(table=self.meta, schema=self.schema)
        if force or (not exist_meta):
            if exist_meta:
                self.connection_context.drop_table(table=self.meta, schema=self.schema)
            self.connection_context.create_table(self.meta,
                                                 table_structure={"NAME" : "VARCHAR(255)",
                                                                  "VERSION" : "INT",
                                                                  "SCHEMA" : "VARCHAR(255)",
                                                                  "TABLE" : "VARCHAR(255)",
                                                                  "REGISTER_TIME" : "TIMESTAMP",
                                                                  "STORAGE" : "VARCHAR(100)",
                                                                  "LOADED" : "VARCHAR(100)"},
                                                 schema=self.schema,
                                                 table_type='ROW')

    def _get_new_version_no(self, name):
        """
        Gets the next version number for a model.

        Parameters
        ----------
        name: str
            The dataset name
        version: int
            The current version number.

        Returns
        -------
        new_version: int
        """
        last_vers = self._get_last_version_no(name)
        return last_vers + 1

    def _get_last_version_no(self, name):
        """
        Gets the next version number for a model.

        Parameters
        ----------
        name: str
            The dataset name.
        version: int
            0 if the dataset does not exist.
            A number greater than 0 if the dataset already exists.

        Returns
        -------
        new_version: int
        """
        cond = "NAME='{}'".format(name.replace("'", "''"))
        sql = "select max(version)  NEXT_VER from {schema_name}.{table_name}" \
              " where {filter}".format(
                  schema_name=quotename(self.schema),
                  table_name=quotename(self.meta),
                  filter=cond)
        hana_df = DataFrame(connection_context=self.connection_context, select_statement=sql)
        pd_df = hana_df.collect()
        if pd_df.empty:
            return 0
        new_ver = pd_df['NEXT_VER'].iloc[0]
        if new_ver:  # it could be None
            return new_ver
        return 0

    def _get_info(self, name, version, info='TABLE'):
        """
        Gets the table information given name and version.

        Parameters
        ----------
        name: str
            The dataset name.
        version: int
            The version number.
        """
        return DataFrame(self.connection_context, "SELECT * FROM {}.{}".format(quotename(self.schema),
                                                                               quotename(self.meta)))\
                .select('NAME', 'VERSION', info)\
                .filter("NAME='{}'".format(name))\
                .filter("VERSION='{}'".format(version))\
                .collect().iat[0, 2]

    def delete_meta(self, name, version):
        """
        Deletes the dataset metadata.

        Parameters
        ----------
        name: str
            The name of the dataset.
        version: int
            The version number.
        """
        with self.connection_context.connection.cursor() as cursor:
            sql = "DELETE FROM {schema_name}.{table_name} " \
                  "WHERE NAME='{model_name}' " \
                  "AND VERSION={model_version}".format(
                      schema_name=quotename(self.schema),
                      table_name=quotename(self.meta),
                      model_name=name,
                      model_version=version)
            cursor.execute(sql)

    def delete(self, name, version):
        """
        Deletes the dataset.

        Parameters
        ----------
        name: str
            The name of the dataset.
        version: int
            The version number.
        """
        self.connection_context.drop_table(self._get_info(name, version, "TABLE"), self._get_info(name, version, "SCHEMA"))
        self.delete_meta(name, version)

    def list_datasets(self):
        """
        Lists registered datasets.
        """
        self.refresh()
        return self.connection_context.sql("SELECT * FROM {}.{}"
                                           .format(quotename(self.schema),
                                                   quotename(self.meta))).collect()

    def load_dataset(self, name, version):
        """
        Return the dataframe from dataset manager.
        """
        return DataFrame(connection_context=self.connection_context,
                         select_statement="SELECT * FROM {}.{}".format(quotename(self._get_info(name, version, "SCHEMA")),
                                                                       quotename(self._get_info(name, version, "TABLE"))))

    def load_into_memory(self, name, version): #pylint: disable=unused-argument
        """
        Load dataset into memory.

        Parameters
        ----------
        name: str
            The model name.
        version: str
            The model version.
        """
        schema = self._get_info(name, version, "SCHEMA")
        table_name = self._get_info(name, version, "TABLE")
        data_manipulation(self.connection_context,
                          table_name.replace('"', ""),
                          False,
                          schema=schema.replace('"', ""))

    def unload_from_memory(self, name, version, persistent_memory=None): #pylint: disable=unused-argument
        """
        Unload dataset from memory. The dataset will be loaded back into memory after next query.

        Parameters
        ----------
        name: str
            The model name.
        version: str
            The model version.
        persistent_memory : {'retain', 'delete'}, optional
            Only works when persistent memory is enabled.

            Defaults to None.
        """
        schema = self._get_info(name, version, "SCHEMA")
        table_name = self._get_info(name, version, "TABLE")
        data_manipulation(self.connection_context,
                          table_name.replace('"', ""),
                          True,
                          schema=schema.replace('"', ""),
                          persistent_memory=persistent_memory)

    def enable_persistent_memory(self, name, version): #pylint: disable=unused-argument
        """
        Enable persistent memory.

        Parameters
        ----------
        name: str
            The model name.
        version: str
            The model version.
        """
        schema = self._get_info(name, version, "SCHEMA")
        table_name = self._get_info(name, version, "TABLE")
        with self.connection_context.connection.cursor() as cursor:
            sql = "ALTER TABLE {}.{} PERSISTENT MEMORY ON IMMEDIATE CASCADE".format(quotename(schema), quotename(table_name))
            cursor.execute(sql)

    def disable_persistent_memory(self, name, version): #pylint: disable=unused-argument
        """
        Disable persistent memory.

        Parameters
        ----------
        name: str
            The model name.
        version: str
            The model version.
        """
        schema = self._get_info(name, version, "SCHEMA")
        table_name = self._get_info(name, version, "TABLE")
        with self.connection_context.connection.cursor() as cursor:
            sql = "ALTER TABLE {}.{} PERSISTENT MEMORY OFF IMMEDIATE CASCADE".format(quotename(schema), quotename(table_name))
            cursor.execute(sql)

    def refresh(self):
        """
        Refresh the loaded information.
        """
        meta = self.connection_context.sql("SELECT * FROM {}.{}"
                                           .format(quotename(self.schema),
                                                   quotename(self.meta))).collect()
        with self.connection_context.connection.cursor() as cursor:
            for _, row in meta.iterrows():
                schema = row['SCHEMA']
                table_name = row['TABLE']
                loaded = is_mem_loaded(self.connection_context,
                                       table_name.replace('"', ""),
                                       schema.replace('"', ""))
                nse = is_load_unit_page(self.connection_context,
                                        table_name.replace('"', ""),
                                        schema.replace('"', ""))
                cursor.execute("UPDATE {}.{} SET LOADED='{}', STORAGE='{}' WHERE NAME='{}' AND VERSION={}"
                               .format(quotename(self.schema),
                                       quotename(self.meta),
                                       loaded,
                                       nse,
                                       row['NAME'],
                                       row['VERSION']))

    def materialize(self, data, version=None, name=None, storage='NSE', schema=None, table=None):
        """
        Materialize dataset.

        parameters
        ----------
        data : DataFrame
            The dataframe to be registered.
        version : str, optional
            The dataset version.
            Defaults to max(version) + 1.
        name : str, optional
            The dataset name.

            Defaults to DataFrame's name.
        storage : str, optional
            Storage to persist dataset.

            Defaults to NSE.
        schema : str, optional
            Schema name for materialized table.

            Defualts to the current schema.
        table : str, optional
            Table name for matrerialized table.

            If not specified, it will be generated automatically.
        """
        if name is None:
            name = data.name
            if name is None:
                name = [key for key, value in locals().items() if value == data][0]
        if version is None:
            version = self._get_new_version_no(name)
        if schema is None:
            schema = self.connection_context.get_current_schema()
        if table is None:
            table = str(name) + "_" + str(uuid.uuid4()).replace('-', '_').upper()
        seconds_since_epoch = time.time()
        time_obj = time.localtime(seconds_since_epoch)
        register_time = "{}-{}-{} {}:{}:{}".format(time_obj.tm_year,
                                                   time_obj.tm_mon,
                                                   time_obj.tm_mday,
                                                   time_obj.tm_hour,
                                                   time_obj.tm_min,
                                                   time_obj.tm_sec)
        if storage == 'NSE':
            data.save_nativedisktable(where=(schema, table), force=True)
        else:
            data.save(where=(schema, table), force=True)
        with self.connection_context.connection.cursor() as cursor:
            sql = """
                UPSERT {}.{}
                VALUES(?, ?, ?, ?, ?, ?, ?)
                WHERE NAME='{}' AND VERSION={}
                """.format(quotename(self.schema),
                           quotename(self.meta),
                           name,
                           version)
            loaded = is_mem_loaded(self.connection_context,
                                   table,
                                   schema)
            nse = is_load_unit_page(self.connection_context,
                                    table,
                                    schema)
            cursor.execute(sql, (name,
                                 int(version),
                                 schema,
                                 table,
                                 register_time,
                                 nse,
                                 loaded))
