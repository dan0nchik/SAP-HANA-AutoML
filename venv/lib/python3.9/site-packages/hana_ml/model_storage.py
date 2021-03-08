#pylint: disable=too-many-lines
"""
This module provides the features of **model storage**.

All these features are accessible to the end user via a single class:

    * :class:`ModelStorage`
    * :class:`ModelStorageError`


"""
import logging
import time
import datetime
import importlib
import pandas as pd

from hdbcli import dbapi
from hana_ml.ml_exceptions import Error
from hana_ml.dataframe import DataFrame, quotename
from hana_ml.ml_base import execute_logged, try_drop

logger = logging.getLogger(__name__) #pylint: disable=invalid-name

# pylint: disable=too-few-public-methods
# pylint: disable=protected-access
# pylint: disable=too-many-statements
# pylint: disable=line-too-long
# pylint: disable=invalid-name
class ModelStorageError(Error):
    """Exception class used in Model Storage"""

class ModelStorage(object): #pylint: disable=useless-object-inheritance
    """
    The ModelStorage class allows users to **save**, **list**, **update**, **restore** or
    **delete** models.\n
    Models are saved into SAP HANA tables in a schema specified by the user.\n
    A model is identified with:\n

    - A name (string of 255 characters maximum),\n
      It must not contain any characters such as coma, semi-colon, tabulation, end-of-line,
      simple-quote, double-quote (',', ';', '"', ''', '\\n', '\\t').
    - A version (positive integer starting from 1).

    A model can be saved in three ways:\n

    1) It can be saved for the first time.\n
       No model with the same name and version is supposed to exist.
    2) It can be saved as a replacement.\n
       If a model with the same name and version already exists, it will be overwritten.\n
    3) It can be saved with a higher version.\n
       The model will be saved with an incremented version number.\n

    Internally, a model is stored as two parts:\n

    1) The metadata.\n
       It contains the model identification (name, version, algorithm class) and also its
       python model object attributes required for reinstantiation.
       It is saved in a table named **HANA_ML_STORAGE* by default.
    2) The back-end model.\n
       It consists in the model returned by APL or PAL.\n
       For APL, it is always saved into the table **HANAMl_APL_MODELS_DEFAULT**,
       while for PAL, a model can be saved into different tables depending on the nature of the
       algorithm.

    Parameters
    ----------
    connection_context : ConnectionContext
        The connection object to an SAP HANA database.
        It must be the same as the one used by the model.

    schema : str
        The schema name where the model storage tables are created.

    Examples
    --------

    Creating and training a model:

    >>> conn = ConnectionContext(HDB_HOST, HDB_PORT, HDB_USER, HDB_PASS)
    >>> # Train dataset
    >>> data = hana_df.DataFrame(conn, 'SELECT * '
    ...                               'from PERFTEST_01.IRIS')
    >>> data_test = hana_df.DataFrame(conn, 'SELECT ID, '
    ...                                     '"sepal length (cm)","sepal width (cm)",'
    ...                                     '"petal length (cm)","petal width (cm)" '
    ...                                     'from PERFTEST_01.IRIS_MULTICLASSES '
    ...                                     )
    >>> model_pal_name = 'MLPClassifier 1'
    >>> model_name = 'AutoClassifier 1'
    >>> model_pal = MLPClassifier(conn, hidden_layer_size=[10, ], activation='TANH', \
                            output_activation='TANH', learning_rate=0.01, momentum=0.001)
    >>> model = AutoClassifier(conn_context=conn)
    >>> model_pal.fit(data,
    ...           label='IS_SETOSA',
    ...           key='ID')
    >>> model.fit(data,
    ...           label='IS_SETOSA',
    ...           key='ID')

    Creating an instance of ModelStorage:

    >>> MODEL_SCHEMA = 'MODEL_STORAGE' # HANA schema in which models are to be saved
    >>> model_storage = ModelStorage(connection_context=conn, schema=MODEL_SCHEMA)

    Saving the trained model for the first time:

    >>> # Saves model
    >>> model_pal.name = model_pal_name
    >>> model_storage.save_model(model=model_pal)
    >>> model.name = model_name
    >>> model_storage.save_model(model=model)

    Listing saved models:

    >>> # Lists model
    >>> list_models2 = model_storage.list_models()
    >>> print(list_models2)
                   NAME  VERSION LIBRARY                         ...
    0  AutoClassifier 1        1     APL  hana_ml.algorithms.apl ...
    1  MLPClassifier 1         1     PAL  hana_ml.algorithms.pal ...

    Reloading a saved model:

    >>> # Loads model
    >>> model1 = model_storage.load_model(name=model_pal_name, version=1)
    >>> model2 = model_storage.load_model(name=model_name)

    Using the loaded model for new predictions:

    >>> # predict
    >>> out = model2.predict(data=data_test)
    >>> out = out.head(3).collect()
    >>> print(out)
       ID PREDICTED  PROBABILITY IS_SETOSA
    0   1      True     0.999492      None    ...
    1   2      True     0.999478      None
    2   3      True     0.999460      None

    Saving the model again:

    >>> # Saves model by overwriting
    >>> model_storage.save_model(model=model, if_exists='replace')
    >>> list_models = model_storage.list_models(name=model.name)
    >>> print(list_models)
                   NAME  VERSION LIBRARY                            ...
    0  AutoClassifier 1        1     APL  hana_ml.algorithms.apl    ...

    >>> # Upgrades model
    >>> model_storage.save_model(model=model, if_exists='upgrade')
    >>> list_models = model_storage.list_models(name=model.name)
    >>> print(list_models)
                   NAME  VERSION LIBRARY                            ...
    0  AutoClassifier 1        1     APL  hana_ml.algorithms.apl    ...
    1  AutoClassifier 1        2     APL  hana_ml.algorithms.apl    ...

    Deleting model:

    >>> model_storage.delete_model(name=model.name, version=model.version)

    Deleteing models of all verions

    >>> model_storage.delete_models(name=name)

    Clean up all the models and meta data

    >>> model_storage.clean_up()

    """

    _VERSION = 1
    # Metadata table
    _METADATA_TABLE_NAME = 'HANAML_MODEL_STORAGE'
    _METADATA_TABLE_DEF = ('('
                           'NAME VARCHAR(255) NOT NULL, '
                           'VERSION INT NOT NULL, '
                           'LIBRARY VARCHAR(128) NOT NULL, '
                           'CLASS VARCHAR(255) NOT NULL, '
                           'JSON VARCHAR(5000) NOT NULL, '
                           'TIMESTAMP TIMESTAMP NOT NULL, '
                           'MODEL_STORAGE_VER INT NOT NULL, '
                           'PRIMARY KEY(NAME, VERSION) )'
                          )
    _METADATA_NB_COLS = 7

    def __init__(self, connection_context, schema=None, meta=None):
        self.connection_context = connection_context
        if schema is None:
            schema = connection_context.sql("SELECT CURRENT_SCHEMA FROM DUMMY")\
            .collect()['CURRENT_SCHEMA'][0]
        self.schema = schema
        if meta is not None:
            self._METADATA_TABLE_NAME = meta

    # ===== Methods callable by the end user

    def list_models(self, name=None, version=None):
        """
        Lists existing models.

        Parameters
        ----------
        connection_context : ConnectionContext object
            The SAP HANA connection.
        name : str, optional
            The model name pattern to be matched.

            Defaults to None.
        version : int, optional
            The model version.

            Defaults to None.

        Returns
        -------
        pandas.DataFrame
            The model metadata matching the provided name and version
        """
        if name:
            self._check_valid_name(name)
        if not self._check_metadata_exists():
            raise ModelStorageError('No model was saved (no metadata)')
        sql = "select * from {schema_name}.{table_name}".format(
            schema_name=quotename(self.schema),
            table_name=quotename(self._METADATA_TABLE_NAME))
        where = ''
        param_vals = []
        if name:
            where = "NAME like ?"
            param_vals.append(name)
        if version:
            if where:
                where = where + ' and '
            where = where + 'version = ?'
            param_vals.append(int(version))
        if where:
            sql = sql + ' where ' + where
        with self.connection_context.connection.cursor() as cursor:
            if where:
                logger.info("Executing SQL: %s %s", sql, str(param_vals))
                cursor.execute(sql, param_vals)
            else:
                logger.info("Executing SQL: %s", sql)
                cursor.execute(sql)
            res = cursor.fetchall()

            col_names = [i[0] for i in cursor.description]
            res = pd.DataFrame(res, columns=col_names)
        return res

    def model_already_exists(self, name, version):
        """
        Checks if a model already exists in the model storage.

        Parameters
        ----------
        name : str
            The model name
        version : int
            The model version

        Returns
        -------
        bool
            If True, there is already a model with the same name and version.
            If False, there is no model with the same name.
        """
        self._check_valid_name(name)
        if not self._check_metadata_exists():
            return False
        with self.connection_context.connection.cursor() as cursor:
            try:
                sql = "select count(*) from {schema_name}.{table_name} " \
                      "where NAME = ? " \
                      "and version = ?".format(
                          schema_name=quotename(self.schema),
                          table_name=quotename(self._METADATA_TABLE_NAME))
                params = [name, int(version)]
                logger.info('Execute SQL: %s %s', sql, str(params))
                cursor.execute(sql, params)
                res = cursor.fetchall()
                if res[0][0] > 0:
                    return True
            except dbapi.Error:
                pass
            return False

    def save_model(self, model, if_exists='upgrade'):
        """
        Saves a model.

        Parameters
        ----------
        model : a model instance
            The model name must have been set.
            The couple (name, version) will serve as unique id.
        if_exists : str, optional
            It specifies the behavior when a model with a same name/version already exists:
                - fail: Raises an Error.
                - replace: Overwrites the model.
                - upgrade: Saves the model with an incremented version.

            Defaults to 'upgrade'.
        """
        # Checks the artifact model table exists for the current model
        if not model.is_fitted() and model._is_APL():
            # Problem with PAL; like GLM, some models do not have model_ attribute
            raise ModelStorageError(
                "The model cannot be saved. Please fit the model or load an existing one.")
        # Checks the parameter if_exists is correct
        if if_exists not in {'fail', 'replace', 'upgrade'}:
            raise ValueError("Unexpected value of 'if_exists' parameter: ", if_exists)
        # Checks if the model name is set
        name, version = self._get_model_id(model)
        if not name or not version and model._is_APL():
            raise ModelStorageError('The name of the model must be set.')
        self._check_valid_name(name)
        model_id = {'name': name, 'version': version}
        # Checks a model with the same name already exists
        model_exists = self.model_already_exists(**model_id)
        if model_exists:
            if if_exists == 'fail':
                raise ModelStorageError('A model with the same name/version already exists')
            if if_exists == 'upgrade':
                version = self._get_new_version_no(name=name)
                setattr(model, 'version', version)
                model_id = {'name': name, 'version': version}

        # Starts transaction to save data
        # Sets autocommit to False to ensure transaction isolation
        conn = self.connection_context.connection # SAP HANA connection
        old_autocommit = conn.getautocommit()
        conn.setautocommit(False)
        logger.info("Executing SQL: -- Disable autocommit")
        try:
            # Disables autocommit for tables creation
            with self.connection_context.connection.cursor() as cursor:
                execute_logged(cursor, 'SET TRANSACTION AUTOCOMMIT DDL OFF')
            # Creates metadata table if it does not exist
            self._create_metadata_table()
            # Deletes old version for replacement
            if model_exists and if_exists == 'replace':
                # Deletes before resaving as a new model
                self._delete_model(**model_id)
            # Saves the back-end model and returns the json for metadata
            # pylint: disable=protected-access
            js_str = model._encode_and_save(schema=self.schema)
            # Saves metadata with json
            self._save_metadata(model=model, js_str=js_str)
            # commits changes in database
            logger.info('Executing SQL: commit')
            conn.commit()
        except dbapi.Error as db_er:
            logger.error("An issue occurred in database during model saving: %s",
                         db_er, exc_info=True)
            logger.info('Executing SQL: rollback')
            conn.rollback()
            raise ModelStorageError('Unable to save the model.')
        except Exception as ex:
            logger.error('An issue occurred during model saving: %s', ex, exc_info=True)
            logger.info('Executing SQL: rollback')
            conn.rollback()
            raise ModelStorageError('Unable to save the model')
        finally:
            logger.info('Model %s is correctly saved', model.name)
            conn.setautocommit(old_autocommit)

    def delete_model(self, name, version):
        """
        Deletes the model of a given name and version.

        Parameters
        ----------
        name : str
            The model name.

        version : int
            The model version.
        """
        if not name or not version:
            raise ValueError("The model name and version must be specified.")
        self._check_valid_name(name)
        if self.model_already_exists(name, version):
            # Sets autocommit to False to ensure transaction isolation
            conn = self.connection_context.connection  # SAP HANA connection
            old_autocommit = conn.getautocommit()
            conn.setautocommit(False)
            try:
                # Gets the json string fromp the metadata
                self._delete_model(name=name, version=version)
                logger.info('Executing SQL: commit')
                conn.commit()
            except dbapi.Error as db_er:
                logger.error("An issue occurred in database during model removal:: %s",
                             db_er, exc_info=True)
                logger.info('Executing SQL: rollback')
                conn.rollback()
                raise ModelStorageError('Unable to delete the model.')
            except Exception as ex:
                logger.error('An issue occurred during model removal: %s', ex, exc_info=True)
                logger.info('Executing SQL: rollback')
                conn.rollback()
                raise ModelStorageError('Unable to delete the model.')
            finally:
                logger.info('Model %s is correctly deleted.', name)
                conn.setautocommit(old_autocommit)
        else:
            raise ModelStorageError('There is no model/version with this name:', name)

    def delete_models(self, name, start_time=None, end_time=None):
        """
        Deletes the model in a batch model with specified time range.

        Parameters
        ----------
        name : str
            The model name
        start_time : str, optional
            The start timestamp for deleting.

            Defaults to None.
        end_time : str, optional
            The end timestamp for deleting.

            Defaults to None.
        """
        if not name:
            raise ValueError("The model name must be specified.")
        self._check_valid_name(name)
        meta = self.list_models(name=name)
        if start_time is not None and end_time is not None:
            meta = meta[(meta['TIMESTAMP'] >= start_time) & (meta['TIMESTAMP'] <= end_time)]
        elif start_time is not None and end_time is None:
            meta = meta[meta['TIMESTAMP'] >= start_time]
        elif start_time is None and end_time is not None:
            meta = meta[meta['TIMESTAMP'] <= end_time]
        else:
            pass
        for row in meta.itertuples():
            self.delete_model(name, row.VERSION)

    def clean_up(self):
        """
        Be cautious! This function will delete all the models and the meta table.
        """
        for model in set(self.list_models()['NAME'].to_list()):
            self.delete_models(model)
        try_drop(self.connection_context, self._METADATA_TABLE_NAME)

    def load_model(self, name, version=None, **kwargs):
        """
        Loads an existing model from the database.

        Parameters
        ----------
        name : str
            The model name.
        version : int, optional
            The model version.
            By default, the last version will be loaded.

        Returns
        -------
        PAL/APL object
            The loaded model ready for use.
        """
        self._check_valid_name(name)
        if not version:
            version = self._get_last_version_no(name=name)
        if not self.model_already_exists(name=name, version=version):
            raise ModelStorageError('The model "{}" version {} does not exist'.format(
                name, version))
        metadata = self._get_model_metadata(name=name, version=version)
        # pylint: disable=protected-access
        model_class = self._load_class(metadata['CLASS'])
        js_str = metadata['JSON']
        model = model_class._load_model(
            connection_context=self.connection_context,
            name=name,
            version=version,
            js_str=js_str,
            **kwargs)
        if ("ARIMA" in type(model).__name__) or ("AutoARIMA" in type(model).__name__):
            model.set_conn(self.connection_context)
        return model

    # ===== Private methods

    @staticmethod
    def _load_class(class_full_name):
        """
        Imports the required module for <class_full_name> and returns the class.

        Parameters
        ----------
        class_full_name: str
            The fully qualified class name.

        Returns
        -------
        The class
        """
        components = class_full_name.split('.')
        if "src" in components:
            components.remove("src")
        mod_name = '.'.join(components[:-1])
        mod = importlib.import_module(mod_name)
        cur_class = getattr(mod, components[-1])
        return cur_class

    def _create_metadata_table(self):
        """"
        Creates the metadata table if it does not exists.
        """
        if not self._check_metadata_exists():
            with self.connection_context.connection.cursor() as cursor:
                sql = 'CREATE COLUMN TABLE {schema_name}.{table_name} {cols_def}'.format(
                    schema_name=quotename(self.schema),
                    table_name=quotename(self._METADATA_TABLE_NAME),
                    cols_def=self._METADATA_TABLE_DEF)
                execute_logged(cursor, sql)

    def _save_metadata(self, model, js_str):
        """
        Saves the model metadata.

        Parameters
        ----------
        model : A SAP HANA ML model
            The model instance
        js_str : str
            JSON string to be saved in the metadata table
        """
        with self.connection_context.connection.cursor() as cursor:
            # Inserts data into the metadata table
            now = time.time()  # timestamp
            now_str = datetime.datetime.fromtimestamp(now).strftime('%Y-%m-%d %H:%M:%S')
            lib = 'PAL'  # lib
            fclass_name = model.__module__ + '.' + type(model).__name__ # class name
            if model.__module__.startswith('hana_ml.algorithms.apl'):
                lib = 'APL'
            sql = 'insert into {}.{} values({})'.format(
                quotename(self.schema),
                quotename(self._METADATA_TABLE_NAME),
                ', '.join(['?'] * self._METADATA_NB_COLS)  # seven '?'
                )
            logger.info("Prepare SQL: %s", sql)
            data = [(
                model.name,
                int(model.version),
                lib,
                fclass_name,
                js_str,
                now_str,
                self._VERSION)]
            logger.info("Executing SQL: INSERT INTO %s.%s values %s",
                        quotename(self.schema),
                        quotename(self._METADATA_TABLE_NAME),
                        str(data)
                       )
            cursor.executemany(sql, data)

    @staticmethod
    def _get_model_id(model):
        """
        Returns the model id (name, version).

        Parameters
        ----------
        model : an instance of PAL or APL model.

        Returns
        -------
        A tuple of two elements (name, version).
        """
        name = getattr(model, 'name', None)
        version = getattr(model, 'version', None)
        return name, version

    @staticmethod
    def _check_valid_name(name):
        """
        Checks the model name is correctly set.
        It must not contain the characters: ',', ';', '"', ''', '\n', '\t'

        Returns
        -------
        If a forbidden character is in the name, raises a ModelStorageError exception.
        """
        forbidden_chars = [';', ',', '"', "'", '\n', '\t']
        if any(c in name for c in forbidden_chars):
            raise ModelStorageError('The model name contains unauthorized characters.')

    def _check_metadata_exists(self):
        """
        Checks if the metadata table exists.

        Returns
        -------
        True/False if yes/no.
        Raise error if database issue
        """
        with self.connection_context.connection.cursor() as cursor:
            try:
                sql = 'select 1 from {schema_name}.{table_name} limit 1'.format(
                    schema_name=quotename(self.schema),
                    table_name=quotename(self._METADATA_TABLE_NAME))
                execute_logged(cursor, sql)
                return True
            except dbapi.Error as err:
                if err.errorcode == 259:
                    return False
                if err.errorcode == 258:
                    raise ModelStorageError('Cannot read the schema. ' + err.errortext)
                raise ModelStorageError('Database issue. ' + err.errortext)
        return False

    def _delete_metadata(self, name, version):
        """
        Deletes the model metadata.

        Parameters
        ----------
        name : str
            The name of the model to be deleted
        conn_context : connection_context
            The SAP HANA connection
        """
        with self.connection_context.connection.cursor() as cursor:
            sql = "delete from {schema_name}.{table_name} " \
                  "where NAME='{model_name}' " \
                  "and VERSION={model_version}".format(
                      schema_name=quotename(self.schema),
                      table_name=quotename(self._METADATA_TABLE_NAME),
                      model_name=name.replace("'", "''"),
                      model_version=version)
            execute_logged(cursor, sql)

    def enable_persistent_memory(self, name, version):
        """
        Enable persistent memory.

        Parameters
        ----------
        name : str
            The name of the model to be deleted
        version : int
            The model version
        """
        metadata = self._get_model_metadata(name, version)
        js_str = metadata['JSON']
        model_class = self._load_class(metadata['CLASS'])
        model_class._enable_persistent_memory(connection_context=self.connection_context,
                                              name=name,
                                              version=version,
                                              js_str=js_str)

    def disable_persistent_memory(self, name, version):
        """
        Disable persistent memory.

        Parameters
        ----------
        name : str
            The name of the model to be deleted
        version : int
            The model version
        """
        metadata = self._get_model_metadata(name, version)
        js_str = metadata['JSON']
        model_class = self._load_class(metadata['CLASS'])
        model_class._disable_persistent_memory(connection_context=self.connection_context,
                                               name=name,
                                               version=version,
                                               js_str=js_str)

    def load_into_memory(self, name, version):
        """
        Load a model to memory.

        Parameters
        ----------
        name : str
            The name of the model to be deleted
        version : int
            The model version
        """
        metadata = self._get_model_metadata(name, version)
        js_str = metadata['JSON']
        model_class = self._load_class(metadata['CLASS'])
        model_class._load_mem(connection_context=self.connection_context,
                              name=name,
                              version=version,
                              js_str=js_str)

    def unload_from_memory(self, name, version, persistent_memory=None):
        """
        Unload a model to memory. The dataset will be loaded back into memory after next query.

        Parameters
        ----------
        name : str
            The name of the model to be deleted
        version : int
            The model version
        persistent_memory : {'retain', 'delete'}, optional
            Only works when persistent memory is enabled.

            Defaults to None.
        """
        metadata = self._get_model_metadata(name, version)
        js_str = metadata['JSON']
        model_class = self._load_class(metadata['CLASS'])
        model_class._unload_mem(
            connection_context=self.connection_context,
            name=name, version=version, js_str=js_str,
            persistent_memory=persistent_memory)

    def _delete_model(self, name, version):
        """
        Deletes a model.

        Parameters
        ----------
        name : str
            The name of the model to be deleted
        version : int
            The model version
        """
        # Reads the metadata
        metadata = self._get_model_metadata(name, version)
        js_str = metadata['JSON']
        model_class = self._load_class(metadata['CLASS'])
        # Deletes the metadata
        self._delete_metadata(name=name, version=version)
        # Deletes the back-end model by calling the model class static method _delete_model()
        # pylint: disable=protected-access
        model_class._delete_model(
            connection_context=self.connection_context,
            name=name, version=version, js_str=js_str)

    def _get_model_metadata(self, name, version):
        """
        Reads the json string from the metadata of the model

        Parameters
        ----------
        name : str
            The model name

        Returns
        -------
        metadata : dict
            A dictionary containing the metadata of the model.
            The keys of the dictionary correspond to the columns of the metadata table:
            {'NAME': 'My Model', 'LIB': 'APL', 'CLASS': ...}
        """
        pd_series = self.list_models(name=name, version=version).head(1)
        return pd_series.iloc[0].to_dict()

    def _get_new_version_no(self, name):
        """
        Gets the next version number for a model.

        Parameters
        ----------
        name : str
            The model name
        version : int
            The current version number

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
        name : str
            The model name
        version : int
            0 if the model does not exist.
            A number greater than 0 if the model already exists.

        Returns
        -------
        new_version: int
        """
        cond = "NAME='{}'".format(name.replace("'", "''"))
        sql = "select max(version)  NEXT_VER from {schema_name}.{table_name}" \
              " where {filter}".format(
                  schema_name=quotename(self.schema),
                  table_name=quotename(self._METADATA_TABLE_NAME),
                  filter=cond)
        hana_df = DataFrame(connection_context=self.connection_context, select_statement=sql)
        pd_df = hana_df.collect()
        if pd_df.empty:
            return 0
        new_ver = pd_df['NEXT_VER'].iloc[0]
        if new_ver:  # it could be None
            return new_ver
        return 0
