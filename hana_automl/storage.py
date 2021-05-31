from hana_automl.algorithms.base_algo import BaseAlgorithm
from hana_automl.preprocess.settings import PreprocessorSettings
import json
from types import SimpleNamespace
from hana_ml.model_storage import ModelStorage
import hdbcli
import pandas as pd
from hana_ml.dataframe import ConnectionContext
from hana_automl.automl import AutoML

PREPROCESSORS = "PREPROCESSOR_STORAGE"


class Storage(ModelStorage):
    """Storage for models and more.

    Attributes
    ----------
    address : str
        Host of the database. Example: 'localhost'
    port : int
        Port to connect. Example: 39015
    user: str
        Username of database user.
    password: str
        Well, just user's password.
    connection_context: hana_ml.dataframe.ConnectionContext
        Connection info for HANA database.
    schema : str
        Database schema.
    """

    def __init__(self, connection_context: ConnectionContext, schema: str):
        super().__init__(connection_context, schema)
        self.cursor = connection_context.connection.cursor()
        if not table_exists(self.cursor, self.schema, PREPROCESSORS):
            self.cursor.execute(
                f"CREATE TABLE {self.schema}.{PREPROCESSORS} (MODEL NVARCHAR(256), VERSION INT, JSON NVARCHAR("
                f"5000)); "
            )

    def save_model(self, automl: AutoML, if_exists="upgrade"):
        """
        Saves a model to database.

        Parameters
        ----------
        automl: AutoML
            The model.
        if_exists: str
            Defaults to "upgrade". Not recommended to change.
        """
        super().save_model(automl.algorithm.model, if_exists)
        json_settings = json.dumps(automl.preprocessor_settings.__dict__)
        self.cursor.execute(
            f"INSERT INTO {PREPROCESSORS} (MODEL, VERSION, JSON) VALUES ('{automl.model.name}', {automl.model.version}, '{json_settings}'); "
        )

    def list_preprocessors(self):
        """
        Show preprocessors in database.

        Returns
        -------
        res: pd.DataFrame
            DataFrame containing all preprocessors in database.
        """
        self.cursor.execute(f"SELECT * FROM {self.schema}.{PREPROCESSORS}")
        res = self.cursor.fetchall()

        col_names = [i[0] for i in self.cursor.description]
        res = pd.DataFrame(res, columns=col_names)
        return res

    def delete_model(self, name, version):
        super().delete_model(name, version)
        self.cursor.execute(
            f"DELETE FROM {self.schema}.{PREPROCESSORS} WHERE MODEL = '{name}' AND VERSION = {version}"
        )

    def delete_models(self, name, start_time=None, end_time=None):
        super().delete_models(name, start_time, end_time)
        self.cursor.execute(f"DELETE FROM {self.schema}.{PREPROCESSORS}")

    def load_model(self, name, version=None, **kwargs):
        automl = AutoML(self.connection_context)
        automl.model = super().load_model(name, version, **kwargs)
        automl.algorithm = BaseAlgorithm(model=automl.model)
        self.cursor.execute(
            f"SELECT * FROM {self.schema}.{PREPROCESSORS} WHERE MODEL = '{name}' "
            f"AND VERSION = {self.__extract_version(name)}"
        )
        data = self.cursor.fetchall()[0][2]  # JSON column
        settings_namespace = json.loads(
            str(data), object_hook=lambda d: SimpleNamespace(**d)
        )
        automl.preprocessor_settings = PreprocessorSettings(
            settings_namespace.strategy_by_col
        )
        automl.preprocessor_settings.tuned_num_strategy = (
            settings_namespace.tuned_num_strategy
        )
        automl.preprocessor_settings.tuned_normalizer_strategy = (
            settings_namespace.tuned_normalizer_strategy
        )
        automl.preprocessor_settings.tuned_z_score_method = (
            settings_namespace.tuned_z_score_method
        )
        automl.preprocessor_settings.tuned_normalize_int = (
            settings_namespace.tuned_normalize_int
        )
        automl.preprocessor_settings.strategy_by_col = (
            settings_namespace.strategy_by_col
        )
        automl.preprocessor_settings.categorical_cols = (
            settings_namespace.categorical_cols
        )
        automl.preprocessor_settings.task = settings_namespace.task
        return automl

    def clean_up(self):
        super().clean_up()
        self.cursor.execute(f"DROP TABLE {self.schema}.{PREPROCESSORS}")

    def __extract_version(self, name):
        self.cursor.execute(
            f"SELECT * FROM {self.schema}.{PREPROCESSORS} WHERE MODEL='{name}'"
        )
        res = self.cursor.fetchall()
        versions = []
        for string in res:
            versions.append(string[1])
        return max(versions)


def table_exists(cursor, schema, name):
    cursor.execute(
        f"SELECT count(*) FROM TABLES WHERE SCHEMA_NAME='{schema}' AND TABLE_NAME='{name}';"
    )
    res = cursor.fetchall()
    if res[0][0] > 0:
        return True
    return False
