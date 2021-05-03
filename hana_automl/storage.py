import json
from types import SimpleNamespace
from hana_ml.model_storage import ModelStorage
import hdbcli
import pandas as pd

from hana_automl.automl import AutoML
from hana_automl.preprocess.settings import PreprocessorSettings

PREPROCESSORS = "PREPROCESSOR_STORAGE"


class Storage(ModelStorage):
    """Save your HANA PAL model easily.
    Details here:
    https://help.sap.com/doc/1d0ebfe5e8dd44d09606814d83308d4b/2.0.04/en-US/hana_ml.model_storage.html"""

    def __init__(self, address, port, user, password, connection_context, schema):
        super().__init__(connection_context, schema)
        CONN = hdbcli.dbapi.connect(
            address=address, port=port, user=user, password=password
        )
        self.cursor = CONN.cursor()
        if not table_exists(self.cursor, self.schema, PREPROCESSORS):
            self.cursor.execute(
                f"CREATE TABLE {self.schema}.{PREPROCESSORS} (MODEL NVARCHAR(256), VERSION INT, JSON NVARCHAR("
                f"5000)); "
            )

    def save_model(self, automl: AutoML, if_exists="upgrade"):
        super().save_model(automl.model, if_exists)
        json_settings = json.dumps(automl.preprocessor_settings.__dict__)
        self.cursor.execute(
            f"INSERT INTO {PREPROCESSORS} (MODEL, VERSION, JSON) VALUES ('{automl.model.name}', {automl.model.version}, '{json_settings}'); "
        )

    def list_preprocessors(self):
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
        self.cursor.execute(f"SELECT * FROM {self.schema}.{PREPROCESSORS} WHERE MODEL = '{name}' "
                            f"AND VERSION = {self.__extract_version(name)}")
        data = self.cursor.fetchall()[0][2]  # JSON column
        settings_namespace = json.loads(str(data), object_hook=lambda d: SimpleNamespace(**d))
        automl.preprocessor_settings = settings_namespace
        return automl

    def clean_up(self):
        super().clean_up()
        self.cursor.execute(f"DROP TABLE {self.schema}.{PREPROCESSORS}")

    def __extract_version(self, name):
        self.cursor.execute(f"SELECT * FROM {self.schema}.{PREPROCESSORS} WHERE MODEL='{name}'")
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
