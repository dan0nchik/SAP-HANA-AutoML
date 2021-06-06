import json
from types import SimpleNamespace

import pandas as pd
from hana_ml.dataframe import ConnectionContext
from hana_ml.model_storage import ModelStorage

from hana_automl.algorithms.base_algo import BaseAlgorithm
from hana_automl.algorithms.ensembles.blendcls import BlendingCls
from hana_automl.algorithms.ensembles.blendreg import BlendingReg
from hana_automl.automl import AutoML
from hana_automl.pipeline.modelres import ModelBoard
from hana_automl.preprocess.settings import PreprocessorSettings
from hana_automl.utils.error import StorageError

PREPROCESSORS = "AUTOML_PREPROCESSOR_STORAGE"
ensemble_prefix = 'ensemble'
leaderboard_prefix = 'leaderboard'

class Storage(ModelStorage):
    """Storage for models and more.

    Attributes
    ----------
    connection_context: hana_ml.dataframe.ConnectionContext
        Connection info for HANA database.
    schema : str
        Database schema.

    Examples
    --------
    >>> from hana_automl.storage import Storage
    >>> from hana_ml import ConnectionContext
    >>> cc = ConnectionContext('address', 39015, 'user', 'password')
    >>> storage = Storage(cc, 'your schema')
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

        Note
        ----
        If you have ensemble enabled in AutoML model, method will determine it automatically and split
        ensemble model in multiple usual models.

        Examples
        --------
        >>> from hana_automl.automl import AutoML
        >>> automl.fit(df='table in HANA', target='some target', steps=3)
        >>> storage.save_model(automl)
        """
        if not table_exists(self.cursor, self.schema, PREPROCESSORS):
            self.cursor.execute(
                f"CREATE TABLE {self.schema}.{PREPROCESSORS} (MODEL NVARCHAR(256), VERSION INT, JSON NVARCHAR("
                f"5000)); "
            )
        if isinstance(automl.model, BlendingCls) or isinstance(
            automl.model, BlendingReg
        ):
            if isinstance(automl.model, BlendingCls):
                ensemble_name = "_ensemble_cls_"
            if isinstance(automl.model, BlendingReg):
                ensemble_name = "_ensemble_reg_"
            model_counter = 1
            for model in automl.model.model_list:  # type: ModelBoard
                if automl.model.name is None or automl.model.name == "":
                    raise StorageError("Please give your model a name.")
                name = automl.model.name + ensemble_name + str(model_counter)
                model.algorithm.model.name = name
                json_settings = json.dumps(model.preprocessor.__dict__)
                if self.model_already_exists(name, model.algorithm.model.version):
                    self.cursor.execute(
                        f"UPDATE {self.schema}.{PREPROCESSORS} SET VERSION={model.algorithm.model.version}, JSON='{str(json_settings)}' "
                        f"WHERE MODEL='{name}';"
                    )
                else:
                    self.cursor.execute(
                        f"INSERT INTO {self.schema}.{PREPROCESSORS} (MODEL, VERSION, JSON) VALUES ('{name}', {model.algorithm.model.version}, '{str(json_settings)}'); "
                    )
                super().save_model(
                    model.algorithm.model, if_exists="replace"
                )  # to avoid duplicates

                model_counter += 1

        else:
            if table_exists(self.cursor, self.schema, "HANAML_MODEL_STORAGE"):
                if len(self.__find_models(automl.model.name, ensemble_prefix)) > 0:
                    raise StorageError(
                        "There is an ensemble with the same name in storage. Please change the name of "
                        "the "
                        "model."
                    )
            super().save_model(automl.model, if_exists)
            json_settings = json.dumps(automl.preprocessor_settings.__dict__)
            self.cursor.execute(
                f"INSERT INTO {PREPROCESSORS} (MODEL, VERSION, JSON) VALUES ('{automl.model.name}', {automl.model.version}, '{json_settings}'); "
            )

    def list_preprocessors(self, name: str = None) -> pd.DataFrame:
        """
        Show preprocessors for models in database.

        Parameters
        ----------
        name: str, optional
            Model name.

        Returns
        -------
        res: pd.DataFrame
            DataFrame containing all preprocessors in database.

        Note
        ----
        Do not delete or save preprocessors apart from model!
        They are saved/deleted/changed automatically WITH model.

        Examples
        --------
        >>> storage.list_preprocessors()
             MODEL  VERSION	 JSON
          1.  test        1  {'tuned_num'...}
        """

        if (name is not None) and name != "":
            ensembles = self.__find_models(name, ensemble_prefix)
            if len(ensembles) > 0:
                result = pd.DataFrame(columns=["MODEL", "VERSION", "JSON"])
                for model in ensembles:
                    self.cursor.execute(
                        f"SELECT * FROM {self.schema}.{PREPROCESSORS} WHERE MODEL='{model[0]}';"
                    )
                    res = self.cursor.fetchall()
                    col_names = [i[0] for i in self.cursor.description]
                    df = pd.DataFrame(res, columns=col_names)
                    result = result.append(df, ignore_index=True)
                return result
            else:
                self.cursor.execute(
                    f"SELECT * FROM {self.schema}.{PREPROCESSORS} WHERE MODEL='{name}';"
                )
                res = self.cursor.fetchall()
                col_names = [i[0] for i in self.cursor.description]
                return pd.DataFrame(res, columns=col_names)
        else:
            self.cursor.execute(f"SELECT * FROM {self.schema}.{PREPROCESSORS};")
            res = self.cursor.fetchall()
            col_names = [i[0] for i in self.cursor.description]
            return pd.DataFrame(res, columns=col_names)

    def save_leaderboard(self, leaderboard: list, name: str):
        for model_member in leaderboard:
            pass
    def load_leaderboard(self, name: str):
        pass
    def delete_leaderboard(self, name: str):
        pass
    def delete_model(self, name: str, version: int = None):
        """Deletes model.

        Parameters
        ----------
        name: str
            Model to remove
        version: int, optional
            Model's version.
        """
        ensembles = self.__find_models(name, ensemble_prefix)
        if len(ensembles) > 0:
            for model in ensembles:  # type: tuple
                super().delete_model(model[0], model[1])
                self.cursor.execute(
                    f"DELETE FROM {self.schema}.{PREPROCESSORS} WHERE MODEL = '{model[0]}' AND VERSION = {model[1]}"
                )
        else:
            super().delete_model(name, version)
            self.cursor.execute(
                f"DELETE FROM {self.schema}.{PREPROCESSORS} WHERE MODEL = '{name}' AND VERSION = {version}"
            )

    def delete_models(self, name: str, start_time=None, end_time=None):
        raise NotImplementedError(
            "This method will be implemented soon. Sorry for trouble."
        )

    def load_model(self, name: str, version: int = None, **kwargs):
        """Loads new model.

        Parameters
        ----------
        name: str
            Model to load
        version: int, optional
            Model's version.

        Returns
        -------
        AutoML object
        """
        automl = AutoML(self.connection_context)
        ensembles = self.__find_models(name, ensemble_prefix)
        if len(ensembles) > 0:
            model_list = []
            for model_name in ensembles:  # type: tuple
                self.cursor.execute(
                    f"SELECT * FROM {self.schema}.{PREPROCESSORS} WHERE MODEL = '{model_name[0]}' "
                    f"AND VERSION = {model_name[1]}"
                )
                data = self.cursor.fetchall()[0][2]  # JSON column
                hana_model = super().load_model(model_name[0], model_name[1], **kwargs)
                model_board_member = ModelBoard(
                    BaseAlgorithm(model=hana_model), 0, self.__setup_preprocessor(data)
                )
                model_list.append(model_board_member)
            if "cls" in ensembles[0][0]:
                automl.model = BlendingCls(
                    model_list=model_list, connection_context=self.connection_context
                )
            if "reg" in ensembles[0][0]:
                automl.model = BlendingReg(
                    model_list=model_list, connection_context=self.connection_context
                )
            automl.ensemble = True
        else:
            automl.model = super().load_model(name, version, **kwargs)
            automl.algorithm = BaseAlgorithm(model=automl.model)
            self.cursor.execute(
                f"SELECT * FROM {self.schema}.{PREPROCESSORS} WHERE MODEL = '{name}' "
                f"AND VERSION = {self.__extract_version(name)}"
            )
            data = self.cursor.fetchall()[0][2]  # JSON column
            automl.preprocessor_settings = self.__setup_preprocessor(data)
        return automl

    def clean_up(self):
        """Be careful! This method deletes all models from database!"""
        super().clean_up()
        self.cursor.execute(f"DROP TABLE {self.schema}.{PREPROCESSORS}")

    def __extract_version(self, name: str):
        self.cursor.execute(
            f"SELECT * FROM {self.schema}.{PREPROCESSORS} WHERE MODEL='{name}'"
        )
        res = self.cursor.fetchall()
        versions = []
        for string in res:
            versions.append(string[1])
        return max(versions)

    def __find_models(self, name: str, prefix: str):
        models = self.list_models()["NAME"]
        versions = self.list_models()["VERSION"]
        return [
            (model_name, version)
            for model_name, version in zip(models, versions)
            if (name in model_name) and (prefix in model_name)
        ]

    def __setup_preprocessor(self, data) -> PreprocessorSettings:
        settings_namespace = json.loads(
            str(data), object_hook=lambda d: SimpleNamespace(**d)
        )
        preprocessor = PreprocessorSettings(settings_namespace.strategy_by_col)
        preprocessor.tuned_num_strategy = settings_namespace.tuned_num_strategy
        preprocessor.tuned_normalizer_strategy = (
            settings_namespace.tuned_normalizer_strategy
        )
        preprocessor.tuned_z_score_method = settings_namespace.tuned_z_score_method
        preprocessor.tuned_normalize_int = settings_namespace.tuned_normalize_int
        preprocessor.strategy_by_col = settings_namespace.strategy_by_col
        preprocessor.categorical_cols = settings_namespace.categorical_cols
        preprocessor.task = settings_namespace.task
        preprocessor.normalization_exceptions = (
            settings_namespace.normalization_exceptions
        )
        return preprocessor


def table_exists(cursor, schema, name):
    cursor.execute(
        f"SELECT count(*) FROM TABLES WHERE SCHEMA_NAME='{schema}' AND TABLE_NAME='{name}';"
    )
    res = cursor.fetchall()
    if res[0][0] > 0:
        return True
    return False
