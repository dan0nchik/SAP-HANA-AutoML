import pandas as pd
import json
from pipeline.input import Input
from pipeline.pipeline import Pipeline
from preprocess.preprocessor import Preprocessor
from utils.error import AutoMLError
from hana_ml.model_storage import ModelStorage
import pickle


class AutoML:
    def __init__(self):
        self.opt = None
        self.model = None
        self.predicted = None
        self.preprocessor_settings = None

    def fit(
        self,
        df: pd.DataFrame = None,
        steps: int = 10,
        target: str = None,
        file_path: str = None,
        table_name: str = None,
        columns_to_remove: list = None,
        categorical_features: list = None,
        id_column=None,
        optimizer: str = "OptunaSearch",
        config=None,
    ):
        if steps < 1:
            raise AutoMLError("The number of steps < 1!")
        inputted = Input(
            df=df,
            target=target,
            path=file_path,
            table_name=table_name,
            id_col=id_column,
        )
        inputted.load_data()
        data = inputted.split_data()
        if columns_to_remove is not None:
            data.drop(droplist_columns=columns_to_remove)
        pipe = Pipeline(data, steps)
        self.opt = pipe.train(
            categorical_features=categorical_features, optimizer=optimizer
        )
        self.model = self.opt.get_model()
        self.preprocessor_settings = self.opt.get_preprocessor_settings()

    def predict(
        self,
        df: pd.DataFrame = None,
        file_path: str = None,
        table_name: str = None,
        id_column: str = None,
        preprocessor_file: str = None,
    ):
        data = Input(
            df=df,
            path=file_path,
            table_name=table_name,
            id_col=id_column,
        )
        data.load_data()

        if preprocessor_file is not None:
            with open(preprocessor_file) as json_file:
                json_data = json.load(json_file)
                self.preprocessor_settings = json_data
        print("Preprocessor settings:", self.preprocessor_settings)
        pr = Preprocessor()
        data.hana_df = pr.clean(
            data=data.hana_df, num_strategy=self.preprocessor_settings["imputer"]
        )
        self.predicted = self.model.predict(data.hana_df, id_column)
        res = self.predicted
        if type(self.predicted) == tuple:
            res = res[0]
        print(
            "Prediction results (first 20 rows): \n", res.head(20).collect()
        )

    def save_results_as_csv(self, file_path: str):
        res = self.predicted
        if type(self.predicted) == tuple:
            res = res[0]
        res.collect().to_csv(file_path)

    def save_stats_as_csv(self, file_path: str):
        res = self.predicted
        if type(self.predicted) == tuple:
            res = res[1]
        res.collect().to_csv(file_path)

    def save_preprocessor(self, file_path: str):
        with open(file_path, "w+") as file:
            json.dump(self.preprocessor_settings, file)

    def model_to_file(self, file_path: str):
        with open(file_path, "w+") as file:
            json.dump(self.opt.get_tuned_params(), file)

    def get_model(self):
        return self.model

    @property
    def optimizer(self):
        return self.opt

    @property
    def best_params(self):
        return self.opt.get_tuned_params()


class Storage(ModelStorage):
    def __init__(self, connection_context):
        super().__init__(connection_context)
