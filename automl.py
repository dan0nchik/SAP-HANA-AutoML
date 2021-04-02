from hana_ml.algorithms.pal.unified_classification import UnifiedClassification
from hana_ml.dataframe import ConnectionContext
from numpy import mod, ndarray
from preprocess.preprocessor import Preprocessor
from utils.connection import connection_context
from pipeline.data import Data
from pipeline.input import Input
from pipeline.pipeline import Pipeline
from hana_ml import DataFrame, dataframe
import hana_ml
import pandas as pd
from hana_ml.algorithms.pal.linear_model import LogisticRegression
from hana_ml.algorithms.apl.classification import AutoClassifier
from hana_ml.algorithms.pal.neighbors import KNNClassifier
from utils.error import AutoMLError


class AutoML:
    def __init__(self):
        self.opt = None
        self.model = None
        self.predicted = None

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
        optimizer: str = "BayesianOptimizer",
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

    def predict(
        self,
        df: pd.DataFrame = None,
        file_path: str = None,
        table_name: str = None,
        id_column: str = None,
    ):
        data = Input(
            df=df,
            path=file_path,
            table_name=table_name,
            id_col=id_column,
        )
        data.load_data()
        self.predicted = self.model.predict(data.hana_df, id_column)
        print(self.predicted.head().collect())

    def save_results(self, file_path: str):
        self.predicted.collect().to_csv(file_path)

    def get_model(self):
        return self.model

    @property
    def optimizer(self):
        return self.opt

    @property
    def best_params(self):
        return self.opt.get_tuned_params()
