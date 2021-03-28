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


class AutoML:
    def __init__(self):
        self.opt = None

    def fit(
            self,
            df: pd.DataFrame = None,
            steps: int = 10,
            target: str = None,
            file_path: str = None,
            url: str = None,
            table_name: str = None,
            columns_to_remove: list = None,
            categorical_features: list = None,
            id_column=None,
            optimizer: str = "BayesianOptimizer",
            config=None,

    ):
        inputted = Input(df=df, target=target, file_path=file_path, url=url, table_name=table_name, id_col=id_column)
        inputted.load_data()
        data = inputted.split_data()
        data.drop(droplist_columns=columns_to_remove)
        pipe = Pipeline(data, steps)
        self.opt = pipe.train(
            categorical_features=categorical_features,
            optimizer=optimizer,
        )

    def optimizer(self):
        return self.opt

    @property
    def best_params(self):
        return self.opt.get_tuned_params()
