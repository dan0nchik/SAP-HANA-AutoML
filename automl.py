from hana_ml.dataframe import ConnectionContext
from numpy import ndarray
from utils.connection import connection_context
from pipeline.data import Data
from pipeline.input import Input
from pipeline.pipeline import Pipeline
from hana_ml import DataFrame, dataframe
import hana_ml
import pandas as pd


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
        columns_to_remove: list = None,
        categorical_features: list = None,
        optimizer: str = "BayesianOptimizer",
        config=None,
    ):
        inputted = Input(df, target, file_path, url, config)
        df_after_input = inputted.handle_data()
        hana_df = dataframe.create_dataframe_from_pandas(
            connection_context=connection_context,
            pandas_df=df_after_input,
            table_name="test1",
            replace=True,
        )
        print(hana_df.columns)

    def optimizer(self):
        return self.opt

    @property
    def best_params(self):
        return self.opt.get_tuned_params()
