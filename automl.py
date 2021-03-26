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
from hana_ml.algorithms.pal.linear_model import  LogisticRegression
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
            columns_to_remove: list = None,
            categorical_features: list = None,
            optimizer: str = "BayesianOptimizer",
            config=None,
    ):
        inputted = Input(df, target, file_path, url).handle_data()
        print(inputted)
        hana_df = hana_ml.dataframe.create_dataframe_from_pandas(
            connection_context=connection_context,
            pandas_df=inputted,
            table_name="FROMURL",
            force=True,
            replace=True,
            drop_exist_tab=True,
        )
        print(hana_df.columns)
        lr = LogisticRegression(solver='newton',

                                                                    thread_ratio=0.1, max_iter=1000,

                                                                    pmml_export='single-row',

                                                                    stat_inf=True, tol=0.000001)
        lr.fit(hana_df,label='Survived')
        print('fitted')

    def optimizer(self):
        return self.opt

    @property
    def best_params(self):
        return self.opt.get_tuned_params()
