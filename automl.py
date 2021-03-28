import pandas as pd

from pipeline.input import Input
from pipeline.pipeline import Pipeline


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

        inputted = Input(df, target, file_path, url)
        inputted.load_data()
        data = inputted.split_data()
        pipe = Pipeline(data, steps)
        self.opt = pipe.train(
            columns_to_remove=columns_to_remove,
            categorical_features=categorical_features,
            optimizer=optimizer,
        )

    def optimizer(self):
        return self.opt

    @property
    def best_params(self):
        return self.opt.get_tuned_params()
