from numpy import ndarray

from pipeline.data import Data
from pipeline.input import Input
from pipeline.pipeline import Pipeline


class AutoML:
    def __init__(self):
        self.opt = None

    def fit(
        self,
        X_train: ndarray = None,
        y_train: ndarray = None,
        X_test=None,
        y_test=None,
        steps=10,
        target: str = None,
        file_path=None,
        url=None,
        columns_to_remove=None,
        categorical_features=None,
        optimizer="BayesianOptimizer",
        config=None,
    ):
        data = Data(X_train, X_test, y_train, y_test)
        inputted = Input(data, target, file_path, url, config)
        inputted.handle_data()
        data_after_input = inputted.handle_data()
        pipe = Pipeline(data_after_input, steps)
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
