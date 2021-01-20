from numpy import ndarray

from pipeline.data import Data
from pipeline.input import Input
from pipeline.pipeline import Pipeline


class AutoML:

    def fit(self, X_train: ndarray = None,
            y_train: ndarray = None,
            X_test=None,
            y_test=None,
            steps=20,
            target: str = None,
            file_path=None,
            url=None,
            columns_to_remove=None,
            categorical_features=None,
            optimizer='BayesianOptimizer',
            config=None):
        data = Data(X_train, X_test, y_train, y_test)
        data_after_input = Input(data, target, file_path,
                                 url, config).return_data()

        pipe = Pipeline(data_after_input, steps)
        pipe.train(columns_to_remove=columns_to_remove, categorical_features=categorical_features, optimizer=optimizer)
