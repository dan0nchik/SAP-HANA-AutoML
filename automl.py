from pipeline.input_ import Input
from pipeline.pipeline import Pipeline
from numpy import ndarray
from pipeline.data import Data


class AutoML:

    def fit(self, X_train: ndarray = None,
            y_train: ndarray = None,
            X_test=None,
            y_test=None,
            steps=20,
            target: str = None,
            file_path=None,
            url=None,
            colmnsforremv=None,
            categorical=None,
            config=None):
        data = Data(X_train, X_test, y_train, y_test)
        data_after_input = Input(data, target, file_path,
                                 url, config).return_data()

        pipe = Pipeline(data_after_input, steps)
        pipe.train(colmnsforremv=colmnsforremv, categorical=categorical)
