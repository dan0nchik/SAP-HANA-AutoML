from pipeline.input import Input
from pipeline.pipeline import Pipeline
from numpy import ndarray


class AutoML:

    def fit(self, X_train: ndarray = None,
            y_train: ndarray = None,
            x_test=None,
            y_test=None,
            steps=20,
            target: str = None,
            file_path=None,
            url=None,
            config=None):
        input_data = Input(X_train, y_train, x_test, y_test, target, file_path,
                           url, config)
        pipe = Pipeline(input_data, steps)
        pipe.train()
