import pandas as pd


class Preprocessor:
    def __init__(self, X_train, X_test, y_train, y_test, params):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.params = params

        for df in locals()[:1]:
            self.clean(df)

    def clean(self, df):
        pass
