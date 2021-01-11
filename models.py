import requests
import pandas as pd
from sklearn.model_selection import train_test_split
import io
from pipeline import Pipeline
from sklearn.metrics import accuracy_score, r2_score


class Model:

    def __init__(self):
        self.df = pd.DataFrame()
        self.config = None
        self.y = None
        self.X = None
        pass

    def automl(self, X_train=None, y_train=None, x_test=None, y_test=None, iterations=20, target=None, file_path=None,
               url=None, config=None):
        if X_train or y_train is None:
            if url is not None:
                # TODO: url validation
                data = requests.get(url).content.decode('utf-8')
                self.df = pd.read_csv(io.StringIO(data))
            if file_path is not None:
                self.df = pd.read_csv(file_path)
            self.config = config
            if target is None:
                print("Enter target value!")
                return
            if not isinstance(target, list):
                target = [target]
            self.y = self.df[target]
            self.X = self.df.drop(target, axis=1)
            X_train, X_test, y_train, y_test = self.split_data()
            pipe = Pipeline(X_train, y_train, X_test, y_test, iterations=iterations)
        else:
            pipe = Pipeline(X_train, y_train, x_test, y_test, iterations=iterations)
        pipe.train()

    def split_data(self, random_state=42, test_size=0.33):
        if self.config is not None:
            test_size = self.config['test_size']
            random_state = self.config['random_state']
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, random_state=random_state,
                                                            test_size=test_size)
        return X_train, X_test, y_train, y_test


class Validate:
    @staticmethod
    def val(model, X_test, y_test, metrics=None):
        # TODO get metrics from config (?)
        pred = model.predict(X_test)
        # TODO understand model's class to find out right metric
        # return r2_score(y_test, pred)
        return accuracy_score(y_test, pred)


class Fit:
    @staticmethod
    def fit(model, X_train, y_train):
        model.fit(X_train, y_train)
