import requests
import pandas as pd
from sklearn.model_selection import train_test_split
import io
import numpy as np


class Input:
    def __init__(self, X_train, y_train, X_test, y_test, target, file_path,
                 url, config):
        self.config = config
        df = pd.DataFrame()
        if X_train or y_train is None:
            if url is not None:
                # TODO: url validation
                data = requests.get(url).content.decode('utf-8')
                df = pd.read_csv(io.StringIO(data))
            if file_path is not None:
                # TODO: Add other file types
                df = pd.read_csv(file_path)
            if target is None:
                # TODO: throw excep. instead of print
                print("Enter target value!")
                return
            if not isinstance(target, list):
                target = [target]
            y = df[target]
            X = df.drop(target, axis=1)
            self.X_train, self.X_test, self.y_train, self.y_test = self.split_data(X, y)
        else:
            self.X_train, self.y_train = X_train, y_train
        if X_test is not None:
            self.X_test, self.y_test = X_test, y_test

    def split_data(self, X, y, random_state=42, test_size=0.33):
        if self.config is not None:
            test_size = self.config['test_size']
            random_state = self.config['random_state']
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=random_state,
                                                            test_size=test_size)
        return X_train, X_test, y_train, y_test
