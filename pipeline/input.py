import requests
import pandas as pd
from sklearn.model_selection import train_test_split
import io
from pipeline.data import Data
from utils.error import InputError


class Input:
    def __init__(self, data: Data,
                 target,
                 file_path,
                 url,
                 config):
        self.config = config
        self.data = data
        df = pd.DataFrame()
        if data.X_train or data.y_train is None:
            if url is not None:
                df = self.load_from_url(url)
            if file_path is not None:
                df = pd.read_csv(file_path)
            if target is None or target == '':
                raise InputError('No target variable provided!')
            if not isinstance(target, list):
                target = [target]
            y = df[target]
            X = df.drop(target, axis=1)
            self.data.X_train, self.data.X_test, self.data.y_train, self.data.y_test = self.split_data(X, y)

    def return_data(self):
        return self.data

    def split_data(self, X, y, random_state=42, test_size=0.33):
        if self.config is not None:
            test_size = self.config['test_size']
            random_state = self.config['random_state']
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=random_state,
                                                            test_size=test_size)
        return X_train, X_test, y_train, y_test

    def load_from_url(self, url):
        # TODO: url validation
        url_data = requests.get(url).content.decode('utf-8')
        return pd.read_csv(io.StringIO(url_data))
