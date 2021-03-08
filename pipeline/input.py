import requests
import pandas as pd
from sklearn.model_selection import train_test_split
import io
from pipeline.data import Data
from utils.error import InputError


class Input:
    def __init__(
        self,
        df: pd.DataFrame = None,
        target=None,
        file_path=None,
        url=None,
        config=None,
    ):
        self.config = config
        self.df = df
        self.url = url
        self.target = target
        self.file_path = file_path

    def handle_data(self):
        if self.df is None:
            if self.url is not None:
                self.df = self.load_from_url(self.url)
            if self.file_path is not None:
                self.df = self.read_from_file(self.file_path)
        return self.df

    def split_data(self, X, y, random_state=42, test_size=0.33):
        if self.config is not None:
            test_size = self.config["test_size"]
            random_state = self.config["random_state"]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, random_state=random_state, test_size=test_size
        )
        return X_train, X_test, y_train, y_test

    def load_from_url(self, url):
        # TODO: url validation & more file typess
        url_data = requests.get(url).content.decode("utf-8")
        return pd.read_csv(io.StringIO(url_data))

    def read_from_file(self, file_path):
        # TODO: more file types
        return pd.read_csv(file_path)
