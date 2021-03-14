import requests
import pandas as pd
from sklearn.model_selection import train_test_split
import io
import os
from omegaconf import DictConfig, OmegaConf
import hydra


class Input:
    def __init__(
        self,
        df: pd.DataFrame = None,
        target=None,
        file_path=None,
        url=None,
    ):
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

    def load_from_url(self, url):
        url_data = requests.get(url).content.decode("utf-8")
        if get_url_file_type(url) == ".csv":
            return pd.read_csv(io.StringIO(url_data))
        if get_url_file_type(url) == ".xlsx":
            return pd.read_excel(io.StringIO(url_data))

    def read_from_file(self, file_path):
        if file_type(file_path) == ".csv":
            return pd.read_csv(file_path)
        if file_type(file_path) == ".xlsx":
            return pd.read_excel(file_path)


def file_type(file: str) -> str:
    return os.path.splitext(file)[1]


def get_url_file_type(url: str) -> str:
    extension = ""
    for letter in url[::-1]:
        extension += letter
        if letter == ".":
            extension = extension[::-1]
            return extension
