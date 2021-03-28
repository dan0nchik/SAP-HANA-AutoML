import io
import os

import pandas as pd
import requests
from hana_ml.algorithms.pal.partition import train_test_val_split
from hana_ml.dataframe import create_dataframe_from_pandas

from pipeline.data import Data
from utils.connection import connection_context
from utils.error import InputError


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
        self.hana_df = None

    def load_data(self):
        if self.df is None:
            if self.url is not None:
                self.df = self.load_from_url(self.url)
            if self.file_path is not None:
                self.df = self.read_from_file(self.file_path)
        self.hana_df = create_dataframe_from_pandas(connection_context, self.df, f'data', replace=True,
                                                    drop_exist_tab=True, force=True)
        print('yes')

    def split_data(self) -> Data:
        train, test, valid = train_test_val_split(data=self.hana_df)
        return Data(train, test, valid)

    def load_from_url(self, url):
        if url == "":
            raise InputError("Please provide valid url")
        url_data = requests.get(url).content.decode("utf-8")
        if get_url_file_type(url) == ".csv":
            return pd.read_csv(io.StringIO(url_data))
        if get_url_file_type(url) == ".xlsx":
            return pd.read_excel(io.StringIO(url_data))

    def read_from_file(self, file_path):
        if file_path == "":
            raise InputError("Please provide valid file path")
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
