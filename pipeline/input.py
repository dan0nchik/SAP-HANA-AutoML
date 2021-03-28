import io
import os
import pandas as pd
import requests
import hana_ml
import uuid
from pipeline.data import Data
from hana_ml.algorithms.pal.partition import train_test_val_split
from utils.connection import connection_context
from hana_ml.dataframe import create_dataframe_from_pandas
from utils.error import InputError


class Input:
    def __init__(
            self,
            df: pd.DataFrame = None,
            target=None,
            file_path=None,
            url=None,
            id_col=None,
            table_name=None
    ):
        self.df = df
        self.id_col = id_col
        self.url = url
        self.target = target
        self.file_path = file_path
        self.table_name = table_name

    def load_data(self):
        if self.df is None:
            if self.url is not None:
                self.df = self.load_from_url(self.url)
            if self.file_path is not None:
                self.df = self.read_from_file(self.file_path)
        if self.table_name is None:
            name = f'AUTOML{str(uuid.uuid4())}'
            print(f"Creating table with name: {name}")
            self.hana_df = create_dataframe_from_pandas(connection_context, self.df, name, force=True)
            print(f"Done")
        else:
            print(f"Connecting to existing table {self.table_name}")
            self.hana_df = connection_context.table(self.table_name)
            print("Connected")

    def split_data(self) -> Data:
        train, test, valid = train_test_val_split(data=self.hana_df)
        return Data(train, test, valid, self.target, id_col=self.id_col)

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
