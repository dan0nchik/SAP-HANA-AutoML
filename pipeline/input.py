import os
import pandas as pd
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
        path: str = None,
        id_col=None,
        table_name: str = None,
    ):
        self.df = df
        self.id_col = id_col
        self.file_path = path
        self.target = target
        self.table_name = table_name

    def load_data(self):
        if self.df is not None:
            pass
        elif self.file_path is not None:
            self.df = self.download_data(self.file_path)
        elif self.table_name is not None or self.table_name != "":
            print(f"Connecting to existing table {self.table_name}")
            self.hana_df = connection_context.table(self.table_name)
            print("Connected")
            return
        else:
            raise InputError("No data provided")

        name = f"AUTOML{str(uuid.uuid4())}"
        print(f"Creating table with name: {name}")
        self.hana_df = create_dataframe_from_pandas(
            connection_context, self.df, name, force=True
        )
        print(f"Done")
        return

    def split_data(self) -> Data:
        train, test, valid = train_test_val_split(data=self.hana_df)
        return Data(train, test, valid, self.target, id_col=self.id_col)
    @staticmethod
    def download_data(path):
        if path == "":
            raise InputError("Please provide valid file path or url")
        if file_type(path) == ".csv":
            return pd.read_csv(path)
        if file_type(path) == ".xlsx":
            return pd.read_excel(path)
        raise InputError("The file format is missing or not supported")


def file_type(file: str) -> str:
    return os.path.splitext(file)[1]
