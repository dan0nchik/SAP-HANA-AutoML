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
        self.hana_df = None

    def load_data(self):
        name = f"AUTOML{str(uuid.uuid4())}"
        with open("utils/tables.txt", "a+") as file:
            file.write(name + "\n")
            file.close()

        if (
            self.df is not None or self.file_path is not None
        ) and self.table_name is None:
            if self.file_path is not None:
                self.df = self.download_data(self.file_path)
            print(f"Creating table with name: {name}")
            self.hana_df = create_dataframe_from_pandas(
                connection_context, self.df, name
            )
        elif (
            self.table_name is not None or self.table_name != ""
        ) and self.file_path is None:
            print(f"Connecting to existing table {self.table_name}")
            self.hana_df = connection_context.table(self.table_name)
        elif self.table_name is not None and self.file_path is not None:
            print(f"Recreating table {self.table_name} with data from file")
            self.hana_df = create_dataframe_from_pandas(
                connection_context, self.download_data(self.file_path), name, force=True
            )
        elif self.table_name is not None and self.df is not None:
            print(f"Recreating table with data from dataframe")
            self.hana_df = create_dataframe_from_pandas(
                connection_context, self.df, name, force=True
            )
        else:
            raise InputError("No data provided")
        print("Done")
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
