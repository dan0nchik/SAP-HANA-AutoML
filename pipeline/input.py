import os
import pandas as pd
import uuid
from pipeline.data import Data
from hana_ml.algorithms.pal.partition import train_test_val_split
from hana_ml.dataframe import create_dataframe_from_pandas
from utils.error import InputError
from pandas import DataFrame


class Input:
    """Handles input data.

    Attributes
    ----------
    connection_context : hana_ml.ConnectionContext
        Connection info to HANA database.
    df : DataFrame
        Pandas dataframe with data.
    id_col : str
        ID column for HANA table.
    file_path : str
        Path to data file.
    target : str
        Target variable that we want to predict.
    table_name : str
        Table's name in HANA database.
    hana_df : hana_ml.dataframe
        Converted HANA dataframe.
    """

    def __init__(
        self,
        connection_context=None,
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
        self.connection_context = connection_context

    def load_data(self):
        """Loads data to HANA database."""

        name = f"AUTOML{str(uuid.uuid4())}"

        if (
            self.df is not None or self.file_path is not None
        ) and self.table_name is None:
            if self.file_path is not None:
                self.df = self.download_data(self.file_path)
            print(f"Creating table with name: {name}")
            self.hana_df = create_dataframe_from_pandas(
                self.connection_context, self.df, name
            )
        elif (
            (self.table_name is not None or self.table_name != "")
            and self.file_path is None
            and self.df is None
        ):
            print(f"Connecting to existing table {self.table_name}")
            self.hana_df = self.connection_context.table(self.table_name)
        elif self.table_name is not None and self.file_path is not None:
            print(f"Recreating table {self.table_name} with data from file")
            self.hana_df = create_dataframe_from_pandas(
                self.connection_context,
                self.download_data(self.file_path),
                name,
                force=True,
            )
        elif self.table_name is not None and self.df is not None:
            print(f"Recreating table {self.table_name} with data from dataframe")
            self.hana_df = create_dataframe_from_pandas(
                self.connection_context, self.df, name, force=True
            )
        else:
            raise InputError("No data provided")
        print("Done")
        return

    def split_data(self) -> Data:
        """Splits single dataframe into multiple dataframes and passes them to Data.

        Returns
        -------
        Data
            Data with changes.
        """
        train, test, valid = train_test_val_split(data=self.hana_df)
        return Data(train, test, valid, self.target, id_col=self.id_col)

    @staticmethod
    def download_data(path):
        """Downloads data from path

        Parameters
        ----------
        path : str
            Path/url to the file.

        Raises
        ------
        InputError
            If file format is wrong.
        """
        if path == "":
            raise InputError("Please provide valid file path or url")
        if file_type(path) == ".csv":
            return pd.read_csv(path)
        if file_type(path) == ".xlsx":
            return pd.read_excel(path)
        raise InputError("The file format is missing or not supported")


def file_type(file: str) -> str:
    """Return type of given file"""
    return os.path.splitext(file)[1]
