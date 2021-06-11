import os

import hana_ml.dataframe
import pandas as pd
import uuid
from hana_automl.pipeline.data import Data
from hana_ml.algorithms.pal.partition import train_test_val_split
from hana_ml.dataframe import create_dataframe_from_pandas
from typing import Union
from hana_automl.preprocess.preprocessor import Preprocessor
from hana_automl.utils.error import InputError
import pandas


class Input:
    """Handles input data. You can use it aside pipeline to load data to database.

    Attributes
    ----------
    connection_context : hana_ml.dataframe.ConnectionContext
        Connection info to HANA database.
    df : pandas.DataFrame or hana_ml.dataframe.DataFrame or str
        Pandas dataframe with data, or hana_ml dataframe, or string containing existing table name.
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
    verbose
        Level of output
    """

    def __init__(
        self,
        connection_context: hana_ml.ConnectionContext = None,
        df: Union[pandas.DataFrame, hana_ml.dataframe.DataFrame, str] = None,
        target: str = None,
        path: str = None,
        id_col: str = None,
        table_name: str = None,
        verbose: bool = True,
    ):
        self.df = df
        self.id_col = id_col
        self.file_path = path
        self.target = target
        self.table_name = table_name
        self.verbose = verbose
        self.hana_df: hana_ml.dataframe.DataFrame = None
        self.connection_context = connection_context

    def load_data(self):
        """Loads data to HANA database."""

        name = f"AUTOML{str(uuid.uuid4())}"
        if self.df is None and self.file_path is None and self.table_name is None:
            raise InputError("No data provided")
        if (
            isinstance(self.df, hana_ml.dataframe.DataFrame)
            and self.file_path is None
            and self.table_name is None
        ):
            self.hana_df = self.df
        elif (
            isinstance(self.df, str)
            and self.file_path is None
            and self.table_name is None
        ):
            if self.verbose:
                print(f"Connecting to existing table {self.df}")
            self.hana_df = self.connection_context.table(self.df)
        else:
            if (
                self.df is not None or self.file_path is not None
            ) and self.table_name is None:
                if self.file_path is not None:
                    self.df = self.download_data(self.file_path)
                if self.verbose:
                    print(f"Creating table with name: {name}")
                self.hana_df = create_dataframe_from_pandas(
                    self.connection_context,
                    self.df,
                    name,
                    disable_progressbar=not self.verbose,
                    drop_exist_tab=True,
                    force=True,
                )
                self.table_name = name
            elif (
                self.table_name is not None
                and self.table_name != ""
                and self.file_path is None
                and self.df is None
            ):
                if self.verbose:
                    print(f"Connecting to existing table {self.table_name}")
                self.hana_df = self.connection_context.table(self.table_name)
            elif self.table_name is not None and self.file_path is not None:
                if self.verbose:
                    print(f"Recreating table {self.table_name} with data from file")
                self.hana_df = create_dataframe_from_pandas(
                    self.connection_context,
                    self.download_data(self.file_path),
                    self.table_name,
                    force=True,
                    drop_exist_tab=True,
                    disable_progressbar=not self.verbose,
                )
            elif self.table_name is not None and self.df is not None:
                if self.verbose:
                    print(
                        f"Recreating table {self.table_name} with data from dataframe"
                    )
                self.hana_df = create_dataframe_from_pandas(
                    self.connection_context,
                    self.df,
                    self.table_name,
                    force=True,
                    drop_exist_tab=True,
                    disable_progressbar=not self.verbose,
                )
            self.hana_df.declare_lttab_usage(True)  # TODO: research
        if self.id_col is None:
            self.hana_df = self.hana_df.add_id(id_col="ID")
            self.id_col = "ID"

        # make id column UPPER
        self.hana_df = self.hana_df.rename_columns({self.id_col: self.id_col.upper()})
        self.id_col = self.id_col.upper()
        return

    def split_data(self) -> Data:
        """Splits single dataframe into multiple dataframes and passes them to Data.

        Returns
        -------
        Data
            Data with changes.
        """
        train, test, valid = train_test_val_split(
            data=self.hana_df, id_column=self.id_col, random_seed=17
        )
        return Data(train, test, valid, self.target, id_col=self.id_col)

    @staticmethod
    def download_data(path: str):
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
            if pd.read_csv(path).columns[0] == "Unnamed: 0":
                return pd.read_csv(path, index_col=0)
            else:
                return pd.read_csv(path)
        if file_type(path) == ".xlsx":
            if pd.read_excel(path).columns[0] == "Unnamed: 0":
                return pd.read_excel(path, index_col=0)
            else:
                return pd.read_excel(path)
        raise InputError("The file format is missing or not supported")


def file_type(file: str) -> str:
    """Return type of given file"""
    return os.path.splitext(file)[1]
