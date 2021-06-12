from hana_ml import DataFrame
from hana_ml.algorithms.pal.partition import train_test_val_split

from hana_automl.preprocess.preprocessor import Preprocessor
import pandas as pd

pd.options.display.max_columns = None
pd.options.display.max_rows = None


class Data:
    """We needed to reuse and store data from dataset in one place, so we've created this class.

    Attributes
    ----------
    train: DataFrame
        Train part of dataset
    test: DataFrame
        Test part of dataset (30% of all data)
    valid: DataFrame
        Validation part of dataset for model evaluation in the end of the process (10-15% of all data)
    id_colm: str
        ID column. Needed for HANA.

    """

    def __init__(
        self,
        train: DataFrame = None,
        test: DataFrame = None,
        valid: DataFrame = None,
        target: str = None,
        id_col: str = None,
    ):
        self.train = train
        self.test = test
        self.valid = valid
        self.target = target
        self.id_colm = id_col
        self.binomial = None
        self.strategy_by_col = None

    def drop(self, droplist_columns: list):
        """Drops columns in table

        Parameters
        ----------
        droplist_columns : list
            Columns to remove.
        """
        pr = Preprocessor()
        self.valid = pr.removecolumns(droplist_columns, df=self.valid)
        self.train = pr.removecolumns(droplist_columns, df=self.train)
        self.test = pr.removecolumns(droplist_columns, df=self.test)

    def clear(
        self,
        num_strategy: str = "mean",
        categorical_list: list = None,
        normalizer_strategy: str = "min-max",
        normalizer_z_score_method: str = "",
        normalize_int: bool = False,
        strategy_by_col: list = None,
        drop_outers: bool = False,
        normalization_excp: list = None,
        clean_sets: list = ["test", "train", "valid"],
    ):
        """Clears data using methods defined in parameters.

        Parameters
        ----------
        num_strategy : str
            Strategy to decode numeric variables.
        dropempty : Bool
            Drop empty rows or not.
        categorical_list : list
            List of categorical features.
        normalizer_strategy: str
            Strategy for normalization. Defaults to 'min-max'.
        normalizer_z_score_method : str
            A z-score (also called a standard score) gives you an idea of how far from the mean a data point is
        normalize_int : bool
            Normalize integers or not
        strategy_by_col: ListOfTuples
            Specifies the imputation strategy for a set of columns, which overrides the overall strategy for data imputation.
            Each tuple in the list should contain at least two elements, such that: the 1st element is the name of a column;
            the 2nd element is the imputation strategy of that column(For numerical: "mean", "median", "delete", "als", 'numerical_const'. Or categorical_const for categorical).
            If the imputation strategy is 'categorical_const' or 'numerical_const', then a 3rd element must be included in the tuple, which specifies the constant value to be used to substitute the detected missing values in the column
        clean_sets: ListOfStrings
            Specifies parts of dataset, that will be preprocessed. List should contain 'test','train' or 'valid'. Other values will be ignored


        Returns
        -------
        Data: Data
            Data with changes.

        """
        pr = Preprocessor()
        valid = self.valid
        train = self.train
        test = self.test
        if drop_outers:
            df = test.union([train, valid])
            df = df.sort(self.id_colm, desc=False)
            df = pr.drop_outers(
                df, id=self.id_colm, target=self.target, cat_list=categorical_list
            )
            train, test, valid = train_test_val_split(
                data=df, id_column=self.id_colm, random_seed=17
            )
        if "valid" in clean_sets:
            valid = pr.autoimput(
                df=self.valid,
                id=self.id_colm,
                target=self.target,
                imputer_num_strategy=num_strategy,
                strategy_by_col=strategy_by_col,
                categorical_list=categorical_list,
                normalizer_strategy=normalizer_strategy,
                normalizer_z_score_method=normalizer_z_score_method,
                normalize_int=normalize_int,
                normalization_excp=normalization_excp,
            )
        if "train" in clean_sets:
            train = pr.autoimput(
                df=self.train,
                id=self.id_colm,
                target=self.target,
                imputer_num_strategy=num_strategy,
                strategy_by_col=strategy_by_col,
                categorical_list=categorical_list,
                normalizer_strategy=normalizer_strategy,
                normalizer_z_score_method=normalizer_z_score_method,
                normalize_int=normalize_int,
                normalization_excp=normalization_excp,
            )
        if "test" in clean_sets:
            test = pr.autoimput(
                df=self.test,
                id=self.id_colm,
                target=self.target,
                imputer_num_strategy=num_strategy,
                strategy_by_col=strategy_by_col,
                categorical_list=categorical_list,
                normalizer_strategy=normalizer_strategy,
                normalizer_z_score_method=normalizer_z_score_method,
                normalize_int=normalize_int,
                normalization_excp=normalization_excp,
            )
        return Data(
            train=train, test=test, valid=valid, target=self.target, id_col=self.id_colm
        )

    def check_norm_except(self, categorical_list):
        return Preprocessor.check_normalization_exceptions(
            df=self.test.union([self.train, self.valid]).sort(self.id_colm, desc=False),
            id=self.id_colm,
            target=self.target,
            categorical_list=categorical_list,
        )

    def drop_duplicates(self):
        df = self.test.union([self.train, self.valid])
        cols = df.columns
        cols.remove(self.id_colm)
        df = df.drop_duplicates(cols)
        self.train, self.test, self.valid = train_test_val_split(
            data=df, id_column=self.id_colm, random_seed=17
        )
