from hana_ml import DataFrame

import hana_automl
from hana_automl.preprocess.preprocessor import Preprocessor


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
        cat_strategy=None,
        dropempty: bool = False,
        categorical_list: list = None,
        normalizer_strategy: str = "min-max",
        normalizer_z_score_method: str = "",
        normalize_int: bool = False,
    ):
        """Clears data using methods defined in parameters.

        Parameters
        ----------
        num_strategy : str
            Strategy to decode numeric variables.
        cat_strategy : str
            Strategy to decode categorical variables.
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


        Returns
        -------
        Data: Data
            Data with changes.

        """
        pr = Preprocessor()
        valid = pr.autoimput(
            df=self.valid,
            id=self.id_colm,
            target=self.target,
            imputer_num_strategy=num_strategy,
            cat_strategy=cat_strategy,
            dropempty=dropempty,
            categorical_list=categorical_list,
            normalizer_strategy=normalizer_strategy,
            normalizer_z_score_method=normalizer_z_score_method,
            normalize_int=normalize_int,
        )
        train = pr.autoimput(
            df=self.train,
            id=self.id_colm,
            target=self.target,
            imputer_num_strategy=num_strategy,
            cat_strategy=cat_strategy,
            dropempty=dropempty,
            categorical_list=categorical_list,
            normalizer_strategy=normalizer_strategy,
            normalizer_z_score_method=normalizer_z_score_method,
            normalize_int=normalize_int,
        )
        test = pr.autoimput(
            df=self.test,
            id=self.id_colm,
            target=self.target,
            imputer_num_strategy=num_strategy,
            cat_strategy=cat_strategy,
            dropempty=dropempty,
            categorical_list=categorical_list,
            normalizer_strategy=normalizer_strategy,
            normalizer_z_score_method=normalizer_z_score_method,
            normalize_int=normalize_int,
        )
        return Data(
            train=train, test=test, valid=valid, target=self.target, id_col=self.id_colm
        )
