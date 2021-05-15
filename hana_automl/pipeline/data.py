from hana_ml import DataFrame

from hana_automl.preprocess.preprocessor import Preprocessor


class Data:
    def __init__(
            self, train: DataFrame = None, test: DataFrame = None, valid: DataFrame = None, target=None, id_col=None
    ):
        self.train = train
        self.test = test
        self.valid = valid
        self.target = target
        self.id_colm = id_col
        self.binomial = None

    def drop(self, droplist_columns):
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
            num_strategy="mean",
            cat_strategy=None,
            dropempty=False,
            categorical_list=None,
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

        Returns
        -------
        Data
            Data with changes.

        """
        pr = Preprocessor()
        valid = pr.autoimput(
            df=self.valid,
            num_strategy=num_strategy,
            cat_strategy=cat_strategy,
            dropempty=dropempty,
            categorical_list=categorical_list,
        )
        train = pr.autoimput(
            df=self.train,
            num_strategy=num_strategy,
            cat_strategy=cat_strategy,
            dropempty=dropempty,
            categorical_list=categorical_list,
        )
        test = pr.autoimput(
            df=self.test,
            num_strategy=num_strategy,
            cat_strategy=cat_strategy,
            dropempty=dropempty,
            categorical_list=categorical_list,
        )
        return Data(
            train=train, test=test, valid=valid, target=self.target, id_col=self.id_colm
        )
