from hana_ml import DataFrame

from preprocess.preprocessor import Preprocessor


class Data:
    def __init__(
        self, train: DataFrame, test: DataFrame, valid: DataFrame, target, id_col
    ):
        self.train = train
        self.test = test
        self.valid = valid
        self.target = target
        self.id_colm = id_col

    def drop(self, droplist_columns):
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
