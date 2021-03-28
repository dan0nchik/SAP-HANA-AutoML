from preprocess.preprocessor import Preprocessor


class Data:
    def __init__(self, train, test, valid, target):
        self.train = train
        self.test = test
        self.valid = valid
        self.target = target

    def drop(self, droplist_columns):
        pr = Preprocessor()
        self.valid = pr.removecolumns(droplist_columns, df=self.valid)
        self.train = pr.removecolumns(droplist_columns, df=self.train)
        self.test = pr.removecolumns(droplist_columns, df=self.test)
