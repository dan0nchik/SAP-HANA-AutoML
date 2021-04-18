from algorithms.ensembles.bagging import Bagging
from pipeline.leaderboard import Leaderboard
from preprocess.preprocessor import Preprocessor
from utils.error import BaggingError


class BaggingCls(Bagging):
    def __init__(self, categorical_features, id_col, model_list: list = None, leaderboard: Leaderboard = None):
        super(BaggingCls, self).__init__(categorical_features, id_col, model_list, leaderboard)
        self.title = "BaggingClassifier"

    def score(self, data=None, df=None):
        predictions = list()
        if data is None and df is None:
            raise BaggingError("Provide valid data for accuracy estimation")
        pr = Preprocessor()
        for model in self.model_list:
            if df is not None:
                df2 = pr.clean(
                    data=df, num_strategy=model.preprocessor['imputer']
                )
            else:
                print(data.valid.drop(data.target).hasna())
                df2 = pr.clean(
                    data=data.valid.drop(data.target), num_strategy=model.preprocessor['imputer']
                )
                print(df2.hasna())
            pred = model.algorithm.model.predict(df2, self.id_col)
            if type(pred) == tuple:
                predictions.append(pred[0])
            else:
                predictions.append(pred)
        for res in predictions:
            print(res.collect().head())

    def predict(self, cat):
        pass
