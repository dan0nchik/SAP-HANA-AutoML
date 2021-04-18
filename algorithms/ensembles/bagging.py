from pipeline.leaderboard import Leaderboard
from utils.error import BaggingError


class Bagging:

    def __init__(self, categorical_features, id_col, model_list: dict = None, leaderboard: Leaderboard = None):
        self.id_col = id_col
        self.categorical_features = categorical_features
        self.title = ""
        if model_list is None and leaderboard is None:
            raise BaggingError("Provide list of models or a leaderboard for ensemble creation")
        if model_list is not None:
            self.model_list = model_list
        else:
            self.model_list = leaderboard.board[:3]

    def score(self, data, df):
        pass

    def predict(self, cat):
        pass


