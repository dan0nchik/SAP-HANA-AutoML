from hana_automl.preprocess.settings import PreprocessorSettings


class ModelBoard:
    """This class stores models that are shown in leaderboard."""

    def __init__(
        self,
        algorithm,
        train_score: float,
        preprocessor: PreprocessorSettings,
    ):
        self.algorithm = algorithm
        self.train_score = train_score
        self.valid_score = 0
        self.preprocessor = preprocessor

    def add_valid_score(self, accuracy):
        self.valid_score = accuracy
