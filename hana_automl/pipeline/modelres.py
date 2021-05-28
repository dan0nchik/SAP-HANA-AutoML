import hana_automl
from hana_automl.preprocess.settings import PreprocessorSettings


class ModelBoard:
    """This class stores models that are shown in leaderboard."""

    def __init__(
        self,
        algorithm: hana_automl.algorithms.base_algo,
        train_accuracy: int,
        preprocessor: PreprocessorSettings,
    ):
        self.algorithm = algorithm
        self.train_accuracy = train_accuracy
        self.valid_accuracy = 0
        self.preprocessor = preprocessor

    def add_valid_acc(self, accuracy):
        self.valid_accuracy = accuracy
