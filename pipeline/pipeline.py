from optimizers.bayes import BayesianOptimizer
from optimizers.optuna_optimizer import OptunaOptimizer
from pipeline.data import Data
from pipeline.leaderboard import Leaderboard
from preprocess.preprocessor import Preprocessor
import copy

from utils.error import PipelineError


class Pipeline:
    def __init__(self, data: Data, steps, budget=None):
        self.data = data
        self.iter = steps
        self.opt = None

    def train(self, columns_to_remove=None, categorical_features=None, optimizer=None):
        pr = Preprocessor()
        print(self.data.train.columns)
        self.data.train.drop(['Name'])
        print(self.data.train.columns)
