from optimizers.bayes import BayesianOptimizer
from optimizers.optuna_optimizer import OptunaOptimizer
from pipeline.data import Data
from pipeline.leaderboard import Leaderboard
from preprocess.preprocessor import Preprocessor
import copy

from utils.error import PipelineError


class Pipeline:
    def __init__(self, data: Data, steps):
        self.data = data
        self.iter = steps
        self.opt = None

    def train(self, categorical_features=None, optimizer=None):
        pr = Preprocessor()
        algo_list, task, algo_dict = pr.set_task(self.data, target=self.data.target)
        print(task)
        if optimizer == "BayesianOptimizer":
            self.opt = BayesianOptimizer(
                algo_list=algo_list,
                data=self.data,
                iterations=self.iter,
                categorical_list=categorical_features,
                problem=task,
            )
        elif optimizer == "OptunaSearch":
            self.opt = OptunaOptimizer(
                algo_list=algo_list,
                data=self.data,
                problem=task,
                iterations=self.iter,
                algo_dict=algo_dict,
                categorical_features=categorical_features,
            )
        else:
            raise PipelineError("Optimizer not found!")
        self.opt.tune()
        return self.opt
