from optimizers.bayes import BayesianOptimizer
from optimizers.optuna_optimizer import OptunaOptimizer
from pipeline.data import Data
from pipeline.leaderboard import Leaderboard
from preprocess.preprocessor import Preprocessor
import copy


class Pipeline:
    def __init__(self, data: Data, steps, budget=None):
        self.data = data
        self.iter = steps
        self.opt = None

    def train(self, columns_to_remove=None, categorical_features=None, optimizer=None):
        pr = Preprocessor()
        data_copy = copy.deepcopy(self.data)
        algo_list, task, algo_dict = pr.set_task(data_copy.y_train)
        if optimizer == "BayesianOptimizer":
            self.opt = BayesianOptimizer(
                algo_list=algo_list,
                data=self.data,
                iterations=self.iter,
                categorical_list=categorical_features,
                droplist_columns=columns_to_remove,
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
                droplist_columns=columns_to_remove,
            )
        else:
            print("Optimizer not found. Bayesian optimizer will be used")
            self.opt = BayesianOptimizer(
                algo_list=algo_list,
                data=self.data,
                iterations=self.iter,
                categorical_list=categorical_features,
                droplist_columns=columns_to_remove,
                problem=task,
            )
        self.opt.tune()
        return self.opt
