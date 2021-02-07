from optimizers.bayes import BayesianOptimizer
from optimizers.grid_search import GridSearch
from optimizers.optuna_optimizer import OptunaOptimizer
from pipeline.data import Data
from pipeline.leaderboard import Leaderboard
from preprocess.preprocessor import Preprocessor
import copy


class Pipeline:
    def __init__(self, data: Data, steps, budget=None):
        self.data = data
        self.iter = steps

    def train(self, columns_to_remove=None, categorical_features=None, optimizer=None):
        pr = Preprocessor()
        data_copy = copy.deepcopy(self.data)
        data_copy = pr.clean(
            data=data_copy,
            droplist_columns=columns_to_remove,
            categorical_list=categorical_features,
        )
        algo_list, task, algo_names_list = pr.set_task(data_copy.y_train)
        if optimizer == "BayesianOptimizer":
            opt = BayesianOptimizer(
                algo_list=algo_list,
                data=self.data,
                iterations=self.iter,
                categorical_list=categorical_features,
                droplist_columns=columns_to_remove,
                problem=task,
            )
        elif optimizer == "GridSearch":
            opt = GridSearch(algo_list, self.data, 10, task)
        elif optimizer == "OptunaSearch":
            opt = OptunaOptimizer(
                algo_list=algo_list,
                data=self.data,
                problem=task,
                iterations=self.iter,
                algo_names=algo_names_list,
                categorical_features=categorical_features,
                droplist_columns=columns_to_remove,
            )
        else:
            print("Optimizer not found. Bayesian optimizer will be used")
            opt = BayesianOptimizer(
                algo_list=algo_list,
                data=self.data,
                iterations=self.iter,
                categorical_list=categorical_features,
                droplist_columns=columns_to_remove,
                problem=task,
            )
        opt.get_tuned_params()
