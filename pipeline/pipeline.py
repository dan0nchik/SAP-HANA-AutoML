from optimizers.bayes import BayesianOptimizer, ScikitBayesianOptimizer
from optimizers.grid_search import GridSearch
from pipeline.data import Data
from pipeline.leaderboard import Leaderboard
from preprocess.preprocessor import Preprocessor


class Pipeline:
    def __init__(self, data: Data, steps, budget=None):
        self.data = data
        self.iter = steps

    def train(self, columns_to_remove=None, categorical_features=None, optimizer=None):
        pr = Preprocessor()
        leaderboard = Leaderboard()
        # поместить в оптимайзер
        self.data = pr.clean(data=self.data, droplist_columns=columns_to_remove, categorical_list=categorical_features)
        algo_list, task = pr.set_task(self.data.y_train)
        while self.iter > 0:
            if optimizer == 'BayesianOptimizer':
                opt = ScikitBayesianOptimizer(algo_list, self.data, task)
            elif optimizer == 'GridSearch':
                opt = GridSearch(algo_list, self.data, 10, task)
            else:
                print('Optimizer not found. Bayesian optimizer will be used')
                opt = BayesianOptimizer(algo_list, self.data, task)
            print(opt.get_tuned_params())
            self.iter -= 1
