from optimizers.bayes import BayesianOptimizer
from optimizers.grid_search import GridSearch
from pipeline.data import Data
from preprocess.preprocessor import Preprocessor


class Pipeline:
    def __init__(self, data: Data, steps, budget=None):
        self.data = data
        self.iter = steps

    def train(self, columns_to_remove=None, categorical_features=None, optimizer=None):
        pr = Preprocessor()
        self.data = pr.clean(data=self.data, droplist_columns=columns_to_remove, categorical_list=categorical_features)
        algo_list, task = pr.set_task(self.data.y_train)
        for alg in algo_list:
            if optimizer == 'BayesianOptimizer':
                opt = BayesianOptimizer(alg, self.data, self.iter, task)
            elif optimizer == 'GridSearch':
                opt = GridSearch(alg, self.data, self.iter, task)
            else:
                print('Optimizer not found. Bayesian optimizer will be used')
                opt = BayesianOptimizer(alg, self.data, self.iter, task)
            opt.get_tuned_params()
