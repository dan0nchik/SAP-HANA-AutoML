from algorithms.classification import DecisionTree
from algorithms.regression import Ridge
from optimizers.bayes import BayesianOptimizer
from pipeline.data import Data
from preprocess.preprocessor import Preprocessor


class Pipeline:
    def __init__(self, data: Data, steps, budget=None):
        self.data = data
        self.iter = steps

    def train(self):
        pr = Preprocessor()
        # Что это?
        dataframes = filter(lambda a: not a.startswith('__'), dir(self.data))
        pr.clean(self.data.X_train)
        pr.clean(self.data.X_test)
        algo_list, task = pr.set_task(self.data.y_train)
        for i in algo_list:
            opt = BayesianOptimizer(i, self.data, self.iter, task)
            opt.get_tuned_params()

