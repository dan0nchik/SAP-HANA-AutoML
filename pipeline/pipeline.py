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
        dataframes = filter(lambda a: not a.startswith('__'), dir(self.data))
        # for df in dataframes:
        #     pr.clean(df)
        task = pr.set_task(self.data.y_train)
        algo_list = []
        if task == 'cls':
            algo_list = [DecisionTree()]
        if task == 'reg':
            algo_list = [Ridge()]
        for i in algo_list:
            opt = BayesianOptimizer(i, self.data, self.iter, task)
            opt.get_tuned_params()

