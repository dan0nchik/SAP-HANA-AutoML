from pipeline.data import Data
from preprocess.preprocessor import Preprocessor
from optimizers.base_optimizer import BaseOptimizer
from optimizers.bayes import BayesianOptimizer
from utils.logger import output
from pipeline.input_ import Input
from algorithms.classification.DecisionTree import DecisionTree


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
        algo_list = [DecisionTree()]
        opt = BayesianOptimizer(DecisionTree(), self.data, self.iter, 'cls')
        opt.get_tuned_params()

