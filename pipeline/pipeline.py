from optimizers.bayes import BayesianOptimizer
from pipeline.data import Data
from preprocess.preprocessor import Preprocessor


class Pipeline:
    def __init__(self, data: Data, steps, budget=None):
        self.data = data
        self.iter = steps

    def train(self, colmnsforremv=None, categorical=None):
        pr = Preprocessor()
        self.data = pr.clean(data=self.data, droplist_columns=colmnsforremv, categorlist=categorical)
        algo_list, task = pr.set_task(self.data.y_train)
        for alg in algo_list:
            opt = BayesianOptimizer(alg, self.data, self.iter, task)
            opt.get_tuned_params()
