from preprocess.preprocessor import Preprocessor
from optimizers.baseoptimizer import BaseOptimizer
from utils.logger import output
from input import Input
from algorithms.classification import DecisionTree


class Pipeline:
    def __init__(self, input: Input, steps):
        self.input = input
        self.iter = steps
        self.X_train = input.X_train
        self.y_train = input.y_train
        self.X_test = input.X_test
        self.y_test = input.y_test
        pass

    def train(self):
        pr = Preprocessor()
        dataframes = [self.X_train, self.y_train, self.X_test, self.y_test]
        for df in dataframes:
            pr.clean(df)
        task = pr.set_task(self.y_train)
        algo_list = []

