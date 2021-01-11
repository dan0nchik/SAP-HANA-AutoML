from preprocessor import Preprocessor
from optimizer import DecisionTreeOptimizer
import models


class Pipeline:
    # TODO add x_test and y_test if needed
    def __init__(self, X_train, y_train, X_test=None, y_test=None, iterations=10):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.iter = iterations
        pass

    def train(self):
        pr = Preprocessor()
        dataframes = [self.X_train, self.y_train, self.X_test, self.y_test]
        for df in dataframes:
            pr.clean(df)
        model_list = pr.set_task(self.y_train)
        for model in model_list:
            # TODO: make regression optimizer and choose correct here
            opt = DecisionTreeOptimizer(model, self.X_train, self.y_train, self.X_test, self.y_test,
                                        self.iter)
            print(opt.search_hp())
            # {'target': 0.7966101694915254, 'params': {'max_depth': 1.586677627379058}}
            # TODO: fit model with tuned parameters and output it

