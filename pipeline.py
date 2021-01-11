from preprocessor import Preprocessor
from optimizer import Optimizer
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
        # for df in dataframes:
        #     pr.clean(df)
        model_list = pr.set_task(self.y_train)
        for model in model_list:
            # TODO: make regression optimizer and choose correct here
            opt = Optimizer(model[0], self.X_train, self.y_train, self.X_test, self.y_test,
                            self.iter, model[1])
            print(opt.search_hp())
            # TODO: fit model with tuned parameters and output it
