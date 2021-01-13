from pipeline.validator import Validate
from pipeline.fit import Fit
from algorithms import base_algo


class BaseOptimizer:

    def objective(self, **hyperparameters):
        self.algorithm.set_params(**hyperparameters)
        Fit.fit(self.algorithm, self.X_train, self.y_train)
        return Validate.val(self.algorithm, self.X_test, self.y_test, self.problem)

    def __init__(self, algorithm, data, iterations, problem):
        self.X_train = data.X_train
        self.y_train = data.y_train
        self.X_test = data.X_test
        self.y_test = data.y_test
        self.algorithm = algorithm
        self.iter = iterations
        self.problem = problem
        self.tuned = {}

    def get_tuned_params(self):
        return self.tuned
