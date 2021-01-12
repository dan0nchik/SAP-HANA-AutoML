from pipeline.validator import Validate
from pipeline.fit import Fit
from algorithms import base_algo


class BaseOptimizer:

    def objective(self, **hyperparameters):
        self.model.set_params(**hyperparameters)
        Fit.fit(self.model, self.X_train, self.y_train)
        return Validate.val(self.model, self.X_test, self.y_test, self.problem)

    def __init__(self, model, data, iterations, problem):
        self.X_train = data.X_train
        self.y_train = data.y_train
        self.X_test = data.X_test
        self.y_test = data.y_test
        self.model = model
        self.iter = iterations
        self.problem = problem
        self.tuned = {}

    def get_tuned_params(self):
        return self.tuned
