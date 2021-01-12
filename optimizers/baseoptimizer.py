from bayes_opt import BayesianOptimization
from pipeline.validator import Validate
from pipeline.fit import Fit
from algorithms import base_algo


class BaseOptimizer:

    def objective(self, **hyperparameters):
        self.model.set_params(**hyperparameters)
        Fit.fit(self.model, self.X_train, self.y_train)
        return Validate.val(self.model, self.X_test, self.y_test, self.problem)

    def __init__(self, model: base_algo, X_train, y_train, X_test, y_test, iterations, problem):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.model = model
        self.iter = iterations
        self.problem = problem
