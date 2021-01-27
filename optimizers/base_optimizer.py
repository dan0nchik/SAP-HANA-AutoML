from bayes_opt.bayesian_optimization import BayesianOptimization
from pipeline.validator import Validate
from pipeline.fit import Fit
from algorithms import base_algo


class BaseOptimizer:

    # TODO add preprocess method
    def objective(self, algo_index_tuned):
        self.algo_index = round(algo_index_tuned)
        opt = BayesianOptimization(
            f=self.child_objective,
            pbounds={**self.algo_list[self.algo_index].get_params()},
            verbose=False
        )
        opt.maximize(
            n_iter=10
        )
        self.algo_list[self.algo_index].set_params(**opt.max['params'])
        return opt.max['target']

    def child_objective(self, **hyperparameters):
        model = self.algo_list[self.algo_index]
        print("Child objective")
        model.set_params(**hyperparameters)

        Fit.fit(model, self.X_train, self.y_train)

        return Validate.val(model, self.X_test, self.y_test, self.problem)

    def __init__(self, algo_list: list, data, iterations, problem):
        self.X_train = data.X_train
        self.y_train = data.y_train
        self.X_test = data.X_test
        self.y_test = data.y_test
        self.algo_list = algo_list
        self.iter = iterations
        self.problem = problem
        self.tuned_params = {}
        self.algo_index = 0

    def get_tuned_params(self):
        print('Title: ', self.algo_list[self.algo_index].title, '\nInfo:',
              self.tuned_params
              , '\nModel:', self.algo_list[self.algo_index].model)
