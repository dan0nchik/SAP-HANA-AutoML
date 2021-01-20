from bayes_opt import BayesianOptimization

from optimizers.base_optimizer import BaseOptimizer


class BayesianOptimizer(BaseOptimizer):
    def __init__(self, algorithm, data, iterations, problem):
        super(BayesianOptimizer, self).__init__(algorithm, data, iterations, problem)
        opt = BayesianOptimization(
            f=self.objective,
            pbounds=self.algorithm.get_params()
        )
        opt.maximize(
            n_iter=iterations
        )
        self.tuned_params = opt.max
