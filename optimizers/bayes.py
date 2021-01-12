from bayes_opt import BayesianOptimization
from optimizers.base_optimizer import BaseOptimizer


class BayesianOptimizer(BaseOptimizer):
    def __init__(self, model, data, iterations, problem):
        super(BayesianOptimizer, self).__init__(model, data, iterations, problem)
        opt = BayesianOptimization(
            f=self.objective,
            pbounds=self.model.get_params()
        )
        opt.maximize()
        self.tuned = opt.max
