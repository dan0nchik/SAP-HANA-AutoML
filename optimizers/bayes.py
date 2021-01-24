from bayes_opt import BayesianOptimization

from optimizers.base_optimizer import BaseOptimizer


class BayesianOptimizer(BaseOptimizer):
    def __init__(self, algo_list, data, problem, iterations=10):
        super(BayesianOptimizer, self).__init__(algo_list, data, iterations, problem)
        opt = BayesianOptimization(
            f=self.objective,
            pbounds={'algo_index_tuned': (0, len(algo_list)-1), **self.algo_list[self.algo_index].get_params()},
            random_state=17
        )
        opt.maximize(
            n_iter=iterations
        )
        self.tuned_params = opt.max
