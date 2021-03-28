from bayes_opt.bayesian_optimization import BayesianOptimization

from optimizers.base_optimizer import BaseOptimizer


class BayesianOptimizer(BaseOptimizer):
    def objective(self, algo_index_tuned, preprocess_method):
        self.algo_index = round(algo_index_tuned)
        rounded_preprocess_method = round(preprocess_method)
        self.inner_data = self.data.clear(num_strategy=self.imputerstrategy_list[rounded_preprocess_method], cat_strategy=None,
                                    dropempty=False,
                                    categorical_list=None
                                    )
        opt = BayesianOptimization(
            f=self.child_objective,
            pbounds={**self.algo_list[self.algo_index].get_params()},
            verbose=False,
        )
        opt.maximize(n_iter=10)
        self.algo_list[self.algo_index].set_params(**opt.max["params"])
        return opt.max["target"]

    def child_objective(self, **hyperparameters):
        model = self.algo_list[self.algo_index]
        model.set_params(**hyperparameters)
        ftr: list = self.inner_data.train.columns
        ftr.remove(self.inner_data.target)
        ftr.remove(self.inner_data.id_colm)
        model.model.fit(self.inner_data.train, key=self.inner_data.id_colm, features=ftr, label=self.inner_data.target,
                  categorical_variable=self.categorical_list)
        acc = model.model.score(self.inner_data.valid, key=self.inner_data.id_colm, label=self.inner_data.target)
        print('Itteration accuracy: '+str(acc))
        return acc

    def get_tuned_params(self):
        return {
            "title": self.algo_list[self.algo_index].title,
            "params": self.tuned_params
        }

    def __init__(
            self,
            algo_list: list,
            data,
            iterations,
            problem,
            categorical_list=None
    ):
        self.data = data
        self.inner_data = data
        self.algo_list = algo_list
        self.iter = iterations
        self.problem = problem
        self.tuned_params = {}
        self.algo_index = 0
        self.imputerstrategy_list = ['mean', 'median', 'zero']
        self.categorical_list = categorical_list

    def tune(self):
        opt = BayesianOptimization(
            f=self.objective,
            pbounds={
                "algo_index_tuned": (0, len(self.algo_list) - 1),
                "preprocess_method": (0, len(self.imputerstrategy_list) - 1),
            },
        )
        opt.maximize(n_iter=self.iter)
        self.tuned_params = opt.max
