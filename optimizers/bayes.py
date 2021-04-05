from bayes_opt.bayesian_optimization import BayesianOptimization
import copy
from optimizers.base_optimizer import BaseOptimizer


class BayesianOptimizer(BaseOptimizer):
    def objective(self, algo_index_tuned, preprocess_method):
        self.algo_index = round(algo_index_tuned)
        rounded_preprocess_method = round(preprocess_method)
        self.imputer = self.imputerstrategy_list[rounded_preprocess_method]
        self.inner_data = copy.copy(self.data).clear(
            num_strategy=self.imputer,
            cat_strategy=None,
            dropempty=False,
            categorical_list=None,
        )
        opt = BayesianOptimization(
            f=self.child_objective,
            pbounds={**self.algo_list[self.algo_index].get_params()},
            verbose=False,
            random_state=17,
        )
        opt.maximize(n_iter=1)
        self.algo_list[self.algo_index].set_params(**opt.max["params"])
        return opt.max["target"]

    def child_objective(self, **hyperparameters):
        algorithm = self.algo_list[self.algo_index]
        algorithm.set_params(**hyperparameters)
        ftr: list = self.inner_data.train.columns
        ftr.remove(self.inner_data.target)
        ftr.remove(self.inner_data.id_colm)
        algorithm.model.fit(
            self.inner_data.train,
            key=self.inner_data.id_colm,
            features=ftr,
            label=self.inner_data.target,
            categorical_variable=self.categorical_list,
        )
        acc = algorithm.model.score(
            self.inner_data.valid,
            key=self.inner_data.id_colm,
            label=self.inner_data.target,
        )
        print("Iteration accuracy: " + str(acc))
        self.model = algorithm.model
        return acc

    def get_tuned_params(self):
        return {
            "title": self.algo_list[self.algo_index].title,
            "params": self.tuned_params,
        }

    def get_model(self):
        return self.model

    def get_preprocessor_settings(self):
        return {"imputer": self.imputer}

    def __init__(
        self, algo_list: list, data, iterations, problem, categorical_list=None
    ):
        self.data = data
        self.algo_list = algo_list
        self.iter = iterations
        self.problem = problem
        self.tuned_params = {}
        self.algo_index = 0
        self.imputerstrategy_list = ["mean", "median", "zero"]
        self.categorical_list = categorical_list
        self.inner_data = None
        self.imputer = None
        self.model = None

    def tune(self):
        opt = BayesianOptimization(
            f=self.objective,
            pbounds={
                "algo_index_tuned": (0, len(self.algo_list) - 1),
                "preprocess_method": (0, len(self.imputerstrategy_list) - 1),
            },
            random_state=17,
        )
        opt.maximize(n_iter=self.iter)
        self.tuned_params = opt.max
