from bayes_opt import BayesianOptimization
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from skopt import BayesSearchCV
from skopt.space import Categorical, Integer

from optimizers.base_optimizer import BaseOptimizer


class BayesianOptimizer(BaseOptimizer):
    def __init__(
        self,
        algo_list,
        data,
        problem,
        categorical_list,
        droplist_columns,
        iterations=10,
    ):
        super(BayesianOptimizer, self).__init__(
            algo_list=algo_list,
            data=data,
            problem=problem,
            categorical_list=categorical_list,
            droplist_columns=droplist_columns,
            iterations=iterations,
        )
        print(data.X_train.columns)
        opt = BayesianOptimization(
            f=self.objective,
            pbounds={
                "algo_index_tuned": (0, len(algo_list) - 1),
                "preprocess_method": (0, len(self.preprocess_list) - 1),
            },
        )
        opt.maximize(n_iter=iterations)
        self.tuned_params = opt.max


class ScikitBayesianOptimizer(BaseOptimizer):
    def __init__(self, algo_list, data, problem, iterations=10):
        super(ScikitBayesianOptimizer, self).__init__(
            algo_list, data, iterations, problem
        )
        algl = []
        for alg in algo_list:
            algl.append((alg.params_range, iterations))
        print(algl)
        pipe = Pipeline([("model", algo_list[0].model)])
        opt = BayesSearchCV(pipe, algl, n_jobs=-1, verbose=3)
        opt.fit(self.X_train, self.y_train)
        self.tuned_params = opt.best_params_
