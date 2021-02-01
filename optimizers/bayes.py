from preprocess.impsettings import ImputerSettings
from bayes_opt.bayesian_optimization import BayesianOptimization
from pipeline.validator import Validate
from pipeline.fit import Fit
from preprocess.preprocessor import Preprocessor
from algorithms import base_algo
from pipeline.data import Data
import pandas as pd
import numpy as np
import copy

from optimizers.base_optimizer import BaseOptimizer


class BayesianOptimizer(BaseOptimizer):
    def __init__(
        self,
        algo_list: list,
        data,
        iterations,
        problem,
        categorical_list=None,
        droplist_columns=None,
    ):
        self.data = data
        self.raw_data = data
        self.algo_list = algo_list
        self.iter = iterations
        self.problem = problem
        self.tuned_params = {}
        self.algo_index = 0
        self.preprocess_list = ["LabelEncoder", "OneHotEncoder"]
        self.numimputer_list = ["mean", "most_frequent", "median"]
        self.stringimputer_list = ["most_frequent", "constant"]
        # self.boolimputer_list = ["most_frequent"]
        self.categorical_list = categorical_list
        self.droplist_columns = droplist_columns
        print(data.X_train.columns)

        opt = BayesianOptimization(
            f=self.objective,
            pbounds={
                "algo_index_tuned": (0, len(algo_list) - 1),
                "preprocess_method": (0, len(self.preprocess_list) - 1),
                "numimputer": (0, len(self.numimputer_list) - 1),
                "stringimputer": (0, len(self.stringimputer_list) - 1),
            },
        )
        opt.maximize(n_iter=iterations, init_points=10)
        self.tuned_params = opt.max

    def objective(
        self,
        algo_index_tuned,
        preprocess_method,
        numimputer,
        stringimputer,
    ):
        self.data = copy.deepcopy(self.raw_data)
        self.algo_index = round(algo_index_tuned)

        pr = Preprocessor()
        self.data = pr.clean(
            data=self.data,
            encoder_method=self.preprocess_list[round(preprocess_method)],
            categorical_list=self.categorical_list,
            droplist_columns=self.droplist_columns,
            numimpset=ImputerSettings(strategy=self.numimputer_list[round(numimputer)]),
            stringimpset=ImputerSettings(
                basicvars="string",
                strategy=self.stringimputer_list[round(stringimputer)],
            ),
            # boolimpset=ImputerSettings(
            #     basicvars="bool", strategy=self.boolimputer_list[round(boolimputer)]
            # ),
        )

        opt = BayesianOptimization(
            f=self.child_objective,
            pbounds={**self.algo_list[self.algo_index].get_params()},
            verbose=False,
        )
        opt.maximize(n_iter=50)
        self.algo_list[self.algo_index].set_params(**opt.max["params"])
        return opt.max["target"]

    def child_objective(self, **hyperparameters):
        model = self.algo_list[self.algo_index]
        # print("Child objective")
        model.set_params(**hyperparameters)

        Fit.fit(model, self.data.X_train, self.data.y_train)

        return Validate.val(model, self.data.X_test, self.data.y_test, self.problem)

    def get_tuned_params(self):
        print(
            "Title: ",
            self.algo_list[self.algo_index].title,
            "\nInfo:",
            self.tuned_params,
            "\nModel:",
            self.algo_list[self.algo_index].model,
        )