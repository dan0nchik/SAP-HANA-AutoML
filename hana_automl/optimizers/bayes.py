import time

from bayes_opt.bayesian_optimization import BayesianOptimization

from hana_automl.optimizers.base_optimizer import BaseOptimizer
from hana_automl.pipeline.leaderboard import Leaderboard

from hana_automl.pipeline.modelres import ModelBoard
from hana_automl.preprocess.settings import PreprocessorSettings
from hana_automl.utils.error import OptimizerError

import numpy as np

np.seterr(divide="ignore", invalid="ignore")


class BayesianOptimizer(BaseOptimizer):
    """Bayesian hyperparameters optimizer. (https://github.com/fmfn/BayesianOptimization)

    Attributes
    ----------
    data : Data
        Input data.
    algo_list : list
        List of algorithms to be tuned and compared.
    iter : int
        Number of iterations.
    problem : str
        Machine learning problem.
    tuned_params : str
        Final tuned hyperparameters of best algorithm.
    algo_index : int
        Index of algorithm in algorithms list.
    categorical_features : list
        List of categorical features in dataframe.
    inner_data : Data
        Copy of Data object to prevent from preprocessing data object multiple times while optimizing.
    imputer
        Imputer for preprocessing
    model
        Tuned HANA ML model in algorithm.
    """

    def __init__(
        self,
        algo_list: list,
        data,
        iterations,
        time_limit,
        problem,
        categorical_features=None,
        verbosity=2,
    ):
        self.data = data
        self.algo_list = algo_list
        self.iter = None
        if iterations is not None:
            self.iter = iterations - 1
        self.problem = problem
        self.tuned_params = {}
        self.algo_index = 0
        self.time_limit = time_limit
        self.start_time = None
        self.categorical_features = categorical_features
        self.inner_data = None
        self.imputer: PreprocessorSettings = PreprocessorSettings()
        self.model = None
        self.leaderboard: Leaderboard = Leaderboard()
        self.algorithm = None
        self.verbosity = verbosity

    def objective(self, algo_index_tuned, num_strategy_method):
        """Main objective function. Optimizer uses it to search for best algorithm and preprocess method.

        Parameters
        ----------
        algo_index_tuned : int
            Index of algorithm in algo_list.
        num_strategy_method : int
            Strategy to decode categorical variables.

        Returns
        -------
        Best params
            Optimal hyperparameters.

        Note
        ----
            Total number of child objective iterations is n_iter + init_points!
        """
        if self.time_limit is not None:
            if time.perf_counter() - self.start_time > self.time_limit:
                raise OptimizerError()
        self.algo_index = round(algo_index_tuned)
        imputer = self.imputer.num_strategy[round(num_strategy_method)]
        self.imputer.tuned_num_strategy = imputer
        self.inner_data = self.data.clear(
            num_strategy=imputer,
            cat_strategy=None,
            dropempty=False,
            categorical_list=None,
        )
        target, params = self.algo_list[self.algo_index].bayes_tune(
            f=self.child_objective
        )
        print(target)
        algo = self.algo_list[self.algo_index]
        algo.set_params(**params)
        self.fit(algo, self.inner_data)

        self.leaderboard.addmodel(ModelBoard(algo, target, self.imputer))

        return target

    def child_objective(self, **hyperparameters):
        """Mini objective function. It is used to tune hyperparameters of algorithm that was chosen in main objective.

        Parameters
        ----------
        **hyperparameters
            Parameters of algorithm's model.
        Returns
        -------
        acc
            Accuracy score of a model.
        """
        algorithm = self.algo_list[self.algo_index]
        algorithm.set_params(**hyperparameters)
        self.fit(algorithm, self.inner_data)
        acc = algorithm.score(self.inner_data, self.inner_data.test)
        if self.verbosity > 1:
            print("Child Iteration accuracy: " + str(acc))
        return acc

    def get_tuned_params(self):
        """Returns tuned hyperparameters."""

        return {
            "title": self.algo_list[self.algo_index].title,
            "accuracy": self.leaderboard.board[0].valid_accuracy,
            "info": self.tuned_params,
        }

    def get_model(self):
        """Returns tuned model."""

        return self.model

    def get_algorithm(self):
        """Returns tuned AutoML algorithm"""

        return self.algorithm

    def get_preprocessor_settings(self):
        """Returns tuned preprocessor settings."""

        return self.imputer

    def tune(self):
        """Starts hyperparameter searching."""

        opt = BayesianOptimization(
            f=self.objective,
            pbounds={
                "algo_index_tuned": (0, len(self.algo_list) - 1),
                "num_strategy_method": (0, len(self.imputer.num_strategy) - 1),
            },
            random_state=17,
            verbose=self.verbosity > 1,
        )
        self.start_time = time.perf_counter()
        try:
            if self.iter is None:
                opt.maximize(n_iter=99999999999999, init_points=1)
            else:
                opt.maximize(n_iter=self.iter, init_points=1)
        except OptimizerError:
            if self.verbosity > 0:
                if self.iter is None:
                    print(
                        "There was a stop due to a time limit! Completed "
                        + str(len(opt.res))
                        + " iterations"
                    )
                else:
                    print(
                        "There was a stop due to a time limit! Completed "
                        + str(len(opt.res))
                        + " iterations of "
                        + str(self.iter)
                    )
        else:
            if self.verbosity > 0:
                print("All iterations completed successfully!")
        self.tuned_params = opt.max
        if self.verbosity > 0:
            print("Starting model accuracy evaluation on the validation data!")
        for member in self.leaderboard.board:
            data = self.data.clear(
                num_strategy=member.preprocessor.tuned_num_strategy,
                cat_strategy=None,
                dropempty=False,
                categorical_list=None,
            )
            acc = member.algorithm.score(data=data, df=data.valid)
            member.add_valid_acc(acc)

        self.leaderboard.board.sort(
            key=lambda member: member.valid_accuracy + member.train_accuracy,
            reverse=True,
        )
        self.model = self.leaderboard.board[0].algorithm.model
        self.algorithm = self.leaderboard.board[0].algorithm

    def fit(self, algo, data):
        """Fits given model from data. Small method to reduce code repeating."""
        ftr: list = data.train.columns
        ftr.remove(data.target)
        ftr.remove(data.id_colm)
        algo.fit(data, ftr, self.categorical_features)
