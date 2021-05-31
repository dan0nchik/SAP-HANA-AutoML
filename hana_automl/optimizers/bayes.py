import time

import hana_ml
from bayes_opt.bayesian_optimization import BayesianOptimization

import hana_automl.algorithms.base_algo
from hana_automl.optimizers.base_optimizer import BaseOptimizer
from hana_automl.pipeline.data import Data
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
        Machine learning problem. Currently supported: 'cls', 'reg'
    tuned_params : str
        Final tuned hyperparameters of best algorithm.
    algo_index : int
        Index of algorithm in algorithms list.
    time_limit : int
        Time in seconds.
    categorical_features : list
        List of categorical features in dataframe.
    inner_data : Data
        Copy of Data object to prevent from preprocessing data object multiple times while optimizing.
    prepset
        Imputer for preprocessing
    model
        Tuned HANA ML model in algorithm.
    """

    def __init__(
        self,
        algo_list: list,
        data: Data,
        iterations: int,
        time_limit: int,
        problem: str,
        categorical_features: list = None,
        verbosity=2,
        tuning_metric: str = None,
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
        self.prepset: PreprocessorSettings = PreprocessorSettings(data.strategy_by_col)
        self.model = None
        self.leaderboard: Leaderboard = Leaderboard()
        self.algorithm = None
        self.verbosity = verbosity
        self.tuning_metric = tuning_metric

    def objective(
        self,
        algo_index_tuned: int,
        num_strategy_method: int,
        normalizer_strategy: int,
        z_score_method: int,
        normalize_int: int,
        drop_outers: int,
    ):
        """Main objective function. Optimizer uses it to search for best algorithm and preprocess method.

        Parameters
        ----------
        algo_index_tuned : int
            Index of algorithm in algo_list.
        num_strategy_method : int
            Strategy to decode categorical variables.
        normalizer_strategy: int
            Strategy for normalization
        z_score_method : int
            A z-score (also called a standard score) gives you an idea of how far from the mean a data point is
        normalize_int : int
            How to normalize integers

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
        imputer = self.prepset.num_strategy[round(num_strategy_method)]
        self.prepset.tuned_num_strategy = imputer
        normalizer_strategy_2 = self.prepset.normalizer_strategy[
            round(normalizer_strategy)
        ]
        self.prepset.tuned_normalizer_strategy = normalizer_strategy_2
        z_score_method_2 = self.prepset.z_score_method[round(z_score_method)]
        self.prepset.tuned_z_score_method = z_score_method_2
        normalize_int_2 = self.prepset.normalize_int[round(normalize_int)]
        self.prepset.tuned_normalize_int = normalize_int_2
        drop_outers = self.prepset.normalize_int[round(drop_outers)]
        self.prepset.tuned_drop_outers = drop_outers
        self.inner_data = self.data.clear(
            num_strategy=imputer,
            strategy_by_col=self.prepset.strategy_by_col,
            categorical_list=self.categorical_features,
            normalizer_strategy=normalizer_strategy_2,
            normalizer_z_score_method=z_score_method_2,
            normalize_int=normalize_int_2,
            drop_outers=drop_outers,
            clean_sets=["test", "train"],
        )
        target, params = self.algo_list[self.algo_index].bayes_tune(
            f=self.child_objective
        )
        print("Best child Iteration cycle score: " + str(target))
        algo = self.algo_list[self.algo_index]
        algo.set_params(**params)
        self.fit(algo, self.inner_data)

        self.leaderboard.addmodel(ModelBoard(algo, target, self.prepset))

        return target

    def child_objective(self, **hyperparameters) -> int:
        """Mini objective function. It is used to tune hyperparameters of algorithm that was chosen in main objective.

        Parameters
        ----------
        **hyperparameters
            Parameters of algorithm's model.
        Returns
        -------
        acc: float
            Accuracy score of a model.
        """
        algorithm = self.algo_list[self.algo_index]
        algorithm.set_params(**hyperparameters)
        self.fit(algorithm, self.inner_data)
        acc = algorithm.score(self.inner_data, self.inner_data.test, self.tuning_metric)
        if self.verbosity > 1:
            print("Child Iteration accuracy: " + str(acc))
        return acc

    def get_tuned_params(self) -> dict:
        """Returns tuned hyperparameters."""

        return {
            "title": self.algo_list[self.algo_index].title,
            "accuracy": self.leaderboard.board[0].valid_accuracy,
            "info": self.tuned_params,
        }

    def get_model(self) -> hana_ml.algorithms.pal.pal_base:
        """Returns tuned model."""

        return self.model

    def get_algorithm(self) -> hana_automl.algorithms.base_algo.BaseAlgorithm:
        """Returns tuned AutoML algorithm"""

        return self.algorithm

    def get_preprocessor_settings(self) -> PreprocessorSettings:
        """Returns tuned preprocessor settings."""

        return self.prepset

    def tune(self):
        """Starts hyperparameter searching."""

        opt = BayesianOptimization(
            f=self.objective,
            pbounds={
                "algo_index_tuned": (0, len(self.algo_list) - 1),
                "num_strategy_method": (0, len(self.prepset.num_strategy) - 1),
                "normalizer_strategy": (0, len(self.prepset.normalizer_strategy) - 1),
                "z_score_method": (0, len(self.prepset.z_score_method) - 1),
                "normalize_int": (0, len(self.prepset.normalize_int) - 1),
                "drop_outers": (0, len(self.prepset.drop_outers) - 1),
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
                strategy_by_col=member.preprocessor.strategy_by_col,
                categorical_list=self.categorical_features,
                normalizer_strategy=member.preprocessor.tuned_normalizer_strategy,
                normalizer_z_score_method=member.preprocessor.tuned_z_score_method,
                normalize_int=member.preprocessor.tuned_normalize_int,
                clean_sets=["valid"],
            )
            acc = member.algorithm.score(data=data, df=data.valid, metric=self.tuning_metric)
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
