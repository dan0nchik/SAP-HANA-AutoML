import copy
import uuid

import hana_ml

import hana_automl.algorithms.base_algo
from hana_automl.pipeline.data import Data
from hana_automl.preprocess.settings import PreprocessorSettings
import time

import optuna

from hana_automl.optimizers.base_optimizer import BaseOptimizer
from hana_automl.pipeline.leaderboard import Leaderboard
from hana_automl.pipeline.modelres import ModelBoard


class OptunaOptimizer(BaseOptimizer):
    """Optuna hyperparameters optimizer. (https://optuna.org/)

    Attributes
    ----------
    data : Data
        Input data.
    algo_list : list
        List of algorithms to be tuned and compared.
    algo_dict : dict
        Dictionary of algorithms to be tuned and compared.
    iter : int
        Number of iterations.
    problem : str
        Machine learning problem.
    tuned_params : str
        Final tuned hyperparameters of best algorithm.
    categorical_features : list
        List of categorical features in dataframe.
    prepset
        prepset for preprocessing.
    model
        Tuned HANA ML model in algorithm.
    droplist_columns
        Columns in dataframe to be dropped.
    """

    def __init__(
        self,
        algo_list: list,
        data: Data,
        problem: str,
        iterations: int,
        time_limit: int,
        algo_dict: dict,
        categorical_features: list = None,
        droplist_columns: list = None,
        verbosity=2,
    ):
        self.algo_list = algo_list
        self.data = data
        self.iterations = iterations
        self.problem = problem
        self.time_limit = time_limit
        self.algo_dict = algo_dict
        self.categorical_features = categorical_features
        self.droplist_columns = droplist_columns
        self.verbosity = verbosity
        if self.verbosity < 2:
            optuna.logging.set_verbosity(optuna.logging.WARNING)
        self.model = None
        self.prepset: PreprocessorSettings = PreprocessorSettings(data.strategy_by_col)
        self.leaderboard: Leaderboard = Leaderboard()
        self.accuracy = 0
        self.tuned_params = None
        self.algorithm = None
        self.study = None

    def inner_params(self, study, trial):
        if self.verbosity > 1:
            time.sleep(1)
            print(
                "\033[31m {}\033[0m".format(
                    self.leaderboard.board[
                        len(self.leaderboard.board) - 1
                    ].algorithm.title
                    + " trial params :"
                    + str(
                        self.leaderboard.board[len(self.leaderboard.board) - 1]
                        .algorithm.optuna_opt.trials[
                            len(
                                self.leaderboard.board[
                                    len(self.leaderboard.board) - 1
                                ].algorithm.optuna_opt.trials
                            )
                            - 1
                        ]
                        .params
                    )
                )
            )

    def tune(self):
        self.study = optuna.create_study(
            direction="maximize",
            study_name="hana_automl optimization process(" + str(uuid.uuid4()) + ")",
        )
        if self.iterations is not None and self.time_limit is not None:
            self.study.optimize(
                self.objective,
                n_trials=self.iterations,
                timeout=self.time_limit,
                callbacks=[self.inner_params],
            )
        elif self.iterations is None:
            self.study.optimize(
                self.objective, timeout=self.time_limit, callbacks=[self.inner_params]
            )
        else:
            self.study.optimize(
                self.objective, n_trials=self.iterations, callbacks=[self.inner_params]
            )
        time.sleep(2)
        self.tuned_params = self.study.best_params
        self.prepset.tuned_num_strategy = self.study.best_params.pop("imputer")
        if self.verbosity > 0:
            res = len(self.study.trials)
            if self.iterations is None:
                print(
                    "There was a stop due to a time limit! Completed "
                    + str(res)
                    + " iterations"
                )
            elif res == self.iterations:
                print("All iterations completed successfully!")
            else:
                print(
                    "There was a stop due to a time limit! Completed "
                    + str(res)
                    + " iterations of "
                    + str(self.iterations)
                )
            print("Starting model accuracy evaluation on the validation data!")
        for member in self.leaderboard.board:
            data = self.data.clear(
                num_strategy=member.preprocessor.tuned_num_strategy,
                strategy_by_col=member.preprocessor.strategy_by_col,
                categorical_list=self.categorical_features,
                normalizer_strategy=member.preprocessor.tuned_normalizer_strategy,
                normalizer_z_score_method=member.preprocessor.tuned_z_score_method,
                normalize_int=member.preprocessor.normalize_int,
            )
            acc = member.algorithm.score(data=data, df=data.valid)
            member.add_valid_acc(acc)
        self.leaderboard.board.sort(
            key=lambda member: member.valid_accuracy + member.train_accuracy,
            reverse=True,
        )
        self.model = self.leaderboard.board[0].algorithm.model
        self.algorithm = self.leaderboard.board[0].algorithm

    def objective(self, trial: optuna.trial.Trial) -> int:
        """Objective function. Optimizer uses it to search for best algorithm and preprocess method.

        Parameters
        ----------
        trial: optuna.trial.Trial
            Optuna trial. Details here: https://optuna.readthedocs.io/en/stable/reference/trial.html

        Returns
        -------
        acc: float
            Model's accuracy.

        """
        algo = self.algo_dict.get(
            trial.suggest_categorical("algo", self.algo_dict.keys())
        )
        algo.set_categ(self.categorical_features)
        imputer = trial.suggest_categorical("imputer", self.prepset.num_strategy)
        self.prepset.tuned_num_strategy = imputer
        normalizer_strategy = trial.suggest_categorical(
            "normalizer_strategy", self.prepset.normalizer_strategy
        )
        self.prepset.tuned_normalizer_strategy = normalizer_strategy
        z_score_method = ""
        if normalizer_strategy == "z-score":
            z_score_method = trial.suggest_categorical(
                "z_score_method", self.prepset.z_score_method
            )
            self.prepset.tuned_z_score_method = z_score_method
        normalize_int = trial.suggest_categorical(
            "normalize_int", self.prepset.normalize_int
        )
        self.prepset.tuned_normalize_int = normalize_int
        drop_outers = trial.suggest_categorical("drop_outers", self.prepset.drop_outers)
        self.prepset.tuned_drop_outers = drop_outers
        data = self.data.clear(
            strategy_by_col=self.prepset.strategy_by_col,
            num_strategy=imputer,
            categorical_list=self.categorical_features,
            normalizer_strategy=normalizer_strategy,
            normalizer_z_score_method=z_score_method,
            normalize_int=normalize_int,
            drop_outers=drop_outers,
        )
        acc = algo.optuna_tune(data)
        self.leaderboard.addmodel(
            ModelBoard(copy.copy(algo), acc, copy.copy(self.prepset))
        )
        return acc

    def get_tuned_params(self) -> dict:
        """Returns tuned hyperparameters."""
        return {
            "algorithm": self.tuned_params,
            "accuracy": self.leaderboard.board[0].valid_accuracy,
        }

    def get_model(self) -> hana_ml.algorithms.pal.pal_base:
        """Returns tuned model."""
        return self.model

    def get_algorithm(self) -> hana_automl.algorithms.base_algo.BaseAlgorithm:
        """Returns tuned AutoML algorithm"""
        return self.algorithm

    def get_preprocessor_settings(self) -> PreprocessorSettings:
        """Returns tuned preprocessor settings."""
        return self.leaderboard.board[0].preprocessor

    def fit(self, algo, data):
        """Fits given model from data. Small method to reduce code repeating."""
        ftr: list = data.train.columns
        ftr.remove(data.target)
        ftr.remove(data.id_colm)
        algo.fit(data, ftr, self.categorical_features)
