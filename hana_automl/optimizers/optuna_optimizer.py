from hana_automl.preprocess.settings import PreprocessorSettings
import time

import optuna

# TODO: turn off optuna logging if verbose set to False
optuna.logging.set_verbosity(optuna.logging.WARNING)
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
    imputer
        Imputer for preprocessing.
    model
        Tuned HANA ML model in algorithm.
    droplist_columns
        Columns in dataframe to be dropped.
    """

    def __init__(
        self,
        algo_list,
        data,
        problem,
        iterations,
        time_limit,
        algo_dict,
        categorical_features=None,
        droplist_columns=None,
    ):
        self.algo_list = algo_list
        self.data = data
        self.iterations = iterations
        self.problem = problem
        self.time_limit = time_limit
        self.algo_dict = algo_dict
        self.categorical_features = categorical_features
        self.droplist_columns = droplist_columns
        self.model = None
        self.imputer: PreprocessorSettings = PreprocessorSettings()
        self.leaderboard: Leaderboard = Leaderboard()
        self.accuracy = 0
        self.tuned_params = None
        self.algorithm = None

    def tune(self):
        opt = optuna.create_study(direction="maximize")
        if self.iterations is not None and self.time_limit is not None:
            opt.optimize(
                self.objective, n_trials=self.iterations, timeout=self.time_limit
            )
        elif self.iterations is None:
            opt.optimize(self.objective, timeout=self.time_limit)
        else:
            opt.optimize(self.objective, n_trials=self.iterations)
        time.sleep(2)
        self.tuned_params = opt.best_params
        self.imputer.tuned_num_strategy = opt.best_params.pop("imputer")
        res = len(opt.trials)
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
                cat_strategy=None,
                dropempty=False,
                categorical_list=self.categorical_features,
            )
            acc = member.algorithm.score(data=data, df=data.valid)
            member.add_valid_acc(acc)
        self.leaderboard.board.sort(
            key=lambda member: member.valid_accuracy + member.train_accuracy,
            reverse=True,
        )
        self.model = self.leaderboard.board[0].algorithm.model
        self.algorithm = self.leaderboard.board[0].algorithm

    def objective(self, trial):
        """Objective function. Optimizer uses it to search for best algorithm and preprocess method.

        Parameters
        ----------
        trial
            Optuna trial. Details here: https://optuna.readthedocs.io/en/stable/reference/trial.html

        Returns
        -------
        Score
            Model perfomance.

        """
        algo = self.algo_dict.get(
            trial.suggest_categorical("algo", self.algo_dict.keys())
        )
        algo.set_categ(self.categorical_features)
        imputer = trial.suggest_categorical("imputer", self.imputer.num_strategy)
        self.imputer.tuned_num_strategy = imputer
        data = self.data.clear(
            num_strategy=imputer,
            cat_strategy=None,
            dropempty=False,
            categorical_list=None,
        )
        algo.optunatune(trial)
        self.fit(algo, data)
        acc = algo.score(data=data, df=data.test)
        self.leaderboard.addmodel(ModelBoard(algo, acc, self.imputer))
        return acc

    def get_tuned_params(self):
        """Returns tuned hyperparameters."""
        return {
            "title": self.tuned_params.pop("algo"),
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

    def fit(self, algo, data):
        """Fits given model from data. Small method to reduce code repeating."""
        ftr: list = data.train.columns
        ftr.remove(data.target)
        ftr.remove(data.id_colm)
        algo.fit(data, ftr, self.categorical_features)
