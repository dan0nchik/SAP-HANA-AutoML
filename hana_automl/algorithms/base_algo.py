from time import sleep

import hana_ml
import optuna
from bayes_opt import BayesianOptimization
from hana_ml.algorithms.pal.regression import ExponentialRegression

from hana_automl.metric.mae import mae_score
from hana_automl.metric.mse import mse_score
from hana_automl.metric.rmse import rmse_score
from hana_automl.optimizers.bayes import BayesianOptimizer
from hana_automl.optimizers.optuna_optimizer import OptunaOptimizer


class BaseAlgorithm:
    """Base algorithm class. Inherit from it for creating custom algorithms."""

    def __init__(self, custom_params: dict = None, model=None):
        self.title: str = ""  # for leaderboard
        self.model = model
        self.categorical_features: list = None
        self.params_range: dict = {}
        self.bayes_opt: BayesianOptimizer = None
        self.optuna_opt: OptunaOptimizer = None
        self.temp_data = None
        self.tuning_metric: str = None
        self.tuned_params: dict = None
        if custom_params is not None:
            # self.params_range[custom_params.keys()] = custom_params.values()
            pass

    def get_params(self):
        return self.params_range

    def set_params(self, **params):
        self.model.set_params(**params)

    def optunatune(self, trial):
        pass

    def score(self, data, df: hana_ml.DataFrame, metric: str):
        if metric == "accuracy" or metric == "r2_score" or metric is None:
            return self.model.score(df, key=data.id_colm, label=data.target)
        elif metric in ["mae", "mse", "rmse"]:
            c = df.columns
            c.remove(data.id_colm)
            c.remove(data.target)
            if metric == "mae":
                return mae_score(self.model, df, data.target, c, data.id_colm)
            if metric == "mse":
                return mse_score(self.model, df, data.target, c, data.id_colm)
            if metric == "rmse":
                return rmse_score(self.model, df, data.target, c, data.id_colm)

    def set_categ(self, cat):
        self.categorical_features = cat

    def bayes_tune(
        self,
        f,
    ):
        if self.bayes_opt is None:
            self.bayes_opt = BayesianOptimization(
                f=f, pbounds=self.params_range, verbose=False, random_state=17
            )
        self.bayes_opt.maximize(n_iter=1, init_points=1)
        return self.bayes_opt.max["target"], self.bayes_opt.max["params"]

    def optuna_tune(self, data, tuning_metric):
        self.tuning_metric = tuning_metric
        if self.optuna_opt is None:
            v = optuna.logging.get_verbosity()
            optuna.logging.set_verbosity(optuna.logging.WARNING)
            if self.tuning_metric in ["mse", "rmse"]:
                dirc = "minimize"
            else:
                dirc = "maximize"
            self.optuna_opt = optuna.create_study(
                direction=dirc,
                study_name="hana_automl optimization process(" + self.title + ")",
            )
            optuna.logging.set_verbosity(v)
        self.temp_data = data
        v = optuna.logging.get_verbosity()
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        self.optuna_opt.optimize(self.inner_opt_tune, n_trials=1)
        optuna.logging.set_verbosity(v)
        return self.optuna_opt.trials[len(self.optuna_opt.trials) - 1].value.real

    def inner_opt_tune(self, trial):
        self.optunatune(trial)
        ftr: list = self.temp_data.train.columns
        ftr.remove(self.temp_data.target)
        ftr.remove(self.temp_data.id_colm)
        self.fit(self.temp_data, ftr, self.categorical_features)
        acc = self.score(
            data=self.temp_data, df=self.temp_data.test, metric=self.tuning_metric
        )
        return acc

    def fit(self, data, features, categorical_features):
        if isinstance(
            self.model, ExponentialRegression
        ):  # does not support categorical
            self.model.fit(
                data=data.train,
                key=data.id_colm,
                features=features,
                label=data.target,
            )
        else:
            self.model.fit(
                data=data.train,
                key=data.id_colm,
                features=features,
                categorical_variable=categorical_features,
                label=data.target,
            )

    def __repr__(self):
        return self.title
