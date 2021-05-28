from time import sleep

import optuna
from bayes_opt import BayesianOptimization


class BaseAlgorithm:
    """Base algorithm class. Inherit from it for creating custom algorithms."""

    def __init__(self, custom_params: dict = None):
        self.title = ""  # for leaderboard
        self.model = None
        self.categorical_features = None
        self.params_range = {}
        self.bayes_opt = None
        self.optuna_opt = None
        self.temp_data = None
        if custom_params is not None:
            # self.params_range[custom_params.keys()] = custom_params.values()
            pass

    def get_params(self):
        return self.params_range

    def set_params(self, **params):
        self.model.set_params(**params)

    def optunatune(self, trial):
        pass

    def score(self, data, df):
        return self.model.score(df, key=data.id_colm, label=data.target)

    def set_categ(self, cat):
        self.categorical_features = cat

    def bayes_tune(
        self,
        f,
    ):
        if self.bayes_opt is None:
            self.bayes_opt = BayesianOptimization(
                f=f,
                pbounds=self.params_range,
                verbose=False,
                random_state=17,
            )
        self.bayes_opt.maximize(n_iter=1, init_points=1)
        return self.bayes_opt.max["target"], self.bayes_opt.max["params"]

    def optuna_tune(self, data):
        if self.optuna_opt is None:
            v = optuna.logging.get_verbosity()
            optuna.logging.set_verbosity(optuna.logging.WARNING)
            self.optuna_opt = optuna.create_study(
                direction="maximize",
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
        acc = self.score(data=self.temp_data, df=self.temp_data.test)
        return acc

    def fit(self, data, features, categorical_features):
        self.model.fit(
            data=data.train,
            key=data.id_colm,
            features=features,
            categorical_variable=categorical_features,
            label=data.target,
        )

    def __repr__(self):
        return self.title
