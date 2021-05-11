from bayes_opt import BayesianOptimization


class BaseAlgorithm:
    """Base algorithm class. Inherit from it for creating custom algorithms."""

    def __init__(self, custom_params: dict = None):
        self.title = ""  # for leaderboard
        self.model = None
        self.categorical_features = None
        self.params_range = {}
        self.bayes_opt = None
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
