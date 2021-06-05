from hana_ml.algorithms.pal.trees import GradientBoostingClassifier

from hana_automl.algorithms.base_algo import BaseAlgorithm


class GBCls(BaseAlgorithm):
    def __init__(self):
        super(GBCls, self).__init__()
        self.title = "GradientBoostingClassifier"
        self.params_range = {
            "n_estimators": (10, 100),
            "max_depth": (6, 30),
            "loss": (0, 1),
            "min_sample_weight_leaf": (1, 100),
            "learning_rate": (0.01, 1),
        }

    def set_params(self, **params):
        params["n_estimators"] = round(params["n_estimators"])
        params["max_depth"] = round(params["max_depth"])
        params["min_sample_weight_leaf"] = round(params["min_sample_weight_leaf"])
        params["loss"] = ["linear", "logistic"][round(params["loss"])]
        self.tuned_params = params
        self.model = GradientBoostingClassifier(**params)

    def optunatune(self, trial):
        n_estimators = trial.suggest_int("n_estimators", 10, 100, log=True)
        max_depth = trial.suggest_int("max_depth", 2, 50, log=True)
        min_sample_weight_leaf = trial.suggest_int(
            "min_sample_weight_leaf", 1, 200, log=True
        )
        learning_rate = trial.suggest_float("learning_rate", 0.01, 1, log=True)
        loss = trial.suggest_categorical("loss", ["linear", "logistic"])
        model = GradientBoostingClassifier(
            n_estimators=n_estimators,
            loss=loss,
            max_depth=max_depth,
            min_sample_weight_leaf=min_sample_weight_leaf,
            learning_rate=learning_rate,
            categorical_variable=self.categorical_features,
        )
        self.model = model
