from hana_ml.algorithms.pal.trees import HybridGradientBoostingRegressor

from hana_automl.algorithms.base_algo import BaseAlgorithm


class HGBReg(BaseAlgorithm):
    def __init__(self):
        super(HGBReg, self).__init__()
        self.title = "HybridGradientBoostingRegressor"
        self.params_range = {
            "n_estimators": (10, 100),
            "split_method": (0, 2),
            "max_depth": (6, 30),
            "min_sample_weight_leaf": (1, 100),
            "learning_rate": (0.01, 1),
        }

    def set_params(self, **params):
        params["n_estimators"] = round(params["n_estimators"])
        params["max_depth"] = round(params["max_depth"])
        params["min_sample_weight_leaf"] = round(params["min_sample_weight_leaf"])
        params["split_method"] = ["exact", "sketch", "sampling"][
            round(params["split_method"])
        ]
        self.tuned_params = params
        self.model = HybridGradientBoostingRegressor(**params)

    def optunatune(self, trial):
        n_estimators = trial.suggest_int("n_estimators", 100, 1000, log=True)
        max_depth = trial.suggest_int("max_depth", 2, 50, log=True)
        min_sample_weight_leaf = trial.suggest_int(
            "min_sample_weight_leaf", 1, 20, log=True
        )
        learning_rate = trial.suggest_float("learning_rate", 0.01, 1, log=True)
        split_method = trial.suggest_categorical(
            "split_method", ["exact", "sketch", "sampling"]
        )
        model = HybridGradientBoostingRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            split_method=split_method,
            min_sample_weight_leaf=min_sample_weight_leaf,
            learning_rate=learning_rate,
        )
        self.model = model
