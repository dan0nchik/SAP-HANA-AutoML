from algorithms.base_algo import BaseAlgorithm
from sklearn.ensemble import RandomForestClassifier


class RandomForest(BaseAlgorithm):
    def __init__(self):
        super(RandomForest, self).__init__()
        self.title = "RandomForest"
        self.params_range = {
            "n_estimators": (5, 200),
            "max_features": (0, 2),
            "min_samples_split": (2, 10)
        }
        self.model = RandomForestClassifier()

    def set_params(self, **params):
        params["max_features"] = ["sqrt", "log2"][round(params["max_features"])]
        params["n_estimators"] = round(params["n_estimators"])
        params["min_samples_split"] = round(params["min_samples_split"])
        self.model.set_params(**params)

    def optunatune(self, trial):
        n_estimators = trial.suggest_int("KNeighbors_n_estimators", 5, 200, log=True)
        min_samples_split = trial.suggest_int("KNeighbors_min_samples_split", 2, 10, log=True)
        max_features = trial.suggest_categorical("KNeighbors_max_features", ["sqrt", "log2"])
        model = RandomForestClassifier(
            n_estimators=n_estimators, min_samples_split=min_samples_split, max_features=max_features
        )
        return model
