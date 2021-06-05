from hana_ml.algorithms.pal import metrics
from hana_ml.algorithms.pal.metrics import accuracy_score
from hana_ml.algorithms.pal.neighbors import KNNClassifier
from hana_ml.ml_base import ListOfStrings

from hana_automl.algorithms.base_algo import BaseAlgorithm


class KNeighborsCls(BaseAlgorithm):
    def __init__(self):
        super(KNeighborsCls, self).__init__()
        self.title = "KNeighborsClassifier"
        self.params_range = {
            "algorithm": (0, 1),
            "n_neighbors": (1, 21),
            "voting_type": (0, 1),
            "metric": (0, 4),
        }

    def set_params(self, **params):
        params["voting_type"] = ["majority", "distance-weighted"][
            round(params["voting_type"])
        ]
        params["metric"] = [
            "manhattan",
            "euclidean",
            "minkowski",
            "chebyshev",
            "cosine",
        ][round(params["metric"])]
        params["n_neighbors"] = round(params["n_neighbors"])
        params["algorithm"] = ["brure-force", "kd-tree"][round(params["algorithm"])]
        self.tuned_params = params
        self.model = KNNClassifier(**params)

    def optunatune(self, trial):
        n_neighbors = trial.suggest_int("n_neighbors", 1, 100, log=True)
        algorithm = trial.suggest_categorical("algorithm", ["brure-force", "kd-tree"])
        voting_type = trial.suggest_categorical(
            "voting_type", ["majority", "distance-weighted"]
        )
        metric = trial.suggest_categorical(
            "metric",
            ["manhattan", "euclidean", "minkowski", "cosine"],
        )
        model = KNNClassifier(
            n_neighbors=n_neighbors,
            algorithm=algorithm,
            voting_type=voting_type,
            metric=metric,
        )
        self.model = model

    def score(self, data, df, metric):
        return self.inner_score(df, key=data.id_colm, label=data.target, metric=metric)

    def inner_score(self, data, key, metric, features=None, label=None):
        cols = data.columns
        cols.remove(key)
        if label is None:
            label = cols[-1]
        cols.remove(label)
        if features is None:
            features = cols
        prediction, a = self.model.predict(data=data, key=key, features=features)
        prediction = prediction.select(key, "TARGET").rename_columns(
            ["ID_P", "PREDICTION"]
        )
        actual = data.select(key, label).rename_columns(["ID_A", "ACTUAL"])
        joined = actual.join(prediction, "ID_P=ID_A").select("ACTUAL", "PREDICTION")
        if metric == "accuracy":
            return metrics.accuracy_score(
                joined, label_true="ACTUAL", label_pred="PREDICTION"
            )
