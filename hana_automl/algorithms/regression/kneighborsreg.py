from hana_ml.algorithms.pal import metrics
from hana_ml.algorithms.pal.metrics import r2_score
from hana_ml.algorithms.pal.neighbors import KNNRegressor

from hana_automl.algorithms.base_algo import BaseAlgorithm


class KNeighborsReg(BaseAlgorithm):
    def __init__(self):
        super(KNeighborsReg, self).__init__()
        self.title = "KNNRegressor"
        self.params_range = {
            "algorithm": (0, 1),
            "n_neighbors": (1, 100),
            "aggregate_type": (0, 1),
            "metric": (0, 3),
        }

    def set_params(self, **params):
        params["aggregate_type"] = ["average", "distance-weighted"][
            round(params["aggregate_type"])
        ]
        params["metric"] = ["manhattan", "euclidean", "minkowski", "chebyshev"][
            round(params["metric"])
        ]
        params["n_neighbors"] = round(params["n_neighbors"])
        params["algorithm"] = ["brute_force", "kd-tree"][round(params["algorithm"])]
        self.model = KNNRegressor(**params)

    def optunatune(self, trial):
        aggregate_type = trial.suggest_categorical(
            "REG_KNeighbors_aggregate_type", ["average", "distance-weighted"]
        )
        n_neighbors = trial.suggest_int("REG_KNeighbors_n_neighbors", 1, 100, log=True)
        algorithm = trial.suggest_categorical(
            "REG_KNeighbors_algorithm", ["brute_force", "kd-tree"]
        )
        metric = trial.suggest_categorical(
            "REG_KNeighbors_metric",
            ["manhattan", "euclidean", "minkowski", "chebyshev"],
        )
        model = KNNRegressor(
            n_neighbors=n_neighbors,
            algorithm=algorithm,
            aggregate_type=aggregate_type,
            metric=metric,
        )
        self.model = model

    def score(self, data, df):
        return self.inner_score(df, key=data.id_colm, label=data.target)

    def inner_score(self, data, key, features=None, label=None):
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
        return metrics.r2_score(joined, label_true="ACTUAL", label_pred="PREDICTION")
