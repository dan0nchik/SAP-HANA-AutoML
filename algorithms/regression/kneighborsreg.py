from hana_ml.algorithms.pal.neighbors import KNNRegressor

from algorithms.base_algo import BaseAlgorithm


class KNeighborsReg(BaseAlgorithm):
    def __init__(self):
        super(KNeighborsReg, self).__init__()
        self.title = "KNNRegressor"
        self.params_range = {
            "algorithm": (0, 1),
            "n_neighbors": (5, 100),
            "aggregate_type": (0, 1),
            "metric": (0, 3),
        }
        self.model = KNNRegressor()

    def set_params(self, **params):
        params["aggregate_type"] = ["average", "distance-weighted"][
            round(params["aggregate_type"])
        ]
        params["metric"] = ["manhattan", "euclidean", "minkowski", "chebyshev"][
            round(params["metric"])
        ]
        params["n_neighbors"] = round(params["n_neighbors"])
        params["algorithm"] = ["brute-force", "kd-tree"][round(params["algorithm"])]
        self.model = KNNRegressor(**params)

    def optunatune(self, trial):
        aggregate_type = trial.suggest_categorical(
            "REG_KNeighbors_aggregate_type", ["average", "distance-weighted"]
        )
        n_neighbors = trial.suggest_int("REG_KNeighbors_n_neighbors", 5, 100, log=True)
        algorithm = trial.suggest_categorical(
            "REG_KNeighbors_algorithm", ["brute-force", "kd-tree"]
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
        return model
