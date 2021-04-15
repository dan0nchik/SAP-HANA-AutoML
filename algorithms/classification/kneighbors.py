from hana_ml.algorithms.pal.neighbors import KNNClassifier

from algorithms.base_algo import BaseAlgorithm


class KNeighbors(BaseAlgorithm):
    def __init__(self):
        super(KNeighbors, self).__init__()
        self.title = "KNeighborsClassifier"
        self.params_range = {
            "algorithm": (0, 1),
            "n_neighbors": (5, 100),
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
        self.model = KNNClassifier(**params)

    def optunatune(self, trial):
        n_neighbors = trial.suggest_int("CLS_KNeighbors_n_neighbors", 5, 100, log=True)
        algorithm = trial.suggest_categorical(
            "CLS_KNeighbors_algorithm", ["brure-force", "kd-tree"]
        )
        voting_type = trial.suggest_categorical(
            "CLS_KNeighbors_voting_type", ["majority", "distance-weighted"]
        )
        metric = trial.suggest_categorical(
            "CLS_KNeighbors_metric",
            ["manhattan", "euclidean", "minkowski", "chebyshev", "cosine"],
        )
        model = KNNClassifier(
            n_neighbors=n_neighbors,
            algorithm=algorithm,
            voting_type=voting_type,
            metric=metric,
        )
        self.model = model
