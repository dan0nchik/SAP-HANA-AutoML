from algorithms.base_algo import BaseAlgorithm
from sklearn.neighbors import KNeighborsClassifier


class KNeighbors(BaseAlgorithm):
    def __init__(self):
        super(KNeighbors, self).__init__()
        self.title = "KNeighborsClassifier"
        self.params_range = {
            "n_neighbors": (5, 200),
            "weights": (0, 1),
            "algorithm": (0, 2),
            "leaf_size": (30, 400),
        }
        self.model = KNeighborsClassifier()

    def set_params(self, **params):
        params["weights"] = ["uniform", "distance"][round(params["weights"])]
        params["algorithm"] = ["ball_tree", "kd_tree", "brute"][round(params["algorithm"])]
        params["n_neighbors"] = round(params["n_neighbors"])
        params["leaf_size"] = round(params["leaf_size"])
        self.model.set_params(**params)

    def optunatune(self, trial):
        n_neighbors = trial.suggest_int("KNeighbors_n_neighbors", 5, 200, log=True)
        leaf_size = trial.suggest_int("KNeighbors_leaf_size", 30, 400, log=True)
        weights = trial.suggest_categorical("KNeighbors_weights", ["uniform", "distance"])
        algorithm = trial.suggest_categorical("KNeighbors_algorithm", ["ball_tree", "kd_tree", "brute"])
        model = KNeighborsClassifier(
            n_neighbors=n_neighbors, leaf_size=leaf_size, weights=weights, algorithm=algorithm
        )
        return model
