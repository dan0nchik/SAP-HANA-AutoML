from algorithms.base_algo import BaseAlgorithm
from sklearn.linear_model import SGDClassifier


class SGD(BaseAlgorithm):
    def __init__(self):
        super(SGD, self).__init__()
        self.title = "SGDClassifier"
        self.params_range = {
            "alpha": (1e-5, 1e-3),
            "tol": (1e-3, 0.5),
            "loss": (0, 3),
            "penalty": (0, 1)
        }
        self.model = SGDClassifier()

    def set_params(self, **params):
        params["loss"] = ["hinge", "log", "modified_huber", "squared_hinge", "perceptron"][round(params["loss"])]
        params["penalty"] = ["l2", "l1"][round(params["penalty"])]
        self.model.set_params(**params)

    def optunatune(self, trial):
        alpha = trial.suggest_float("SGD_alpha", 1e-5, 1e-3)
        tol = trial.suggest_float("SGD_tol", 1e-3, 0.5)
        loss = trial.suggest_categorical("SGD_loss", ["hinge", "log", "modified_huber", "squared_hinge", "perceptron"])
        penalty = trial.suggest_categorical("SGD_penalty", ["l2", "l1"])
        model = SGDClassifier(
            alpha=alpha, tol=tol, loss=loss, penalty=penalty
        )
        return model
