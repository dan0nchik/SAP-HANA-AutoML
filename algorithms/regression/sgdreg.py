from sklearn.linear_model import SGDRegressor

from algorithms.base_algo import BaseAlgorithm


class SGDRegression(BaseAlgorithm):
    def __init__(self):
        super(SGDRegression, self).__init__()
        self.title = "SGDRegressor"
        self.params_range = {
            "loss": (0, 3),
            "penalty": (0, 1),
            "alpha": (1e-5, 1e-3),
            "tol": (1e-3, 0.5)
        }
        self.model = SGDRegressor()

    def set_params(self, **params):
        params["loss"] = ["squared_loss", "huber", "epsilon_insensitive", "squared_epsilon_insensitive"][
            round(params["loss"])]
        params["penalty"] = ["l2", "l1"][round(params["penalty"])]
        self.model.set_params(**params)

    def optunatune(self, trial):
        alpha = trial.suggest_float("SGDReg_alpha", 1e-5, 1e-3)
        tol = trial.suggest_float("SGDReg_tol", 1e-3, 0.5)
        loss = trial.suggest_categorical("SGDReg_loss", ["squared_loss", "huber", "epsilon_insensitive",
                                                         "squared_epsilon_insensitive"])
        penalty = trial.suggest_categorical("SGDReg_penalty", ["l2", "l1"])
        model = SGDRegressor(
            alpha=alpha, tol=tol, loss=loss, penalty=penalty
        )
        return model
