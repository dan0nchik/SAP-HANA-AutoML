from algorithms.base_algo import BaseAlgorithm
from sklearn.linear_model import Ridge


class RidgeRegression(BaseAlgorithm):
    def __init__(self):
        super(RidgeRegression, self).__init__()
        self.title = "Ridge"
        self.params_range = {"alpha": (1, 50), "tol": (1e-4, 1e-2), "solver": (0, 5)}
        self.model = Ridge()

    def set_params(self, **params):
        params["solver"] = ['svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga'][round(params["solver"])]
        self.model.set_params(**params)

    def optunatune(self, trial):
        alpha = trial.suggest_float("max_depth", 1, 50, log=True)
        tol = trial.suggest_float("tol", 1e-4, 1e-2, log=True)
        solver = trial.suggest_categorical("solver", ['svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga'])
        model = Ridge(
            alpha=alpha, tol=tol, solver=solver
        )
        return model
