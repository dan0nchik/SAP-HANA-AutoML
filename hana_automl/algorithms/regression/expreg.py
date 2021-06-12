from hana_ml.algorithms.pal.regression import ExponentialRegression

from hana_automl.algorithms.base_algo import BaseAlgorithm


class ExponentialReg(BaseAlgorithm):
    def __init__(self):
        super(ExponentialReg, self).__init__()
        self.title = "ExponentialRegressor"
        self.params_range = {
            "decomposition": (0, 3),
            "adjusted_r2": (0, 1),
        }

    def set_params(self, **params):
        params["decomposition"] = ["LU", "QR", "SVD", "Cholesky"][
            round(params["decomposition"])
        ]
        params["adjusted_r2"] = round(params["adjusted_r2"])
        self.tuned_params = params
        self.model = ExponentialRegression(**params)

    def optunatune(self, trial):
        decomposition = trial.suggest_categorical(
            "decomposition", ["LU", "QR", "SVD", "Cholesky"]
        )
        adjusted_r2 = trial.suggest_categorical("adjusted_r2", [True, False])
        model = ExponentialRegression(
            decomposition=decomposition,
            adjusted_r2=adjusted_r2,
        )
        self.model = model
