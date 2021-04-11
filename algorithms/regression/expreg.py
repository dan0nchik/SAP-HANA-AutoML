from hana_ml.algorithms.pal.regression import ExponentialRegression

from algorithms.base_algo import BaseAlgorithm


class ExponentialReg(BaseAlgorithm):
    def __init__(self):
        super(ExponentialReg, self).__init__()
        self.title = "ExponentialRegression"
        self.params_range = {
            "decomposition": (0, 3),
            "adjusted_r2": (0, 1),
        }

    def set_params(self, **params):
        params["decomposition"] = ["LU", "QR", "SVD", "Cholesky"][
            round(params["decomposition"])
        ]
        params["adjusted_r2"] = round(params["adjusted_r2"])
        self.model = ExponentialRegression(**params)

    def optunatune(self, trial):
        decomposition = trial.suggest_categorical(
            "REG_EXP_decomposition", ["LU", "QR", "SVD", "Cholesky"]
        )
        adjusted_r2 = trial.suggest_categorical("REG_EXP_adjusted_r2", [True, False])
        model = ExponentialRegression(
            decomposition=decomposition,
            adjusted_r2=adjusted_r2,
        )
        return model
