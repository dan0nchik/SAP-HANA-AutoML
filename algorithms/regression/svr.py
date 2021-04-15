from hana_ml.algorithms.pal.svm import SVR

from algorithms.base_algo import BaseAlgorithm


class SVReg(BaseAlgorithm):
    def __init__(self):
        super(SVReg, self).__init__()
        self.title = "SupportVectorRegressor"
        self.params_range = {
            "c": (50, 300),
            "kernel": (0, 3),
            "shrink": (0, 1),
            "tol": (0.001, 1),
            "scale_info": (0, 1),
        }

    def set_params(self, **params):
        params["c"] = round(params["с"])
        params["kernel"] = ["linear", "poly", "rbf", "sigmoid"][round(params["kernel"])]
        params["shrink"] = [True, False][round(params["shrink"])]
        params["scale_info"] = ["no", "standardization", "rescale"][
            round(params["scale_info"])
        ]
        self.model = SVR(**params)

    def optunatune(self, trial):
        c = trial.suggest_int("REG_SV_c", 50, 300, log=True)
        kernel = trial.suggest_categorical(
            "REG_SV_kernel", ["linear", "poly", "rbf", "sigmoid"]
        )
        shrink = trial.suggest_categorical("REG_SV_shrink", [True, False])
        tol = trial.suggest_float("REG_SV_tol", 0.001, 1, log=True)
        scale_info = trial.suggest_categorical(
            "REG_SV_scale_info", ["no", "standardization", "rescale"]
        )
        model = SVR(c=c, kernel=kernel, shrink=shrink, scale_info=scale_info, tol=tol)
        self.model = model
