from hana_ml.algorithms.pal.svm import SVC

from algorithms.base_algo import BaseAlgorithm


class SVCls(BaseAlgorithm):
    def __init__(self):
        super(SVCls, self).__init__()
        self.title = "SupportVectorClassifier"
        self.params_range = {
            "c": (10, 150),
            "kernel": (0, 3),
            "shrink": (0, 1),
            "tol": (0.001, 1),
            "scale_info": (0, 1),
        }

    def set_params(self, **params):
        params1 = {}
        params1["c"] = params.get("с", 30)
        params1["kernel"] = ["linear", "poly", "rbf", "sigmoid"][round(params["kernel"])]
        params1["shrink"] = [True, False][round(params["shrink"])]
        params1["scale_info"] = ["no", "standardization", "rescale"][
            round(params["scale_info"])
        ]
        self.model = SVC(**params1)

    def optunatune(self, trial):
        c = trial.suggest_int("CLS_SV_c", 50, 300, log=True)
        kernel = trial.suggest_categorical(
            "CLS_SV_kernel", ["linear", "poly", "rbf", "sigmoid"]
        )
        shrink = trial.suggest_categorical("CLS_SV_shrink", [True, False])
        tol = trial.suggest_float("CLS_SV_tol", 0.001, 1, log=True)
        scale_info = trial.suggest_categorical(
            "CLS_SV_scale_info", ["no", "standardization", "rescale"]
        )
        model = SVC(c=c, kernel=kernel, shrink=shrink, scale_info=scale_info, tol=tol)
        self.model = model
