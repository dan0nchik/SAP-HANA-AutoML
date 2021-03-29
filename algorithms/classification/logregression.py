from hana_ml.algorithms.pal.linear_model import LogisticRegression

from algorithms.base_algo import BaseAlgorithm


class LogRegression(BaseAlgorithm):
    def __init__(self):
        super(LogRegression, self).__init__()
        self.title = "Logistic Regression"
        self.params_range = {
            "max_iter": (100, 1000),
        }
        self.model = LogisticRegression()

    def set_params(self, **params):
        params["max_iter"] = round(params["max_iter"])

        params["multi_class"] = True
        self.model = LogisticRegression(**params)

    def optunatune(self, trial):

        max_iter = trial.suggest_int("LGReg_max_iter", 100, 1000, log=True)
        model = LogisticRegression(max_iter=max_iter, multi_class=False)
        return model
