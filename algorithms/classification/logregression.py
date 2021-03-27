from hana_ml.algorithms.pal.linear_model import LogisticRegression

from algorithms.base_algo import BaseAlgorithm



class LogRegression(BaseAlgorithm):
    def __init__(self):
        super(LogRegression, self).__init__()
        self.title = "Logistic Regression"
        self.params_range = {
            "max_iter": (100, 1000),
            "solver": (0, 2)
        }
        self.model = LogisticRegression()

    def set_params(self, **params):
        params["max_iter"] = round(params["max_iter"])
        params["solver"] = ['auto', 'lbfgs', 'cyclical'][round(params["solver"])]
        params["multi_class"] = True
        self.model.set_params(**params)

    def optunatune(self, trial):
        solver = trial.suggest_categorical("solver", ['auto', 'lbfgs', 'cyclical'])
        max_iter = trial.suggest_int("LGReg_max_iter", 100, 1000, log=True)
        model = LogisticRegression(solver=solver, max_iter=max_iter, multi_class=True)
        return model
