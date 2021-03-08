from sklearn.svm import SVR

from algorithms.base_algo import BaseAlgorithm


class SVRRegression(BaseAlgorithm):
    def __init__(self):
        super(SVRRegression, self).__init__()
        self.title = "SVR"
        self.params_range = {
            "C": (1e-6, 1e6),
            "gamma": (1e-6, 1e1),
            "degree": (1, 8),
            'kernel': (0, 2)
        }
        self.model = SVR()

    def set_params(self, **params):
        params["kernel"] = ['linear', 'poly', 'rbf'][round(params["kernel"])]
        self.model.set_params(**params)

    def optunatune(self, trial):
        gamma = trial.suggest_float("gamma", 1e-6, 1e1, log=True)
        c = trial.suggest_float("SVR_C", 1e-6, 1e6, log=True)
        degree = trial.suggest_int("SVR_degree", 1, 8, log=True)
        model = SVR(gamma=gamma, C=c, degree=degree)
        return model