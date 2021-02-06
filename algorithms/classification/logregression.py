from algorithms.base_algo import BaseAlgorithm
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression


class LogRegression(BaseAlgorithm):
    def __init__(self):
        super(LogRegression, self).__init__()
        self.title = "Logistic Regression"
        self.params_range = {
            "C": (1.0, 10),
            "tol": (1e-10, 1e10)
        }
        self.model = LogisticRegression()

    def optunatune(self, trial):
        tol = trial.suggest_float("tol", 1e-10, 1e10, log=True)
        c = trial.suggest_float("C", 0.1, 10.0, log=True)
        model = LogisticRegression(tol=tol, C=c)
        return model
