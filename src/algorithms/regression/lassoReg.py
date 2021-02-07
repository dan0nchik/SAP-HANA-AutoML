from algorithms.base_algo import BaseAlgorithm
from sklearn.linear_model import Lasso


class LassoReg(BaseAlgorithm):
    def __init__(self):
        super(LassoReg, self).__init__()
        self.title = "Lasso"
        self.params_range = {"alpha": (0.1, 1)}
        self.model = Lasso()

    def optunatune(self, trial):
        alpha = trial.suggest_float("alpha", 0.1, 1, log=True)
        model = Lasso(
            alpha=alpha
        )
        return model
