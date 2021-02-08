from algorithms.base_algo import BaseAlgorithm
from sklearn.linear_model import Lasso


class LassoReg(BaseAlgorithm):
    def __init__(self):
        super(LassoReg, self).__init__()
        self.title = "Lasso"
        self.params_range = {"alpha": (0.1, 1),
                             "normalize": (0, 1)}
        self.model = Lasso()

    def optunatune(self, trial):
        alpha = trial.suggest_float("Lasso_alpha", 0.1, 1, log=True)
        normalize = trial.suggest_categorical("Lasso_normalize", [True, False])
        model = Lasso(
            alpha=alpha,
            normalize=normalize
        )
        return model
