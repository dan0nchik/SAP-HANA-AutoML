from hana_ml.algorithms.pal.regression import GLM

from hana_automl.algorithms.base_algo import BaseAlgorithm


class GLMReg(BaseAlgorithm):
    def __init__(self):
        super(GLMReg, self).__init__()
        self.title = "GLMRegressor"
        self.params_range = {"family": (0, 2)}
        self.model = GLM()

    def set_params(self, **params):
        params["family"] = ["gaussian", "normal", "poisson"][round(params["family"])]
        self.tuned_params = params
        self.model = GLM(**params)

    def optunatune(self, trial):
        # TODO: additional hp
        family = trial.suggest_categorical("family", ["gaussian", "normal", "poisson"])
        model = GLM(family=family)
        self.model = model
