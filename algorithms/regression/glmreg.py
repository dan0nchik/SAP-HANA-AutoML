from hana_ml.algorithms.pal.regression import GLM

from algorithms.base_algo import BaseAlgorithm



class GLMRegression(BaseAlgorithm):
    def __init__(self):
        super(GLMRegression, self).__init__()
        self.title = "Generalized linear model"
        self.params_range = {
            "family": (0, 7)
        }
        self.model = GLM()

    def set_params(self, **params):
        params["family"] = ['gaussian', 'normal', 'poisson', 'binomial', 'gamma', 'inversegaussian', 'negativebinomial', 'ordinal'][round(params["family"])]
        self.model.set_params(**params)

    def optunatune(self, trial):
        family = trial.suggest_categorical("family", ['gaussian', 'normal', 'poisson', 'binomial', 'gamma', 'inversegaussian', 'negativebinomial', 'ordinal'])
        model = GLM(family=family)
        return model
