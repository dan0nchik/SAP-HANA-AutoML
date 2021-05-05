from hana_ml.algorithms.pal.naive_bayes import NaiveBayes

from hana_automl.algorithms.base_algo import BaseAlgorithm


class NBayesCls(BaseAlgorithm):
    def __init__(self):
        super(NBayesCls, self).__init__()
        self.title = "NaiveBayes"
        self.params_range = {
            "alpha": (0, 2),
            "discretization": (0, 1),
        }
        self.model = NaiveBayes()

    def set_params(self, **params):
        params["discretization"] = ["no", "supervised"][round(params["discretization"])]
        self.model = NaiveBayes(**params)

    def optunatune(self, trial):
        alpha = trial.suggest_float("CLS_NaiveBayes_alpha", 1e-2, 100, log=True)
        discretization = trial.suggest_categorical(
            "CLS_NaiveBayes_discretization", ["no", "supervised"]
        )
        model = NaiveBayes(alpha=alpha, discretization=discretization)
        self.model = model
