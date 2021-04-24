from hana_ml.algorithms.pal.linear_model import LogisticRegression

from hana_automl.algorithms.base_algo import BaseAlgorithm


class LogRegressionCls(BaseAlgorithm):
    def __init__(self):
        super(LogRegressionCls, self).__init__()
        self.title = "Logistic Regression"
        self.params_range = {
            "max_iter": (100, 1000),
        }

    def set_params(self, **params):
        params["max_iter"] = round(params["max_iter"])

        # ValueError: class_map0 and class_map1 are mandatory
        # when `label` column type is VARCHAR or NVARCHAR.
        # params["multi_class"] = True
        self.model = LogisticRegression(**params)

    def optunatune(self, trial):

        max_iter = trial.suggest_int("LGReg_max_iter", 100, 1000, log=True)
        model = LogisticRegression(max_iter=max_iter, multi_class=False)
        self.model = model
