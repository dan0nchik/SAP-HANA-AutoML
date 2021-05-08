from hana_ml.algorithms.pal.linear_model import LogisticRegression
from hana_ml.algorithms.pal.unified_classification import UnifiedClassification

from hana_automl.algorithms.base_algo import BaseAlgorithm


class LogRegressionCls(BaseAlgorithm):
    def __init__(self, binominal):
        super(LogRegressionCls, self).__init__()
        self.title = "Logistic Regression"
        self.binominal = binominal
        self.params_range = {
            "max_iter": (100, 1000),
        }
        if self.binominal:
            self.params_range["solver"] = (0, 5)
        else:
            self.params_range["solver"] = (0, 2)

    def set_params(self, **params):
        params["max_iter"] = round(params["max_iter"])
        params["solver"] = ['auto', 'cyclical', 'lbfgs', 'newton', 'stochastic', 'proximal'][round(params["solver"])]
        if self.binominal:
            params["multi_class"] = False
        else:
            params["multi_class"] = True
        self.model = UnifiedClassification(func='LogisticRegression', **params)

    def optunatune(self, trial):

        max_iter = trial.suggest_int("LGReg_max_iter", 100, 1000, log=True)
        if self.binominal:
            solver = trial.suggest_categorical("LGReg_solver",
                                               ['auto', 'newton', 'cyclical', 'lbfgs', 'stochastic', 'proximal'] )
        else:
            solver = trial.suggest_categorical("LGReg_solver",
                                               ['auto', 'cyclical', 'lbfgs'])
        if self.binominal:
            multi_class = False
        else:
            multi_class = True
        model = UnifiedClassification(func='LogisticRegression', max_iter=max_iter,
                                      multi_class=multi_class, solver=solver)
        self.model = model
