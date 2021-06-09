from hana_ml.algorithms.pal.linear_model import LogisticRegression

from hana_automl.algorithms.base_algo import BaseAlgorithm


class LogRegressionCls(BaseAlgorithm):
    def __init__(self, binominal, class_map0=None, class_map1=None):
        super(LogRegressionCls, self).__init__()
        self.title = "LogisticRegressionClassifier"
        self.binominal = binominal
        self.class_map0 = class_map0
        self.class_map1 = class_map1
        self.params_range = {
            "max_iter": (100, 1000),
        }
        if self.binominal:
            self.params_range["solver"] = (0, 0)
        else:
            self.params_range["solver"] = (0, 0)

    def set_params(self, **params):
        params["max_iter"] = round(params["max_iter"])
        params["solver"] = ["auto"][round(params["solver"])]
        if self.binominal:
            params["multi_class"] = False
        else:
            params["multi_class"] = True
        if self.class_map0 is not None:
            if type(self.class_map0) is str and type(self.class_map1) is str:
                params["class_map0"] = self.class_map0
                params["class_map1"] = self.class_map1
        # self.model = UnifiedClassification(func='LogisticRegression', **params)
        self.tuned_params = params
        self.model = LogisticRegression(**params)

    def optunatune(self, trial):

        max_iter = trial.suggest_int("max_iter", 100, 1000, log=True)
        if self.binominal:
            solver = trial.suggest_categorical("solver", ["auto"])
        else:
            solver = trial.suggest_categorical("solver", ["auto"])
        if self.binominal:
            multi_class = False
        else:
            multi_class = True
        if self.class_map0 is not None:
            if type(self.class_map0) is str and type(self.class_map1) is str:
                params = {"class_map0": self.class_map0, "class_map1": self.class_map1}
            else:
                params = {}
        else:
            params = {}
        """model = UnifiedClassification(func='LogisticRegression', max_iter=max_iter,
                                      multi_class=multi_class, solver=solver)
        """
        model = LogisticRegression(
            max_iter=max_iter, multi_class=multi_class, solver=solver, **params
        )
        self.model = model
