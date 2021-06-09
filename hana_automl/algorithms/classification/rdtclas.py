from hana_ml.algorithms.pal.trees import RDTClassifier

from hana_automl.algorithms.base_algo import BaseAlgorithm


class RDTCls(BaseAlgorithm):
    def __init__(self):
        super(RDTCls, self).__init__()
        super().__init__()
        self.title = "RandomDecisionTreeClassifier"
        self.params_range = {
            "n_estimators": (100, 1000),
            "max_depth": (10, 50),
            "min_samples_leaf": (1, 20),
            "calculate_oob": (0, 1),
        }

    def set_params(self, **params):
        params["calculate_oob"] = [True, False][round(params["calculate_oob"])]
        params["max_depth"] = round(params["max_depth"])
        params["min_samples_leaf"] = round(params["min_samples_leaf"])
        params["n_estimators"] = round(params["n_estimators"])
        # self.model = UnifiedClassification(func='RandomDecisionTree', **params)
        self.tuned_params = params
        self.model = RDTClassifier(**params)

    def optunatune(self, trial):
        calculate_oob = trial.suggest_categorical("calculate_oob", [True, False])
        n_estimators = trial.suggest_int("n_estimators", 100, 1000, log=True)
        max_depth = trial.suggest_int("max_depth", 10, 50, log=True)
        min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 20, log=True)
        """model = UnifiedClassification(
            func='RandomDecisionTree',
            n_estimators=n_estimators,
            max_depth=max_depth,
            calculate_oob=calculate_oob,
            min_samples_leaf=min_samples_leaf,
        )
        """
        model = RDTClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            calculate_oob=calculate_oob,
            min_samples_leaf=min_samples_leaf,
        )
        self.model = model
