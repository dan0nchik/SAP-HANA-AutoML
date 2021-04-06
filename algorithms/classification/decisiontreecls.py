from hana_ml.algorithms.pal.trees import DecisionTreeClassifier

from algorithms.base_algo import BaseAlgorithm


class DecisionTreeCls(BaseAlgorithm):
    def __init__(self):
        super(DecisionTreeCls, self).__init__()
        self.title = "DecisionTreeClassifier"
        self.params_range = {
            "algorithm": (0, 2),
            "max_depth": (2, 80),
            "min_records_of_leaf": (1, 100),
            "min_records_of_parent": (2, 100),
        }
        self.model = DecisionTreeClassifier()

    def set_params(self, **params):
        params["algorithm"] = ["c45", "chaid", "cart"][round(params["algorithm"])]
        params["min_records_of_leaf"] = round(params["min_records_of_leaf"])
        params["min_records_of_parent"] = round(params["min_records_of_parent"])
        params["max_depth"] = round(params["max_depth"])
        self.model = DecisionTreeClassifier(**params)

    def optunatune(self, trial):
        algorithm = trial.suggest_categorical("DTC_algorithm", ["c45", "chaid", "cart"])
        max_depth = trial.suggest_int("DTC_max_depth", 2, 50, log=True)
        min_records_of_leaf = trial.suggest_int(
            "DTC_min_records_of_leaf", 2, 100, log=True
        )
        min_records_of_parent = trial.suggest_int(
            "DTC_min_records_of_parent", 2, 100, log=True
        )
        model = DecisionTreeClassifier(
            algorithm=algorithm,
            max_depth=max_depth,
            min_records_of_leaf=min_records_of_leaf,
            min_records_of_parent=min_records_of_parent,
        )
        return model
