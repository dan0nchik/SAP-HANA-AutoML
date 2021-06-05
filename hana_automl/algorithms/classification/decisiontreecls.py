from hana_ml.algorithms.pal.trees import DecisionTreeClassifier

from hana_automl.algorithms.base_algo import BaseAlgorithm


class DecisionTreeCls(BaseAlgorithm):
    def __init__(self):
        super(DecisionTreeCls, self).__init__()
        self.title = "DecisionTreeClassifier"
        self.params_range = {
            "algorithm": (0, 2),
            "max_depth": (2, 50),
            "min_records_of_leaf": (1, 20),
            "min_records_of_parent": (2, 20),
        }

    def set_params(self, **params):
        params["algorithm"] = ["c45", "chaid", "cart"][round(params["algorithm"])]
        params["min_records_of_leaf"] = round(params["min_records_of_leaf"])
        params["min_records_of_parent"] = round(params["min_records_of_parent"])
        params["max_depth"] = round(params["max_depth"])
        # self.model = UnifiedClassification(func='DecisionTree', **params)
        self.tuned_params = params
        self.model = DecisionTreeClassifier(**params)

    def optunatune(self, trial):
        params = dict()
        params["algorithm"] = trial.suggest_categorical(
            "algorithm", ["c45", "chaid", "cart"]
        )
        params["max_depth"] = trial.suggest_int("max_depth", 2, 50, log=True)
        params["min_records_of_leaf"] = trial.suggest_int(
            "min_records_of_leaf", 1, 20, log=True
        )
        params["min_records_of_leaf"] = trial.suggest_int(
            "min_records_of_parent", 2, 20, log=True
        )
        # model = UnifiedClassification(func='DecisionTree', **params)
        model = DecisionTreeClassifier(**params)
        self.model = model
