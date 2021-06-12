from hana_ml.algorithms.pal.trees import DecisionTreeRegressor

from hana_automl.algorithms.base_algo import BaseAlgorithm


class DecisionTreeReg(BaseAlgorithm):
    def __init__(self):
        super(DecisionTreeReg, self).__init__()
        self.title = "DecisionTreeRegressor"
        self.params_range = {
            "max_depth": (2, 56),
            "min_records_of_leaf": (1, 100),
            "min_records_of_parent": (2, 100),
        }

    def set_params(self, **params):
        params["algorithm"] = "cart"
        params["min_records_of_leaf"] = round(params["min_records_of_leaf"])
        params["min_records_of_parent"] = round(params["min_records_of_parent"])
        params["max_depth"] = round(params["max_depth"])
        self.tuned_params = params
        self.model = DecisionTreeRegressor(**params)

    def optunatune(self, trial):
        algorithm = "cart"
        max_depth = trial.suggest_int("max_depth", 2, 56, log=True)
        min_records_of_leaf = trial.suggest_int("min_records_of_leaf", 1, 20, log=True)
        min_records_of_parent = trial.suggest_int(
            "min_records_of_parent", 2, 20, log=True
        )
        model = DecisionTreeRegressor(
            algorithm=algorithm,
            max_depth=max_depth,
            min_records_of_leaf=min_records_of_leaf,
            min_records_of_parent=min_records_of_parent,
        )
        self.model = model
