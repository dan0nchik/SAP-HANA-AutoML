from hana_ml.algorithms.pal.trees import DecisionTreeRegressor

from algorithms.base_algo import BaseAlgorithm


class DecisionTreeReg(BaseAlgorithm):
    def __init__(self):
        super(DecisionTreeReg, self).__init__()
        self.title = "DecisionTreeRegressor"
        self.params_range = {
            "max_depth": (2, 50),
            "min_records_of_leaf": (1, 100),
            "min_records_of_parent": (2, 100)
        }
        self.model = DecisionTreeRegressor()

    def set_params(self, **params):
        params["algorithm"] = 'cart'
        params["min_records_of_leaf"] = round(params["min_records_of_leaf"])
        params["min_records_of_parent"] = round(params["min_records_of_parent"])
        params["max_depth"] = round(params["max_depth"])
        self.model = DecisionTreeRegressor(**params)

    def optunatune(self, trial):
        algorithm = 'cart'
        max_depth = trial.suggest_int("DTR_max_depth", 2, 100, log=True)
        min_records_of_leaf = trial.suggest_int("DTR_min_records_of_leaf", 2, 100, log=True)
        min_records_of_parent = trial.suggest_int("DTR_min_records_of_parent", 2, 100, log=True)
        model = DecisionTreeRegressor(algorithm=algorithm, max_depth=max_depth,
                                      min_records_of_leaf=min_records_of_leaf,
                                      min_records_of_parent=min_records_of_parent
                                      )
        return model
