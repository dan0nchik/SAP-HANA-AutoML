from hana_ml.algorithms.pal.trees import RDTRegressor

from hana_automl.algorithms.base_algo import BaseAlgorithm


class RDTReg(BaseAlgorithm):
    def __init__(self):
        super(RDTReg, self).__init__()
        self.title = "Random_Decision_Tree_Reg"
        self.params_range = {
            "n_estimators": (50, 300),
            "max_depth": (20, 56),
            "min_samples_leaf": (1, 100),
            "calculate_oob": (0, 1),
        }

    def set_params(self, **params):
        params["calculate_oob"] = [True, False][round(params["calculate_oob"])]
        params["max_depth"] = round(params["max_depth"])
        params["min_samples_leaf"] = round(params["min_samples_leaf"])
        params["n_estimators"] = round(params["n_estimators"])
        self.model = RDTRegressor(**params)

    def optunatune(self, trial):
        calculate_oob = trial.suggest_categorical(
            "REG_RDT_calculate_oob", [True, False]
        )
        n_estimators = trial.suggest_int("REG_RDT_n_estimators", 50, 300, log=True)
        max_depth = trial.suggest_int("REG_RDT_max_depth", 20, 56)
        min_samples_leaf = trial.suggest_int(
            "REG_RDT_min_samples_leaf", 1, 100, log=True
        )
        model = RDTRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            calculate_oob=calculate_oob,
            min_samples_leaf=min_samples_leaf,
        )
        self.model = model