from algorithms.base_algo import BaseAlgorithm
from sklearn.tree import DecisionTreeRegressor


class DecisionTreeReg(BaseAlgorithm):
    def __init__(self):
        super(DecisionTreeReg, self).__init__()
        self.title = "DecisionTreeRegressor"
        self.params_range = {
            "max_depth": (1, 20),
            "max_leaf_nodes": (2, 100),
            "criterion": (0, 3),
            "splitter": (0, 1),
            "max_features": (0, 2)

        }
        self.model = DecisionTreeRegressor()

    def set_params(self, **params):
        params["max_leaf_nodes"] = round(params["max_leaf_nodes"])
        params["max_depth"] = round(params["max_depth"])
        params["criterion"] = ["mse", "friedman_mse", "mae", "poisson"][round(params["criterion"])]
        params["splitter"] = ["best", "random"][round(params["splitter"])]
        params["max_features"] = ["auto", "sqrt", "log2"][round(params["max_features"])]
        self.model.set_params(**params)

    def optunatune(self, trial):
        max_depth = trial.suggest_int("DTReg_max_depth", 1, 20, log=True)
        max_leaf_nodes = trial.suggest_int("DTReg_max_leaf_nodes", 2, 100, log=True)
        criterion = trial.suggest_categorical("DTReg_criterion", ["mse", "friedman_mse", "mae", "poisson"])
        splitter = trial.suggest_categorical("DTReg_splitter", ["best", "random"])
        max_features = trial.suggest_categorical("DTReg_max_features", ["auto", "sqrt", "log2"])
        model = DecisionTreeRegressor(
            max_depth=max_depth, max_leaf_nodes=max_leaf_nodes, criterion=criterion,
            splitter=splitter, max_features=max_features
        )
        return model
