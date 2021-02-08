from algorithms.base_algo import BaseAlgorithm
from sklearn.tree import DecisionTreeClassifier


class DecisionTreeCls(BaseAlgorithm):
    def __init__(self):
        super(DecisionTreeCls, self).__init__()
        self.title = "DecisionTreeClassifier"
        self.params_range = {
            "max_depth": (1, 20),
            "max_leaf_nodes": (2, 100),
            "criterion": (0, 1)
        }
        self.model = DecisionTreeClassifier()

    def set_params(self, **params):
        params["max_leaf_nodes"] = round(params["max_leaf_nodes"])
        params["max_depth"] = round(params["max_depth"])
        params["criterion"] = ["gini", "entropy"][round(params["criterion"])]
        self.model.set_params(**params)

    def optunatune(self, trial):
        max_depth = trial.suggest_int("max_depth", 1, 20, log=True)
        max_leaf_nodes = trial.suggest_int("max_leaf_nodes", 2, 100, log=True)
        criterion = trial.suggest_categorical("criterion", ["gini", "entropy"])
        model = DecisionTreeClassifier(
            max_depth=max_depth, max_leaf_nodes=max_leaf_nodes, criterion=criterion
        )
        return model
