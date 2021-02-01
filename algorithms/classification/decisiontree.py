from algorithms.base_algo import BaseAlgorithm
from sklearn.tree import DecisionTreeClassifier


class DecisionTree(BaseAlgorithm):
    def __init__(self):
        super(DecisionTree, self).__init__()
        self.title = "DecisionTreeClassifier"
        self.params_range = {
            "max_depth": (1, 20),
            "max_leaf_nodes": (2, 100)
            #     ...
        }
        self.model = DecisionTreeClassifier()

    def set_params(self, **params):
        params["max_leaf_nodes"] = round(params["max_leaf_nodes"])
        self.model.set_params(**params)
