from algorithms.base_algo import BaseAlgorithm
from sklearn.tree import DecisionTreeClassifier


class DecisionTree(BaseAlgorithm):
    def __init__(self):
        super(DecisionTree, self).__init__()
        self.title = 'DecisionTreeClassifier'
        self.params_range = {
            'max_depth': (1, 30),
            'min_samples_split': (2, 4)
            #     ...
        }
