from skopt.space import Integer, Categorical

from algorithms.base_algo import BaseAlgorithm
from sklearn.tree import DecisionTreeClassifier


class DecisionTree(BaseAlgorithm):
    def __init__(self):
        super(DecisionTree, self).__init__()
        self.title = 'DecisionTreeClassifier'
        self.params_range = {
            'model': Categorical([DecisionTreeClassifier()]),
            'model__max_depth': Integer(1, 20),
            'model__splitter': Categorical(['best', 'random'])
        }
        self.model = DecisionTreeClassifier()
