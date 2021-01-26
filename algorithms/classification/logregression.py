from skopt.space import Categorical, Real

from algorithms.base_algo import BaseAlgorithm
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression


class LogRegression(BaseAlgorithm):
    def __init__(self):
        super(LogRegression, self).__init__()
        self.title = 'Logistic Regression'
        self.params_range = {
            'model': Categorical([LogisticRegression()]),
            'model__C': Real(1e-6, 1e+6, prior='log-uniform')
            #     ...
        }
        self.model = LogisticRegression()
