from algorithms.base_algo import BaseAlgorithm
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression


class LogRegression(BaseAlgorithm):
    def __init__(self):
        super(LogRegression, self).__init__()
        self.title = "Logistic Regression"
        self.params_range = {
            "C": (1.0, 10)
            #     ...
        }
        self.model = LogisticRegression()
