from sklearn.svm import SVR

from algorithms.base_algo import BaseAlgorithm


class SVRRegression(BaseAlgorithm):
    def __init__(self):
        super(SVRRegression, self).__init__()
        self.title = "SVR"
        self.params_range = {
            "C": (1e-6, 1e6),
            "gamma": (1e-6, 1e1),
            "degree": (1, 8)
            # 'kernel': ['linear', 'poly', 'rbf']
        }
        self.model = SVR()
