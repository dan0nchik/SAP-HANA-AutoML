from algorithms.base_algo import BaseAlgorithm
from sklearn.linear_model import Ridge


class RidgeRegression(BaseAlgorithm):
    def __init__(self):
        super(RidgeRegression, self).__init__()
        self.title = 'Ridge'
        self.model = Ridge()
        self.params_range = {
            'alpha': (1, 2)
        }
