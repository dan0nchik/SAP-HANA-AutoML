from bayes_opt import BayesianOptimization
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import models


class Optimizer:

    def objective(self, **hyperparameters):
        model = self.model(hyperparameters)
        models.Fit.fit(model, self.X_train, self.y_train)
        return models.Validate.val(model, self.X_test, self.y_test)

    def __init__(self, model, X_train, y_train, X_test, y_test, iterations):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.model = model
        self.bounds = model.get_params()
        self.iter = iterations

    def search_hp(self):
        optimizer = BayesianOptimization(
            f=self.objective,
            pbounds=self.bounds,
            random_state=1
        )
        optimizer.maximize(
            init_points=2,
            n_iter=self.iter
        )
        return optimizer.max


class DecisionTreeOptimizer(Optimizer):

    def objective(self, max_depth):
        model = self.model.set_params(max_depth=max_depth)
        models.Fit.fit(model, self.X_train, self.y_train)
        return models.Validate.val(model, self.X_test, self.y_test)

    def __init__(self, model, X_train, y_train, X_test, y_test, iterations):
        super().__init__(model, X_train, y_train, X_test, y_test, iterations)
        self.bounds = {'max_depth': (1, 30)}
