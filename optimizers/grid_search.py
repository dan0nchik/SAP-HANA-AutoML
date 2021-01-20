from optimizers.base_optimizer import BaseOptimizer
from sklearn.model_selection import GridSearchCV


class GridSearch(BaseOptimizer):
    def __init__(self, algorithm, data, iterations, problem):
        super(GridSearch, self).__init__(algorithm, data, iterations, problem)
        opt = GridSearchCV(self.algorithm.model, self.algorithm.get_params(), cv=iterations, verbose=False,
                           return_train_score=True, n_jobs=-1)
        opt.fit(self.X_train, self.y_train)
        self.tuned_params = {'target': str(opt.best_score_),
                             'params': str(opt.best_params_)}