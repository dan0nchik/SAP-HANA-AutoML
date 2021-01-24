from pipeline.validator import Validate
from pipeline.fit import Fit
from algorithms import base_algo


class BaseOptimizer:

    # алгоритм, способ обработки
    def objective(self, algo_index_tuned, **hyperparameters):
        self.algo_index = round(algo_index_tuned)
        print(self.algo_index) # берет реш. деревья из массива
        print(hyperparameters) # параметры от лог. регрессии -> ошибка

        self.algo_list[self.algo_index].set_params(**hyperparameters)

        Fit.fit(self.algo_list[self.algo_index], self.X_train, self.y_train)

        return Validate.val(self.algo_list[self.algo_index], self.X_test, self.y_test, self.problem)

    def __init__(self, algo_list: list, data, iterations, problem):
        self.X_train = data.X_train
        self.y_train = data.y_train
        self.X_test = data.X_test
        self.y_test = data.y_test
        self.algo_list = algo_list
        self.iter = iterations
        self.problem = problem
        self.tuned_params = {}
        self.algo_index = 0

    def get_tuned_params(self):
        return self.algo_list[self.algo_index].title, self.tuned_params
