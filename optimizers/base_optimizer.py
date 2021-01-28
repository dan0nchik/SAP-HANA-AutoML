import copy

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier

from pipeline.validator import Validate
from pipeline.fit import Fit
from algorithms import base_algo
from preprocess.preprocessor import Preprocessor


class BaseOptimizer:

    # алгоритм, способ обработки
    def objective(self, algo_index_tuned, **hyperparameters):
        self.algo_index = round(algo_index_tuned)
        print(self.algo_index)  # берет реш. деревья из массива
        print(hyperparameters)  # параметры от лог. регрессии -> ошибка

        self.algo_list[self.algo_index].set_params(**hyperparameters)

        Fit.fit(self.algo_list[self.algo_index], self.data.X_train, self.data.y_train)

        return Validate.val(self.algo_list[self.algo_index], self.data.X_test, self.data.y_test, self.problem)

    def optobjective(self, trial):
        algo = trial.suggest_categorical("algo", self.algo_names)
        if self.categorical_features is not None:
            catencoder = trial.suggest_categorical("catencoder", ["LabelEncoder", "OneHotEncoder"])
            pr = Preprocessor()
            if catencoder == "LabelEncoder":
                print(catencoder)
                data2 = pr.clean(
                    data=copy.copy(self.data), categorical_list=self.categorical_features, encoder_method='LabelEncoder',
                    droplist_columns=self.droplist_columns)
            elif catencoder == "OneHotEncoder":
                data2 = pr.clean(
                    data=copy.copy(self.data), categorical_list=self.categorical_features, encoder_method='OneHotEncoder',
                    droplist_columns=self.droplist_columns)
        print(algo)
        if algo == 'DecisionTree':
            max_depth = trial.suggest_int("max_depth", 1, 20, log=True)
            max_leaf_nodes = trial.suggest_int("max_leaf_nodes", 2, 100, log=True)
            criterion = trial.suggest_categorical("criterion", ["gini", "entropy"])
            model = DecisionTreeClassifier(max_depth=max_depth, max_leaf_nodes=max_leaf_nodes, criterion=criterion)
        elif algo == 'Logistic Regression':
            tol = trial.suggest_float("tol", 1e-10, 1e10, log=True)
            c = trial.suggest_float("C", 0.1, 10.0, log=True)
            model = LogisticRegression(tol=tol, C=c)
        score = cross_val_score(model, data2.X_train, data2.y_train.values.ravel(), n_jobs=-1, cv=3)
        accuracy = score.mean()
        return accuracy

    def __init__(self, algo_list: list, data, iterations, problem, categorical_features=None, algo_names: list = None,
                 droplist_columns=None):
        self.data = data
        self.categorical_features = categorical_features
        self.algo_list = algo_list
        self.iter = iterations
        self.problem = problem
        self.tuned_params = {}
        self.algo_index = 0
        self.algo_names = algo_names
        self.droplist_columns = droplist_columns

    def get_tuned_params(self):
        return self.algo_list[self.algo_index].title, self.tuned_params
