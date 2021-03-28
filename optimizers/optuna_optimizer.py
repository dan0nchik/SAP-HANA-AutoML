from hana_ml.algorithms.pal.metrics import accuracy_score

from preprocess.impsettings import ImputerSettings
import copy
from preprocess.preprocessor import Preprocessor
from optimizers.base_optimizer import BaseOptimizer
import optuna
from sklearn.model_selection import cross_val_score


class OptunaOptimizer(BaseOptimizer):
    def __init__(
            self,
            algo_list,
            data,
            problem,
            iterations,
            algo_dict,
            categorical_features=None,
            droplist_columns=None,
    ):
        self.algo_list = algo_list
        self.data = data
        self.iterations = iterations
        self.problem = problem
        self.algo_dict = algo_dict
        self.categorical_features = categorical_features
        self.droplist_columns = droplist_columns

    def tune(self):
        opt = optuna.create_study(direction="maximize")
        opt.optimize(self.objective, n_trials=self.iterations)
        self.tuned_params = opt.best_params

    def objective(self, trial):
        algo = self.algo_dict.get(trial.suggest_categorical("algo", self.algo_dict.keys()))
        pr = Preprocessor()
        data2 = pr.clean(
            data=copy.deepcopy(self.data),
            categorical_list=self.categorical_features,
            droplist_columns=self.droplist_columns
        )
        ftr: list = self.data.train.columns
        ftr.remove(self.problem)
        model = algo.optunatune(trial)
        model.fit(self.data.train, features=ftr, label=self.problem, categorical_variable=self.categorical_features)
        val = self.data.test.distinct(self.problem)
        self.data.test.drop(self.problem)
        res, stats = model.predict(self.data.test, categorical_variable=self.categorical_features)
        val.union(res)
        return accuracy_score(val, label_pred='Target', label_true=self.problem)

    def get_tuned_params(self):
        print(
            "Title: ",
            self.tuned_params.pop("algo"),
            "\nInfo:",
            self.tuned_params
        )
