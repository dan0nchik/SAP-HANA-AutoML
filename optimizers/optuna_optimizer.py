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
        imputer = trial.suggest_categorical(
            "imputer", ['mean', 'median', 'zero']
        )
        print(self.data.train.hasna())
        if not self.data.train.hasna() or not self.data.test.hasna() or not self.data.train.hasna():
            data = self.data.clear(num_strategy=imputer, cat_strategy=None, dropempty=False,
                                   categorical_list=None
                                   )
        else:
            data = copy.deepcopy(self.data)
        ftr: list = data.train.columns
        ftr.remove(data.target)
        ftr.remove(data.id_colm)
        model = algo.optunatune(trial)
        model.fit(data.train, key=data.id_colm, features=ftr, label=data.target,
                  categorical_variable=self.categorical_features)
        val = data.test.select(data.target)
        train = data.test.drop(data.target)
        res = model.predict(train, key=data.id_colm)
        return model.score(data.valid, key=data.id_colm, label=data.target)

    def get_tuned_params(self):
        print(
            "Title: ",
            self.tuned_params.pop("algo"),
            "\nInfo:",
            self.tuned_params
        )
