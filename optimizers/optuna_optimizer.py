from hana_ml.algorithms.pal.metrics import accuracy_score

from preprocess.impsettings import ImputerSettings
import copy
from preprocess.preprocessor import Preprocessor
from optimizers.base_optimizer import BaseOptimizer
import optuna


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
        self.model = None

    def tune(self):
        opt = optuna.create_study(direction="maximize")
        opt.optimize(self.objective, n_trials=self.iterations)
        self.tuned_params = opt.best_params

    def objective(self, trial):
        algo = self.algo_dict.get(
            trial.suggest_categorical("algo", self.algo_dict.keys())
        )
        imputer = trial.suggest_categorical("imputer", ["mean", "median", "zero"])
        data = self.data.clear(
            num_strategy=imputer,
            cat_strategy=None,
            dropempty=False,
            categorical_list=None,
        )
        model = algo.optunatune(trial)
        ftr: list = data.train.columns
        ftr.remove(data.target)
        ftr.remove(data.id_colm)
        model.fit(
            data.train,
            key=data.id_colm,
            features=ftr,
            label=data.target,
            categorical_variable=self.categorical_features,
        )
        self.model = model
        return model.score(data.valid, key=data.id_colm, label=data.target)

    def get_tuned_params(self):
        print("Title: ", self.tuned_params.pop("algo"), "\nInfo:", self.tuned_params)

    def get_model(self):
        return self.model
