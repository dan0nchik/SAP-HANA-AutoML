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

        opt = optuna.create_study(direction="maximize")
        opt.optimize(self.objective, n_trials=iterations)
        self.tuned_params = opt.best_params

    def objective(self, trial):
        algo = self.algo_dict.get(trial.suggest_categorical("algo", self.algo_dict.keys()))
        encoder_method = "LabelEncoder"
        pr = Preprocessor()
        if self.categorical_features is not None:
            catencoder = trial.suggest_categorical(
                "catencoder", ["LabelEncoder", "OneHotEncoder"]
            )
            if catencoder == "LabelEncoder":
                encoder_method = "LabelEncoder"
            elif catencoder == "OneHotEncoder":
                encoder_method = "OneHotEncoder"
        numimputer = trial.suggest_categorical(
            "numimputer", ["mean", "most_frequent", "median"]
        )
        stringimputer = trial.suggest_categorical(
            "stringimputer", ["most_frequent", "constant"]
        )
        boolimputer = trial.suggest_categorical("boolimputer", ["most_frequent"])
        data2 = pr.clean(
            data=copy.deepcopy(self.data),
            categorical_list=self.categorical_features,
            encoder_method=encoder_method,
            droplist_columns=self.droplist_columns,
            numimpset=ImputerSettings(strategy=numimputer),
            stringimpset=ImputerSettings(basicvars="string", strategy=stringimputer),
            boolimpset=ImputerSettings(basicvars="bool", strategy=boolimputer),
        )
        model = algo.optunatune(trial)
        score = cross_val_score(
            model, data2.X_train, data2.y_train.values.ravel(), n_jobs=-1, cv=3
        )
        accuracy = score.mean()
        return accuracy

    def get_tuned_params(self):
        print(
            "Title: ",
            self.tuned_params.pop("algo"),
            "\nInfo:",
            self.tuned_params
        )
