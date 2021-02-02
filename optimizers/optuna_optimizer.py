from preprocess.impsettings import ImputerSettings
import copy
from preprocess.preprocessor import Preprocessor
from optimizers.base_optimizer import BaseOptimizer
from algorithms.classification.decisiontree import DecisionTreeClassifier
from algorithms.classification.logregression import LogisticRegression
from algorithms.regression.ridge import RidgeRegression
import optuna
from sklearn.model_selection import cross_val_score


class OptunaOptimizer(BaseOptimizer):
    def __init__(
        self,
        algo_list,
        data,
        problem,
        iterations,
        algo_names: list,
        categorical_features=None,
        droplist_columns=None,
    ):
        self.algo_list = algo_list
        self.data = data
        self.iterations = iterations
        self.problem = problem
        self.algo_names = algo_names
        self.categorical_features = categorical_features
        self.droplist_columns = droplist_columns

        opt = optuna.create_study(direction="maximize")
        opt.optimize(self.objective, n_trials=iterations)
        self.tuned_params = opt.best_trial

    def objective(self, trial):
        algo = trial.suggest_categorical("algo", self.algo_names)
        encoder_method = "LabelEncoder"
        if self.categorical_features is not None:
            catencoder = trial.suggest_categorical(
                "catencoder", ["LabelEncoder", "OneHotEncoder"]
            )
            pr = Preprocessor()
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
        print(algo)
        if algo == "DecisionTree":
            max_depth = trial.suggest_int("max_depth", 1, 20, log=True)
            max_leaf_nodes = trial.suggest_int("max_leaf_nodes", 2, 100, log=True)
            criterion = trial.suggest_categorical("criterion", ["gini", "entropy"])
            model = DecisionTreeClassifier(
                max_depth=max_depth, max_leaf_nodes=max_leaf_nodes, criterion=criterion
            )
        elif algo == "Logistic Regression":
            tol = trial.suggest_float("tol", 1e-10, 1e10, log=True)
            c = trial.suggest_float("C", 0.1, 10.0, log=True)
            model = LogisticRegression(tol=tol, C=c)
        score = cross_val_score(
            model, data2.X_train, data2.y_train.values.ravel(), n_jobs=-1, cv=3
        )
        accuracy = score.mean()
        return accuracy

    def get_tuned_params(self):
        return self.tuned_params