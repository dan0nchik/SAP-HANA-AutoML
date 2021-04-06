from bayes_opt.bayesian_optimization import BayesianOptimization
import copy
from optimizers.base_optimizer import BaseOptimizer


class BayesianOptimizer(BaseOptimizer):
    """Bayesian hyperparameters optimizer. (https://github.com/fmfn/BayesianOptimization)

    Attributes
    ----------
    data : Data
        Input data.
    algo_list : list
        List of algorithms to be tuned and compared.
    iter : int
        Number of iterations.
    problem : str
        Machine learning problem.
    tuned_params : str
        Final tuned hyperparameters of best algorithm.
    algo_index : int
        Index of algorithm in algorithms list.
    imputerstrategy_list : list
        List of imputer strategies for preprocessing.
    categorical_list : list
        List of categorical features in dataframe.
    inner_data : Data
        Copy of Data object to prevent from preprocessing data object multiple times while optimizing.
    imputer
        Imputer for preprocessing
    model
        Tuned HANA ML model in algorithm.
    """

    def __init__(
        self, algo_list: list, data, iterations, problem, categorical_list=None
    ):
        self.data = data
        self.algo_list = algo_list
        self.iter = iterations
        self.problem = problem
        self.tuned_params = {}
        self.algo_index = 0
        self.imputerstrategy_list = ["mean", "median", "zero"]
        self.categorical_list = categorical_list
        self.inner_data = None
        self.imputer = None
        self.model = None

    def objective(self, algo_index_tuned, preprocess_method):
        """Main objective function. Optimizer uses it to search for best algorithm and preprocess method.

        Parameters
        ----------
        algo_index_tuned : int
            Index of algorithm in algo_list.
        preprocess_method : in
            Strategy to decode categorical variables.
        dropempty : Bool
            Drop empty rows or not.
        categorical_list : list
            List of categorical features.

        Returns
        -------
        Best params
            Optimal hyperparameters.

        """
        self.algo_index = round(algo_index_tuned)
        rounded_preprocess_method = round(preprocess_method)
        self.imputer = self.imputerstrategy_list[rounded_preprocess_method]
        self.inner_data = copy.copy(self.data).clear(
            num_strategy=self.imputer,
            cat_strategy=None,
            dropempty=False,
            categorical_list=None,
        )
        opt = BayesianOptimization(
            f=self.child_objective,
            pbounds={**self.algo_list[self.algo_index].get_params()},
            verbose=False,
            random_state=17,
        )
        opt.maximize(n_iter=1)
        self.algo_list[self.algo_index].set_params(**opt.max["params"])
        return opt.max["target"]

    def child_objective(self, **hyperparameters):
        """Mini objective function. It is used to tune hyperparameters of algorithm that was chosen in main objective.

        Parameters
        ----------
        **hyperparameters
            Parameters of algorithm's model.

        Returns
        -------
        acc
            Accuracy score of a model.

        """
        algorithm = self.algo_list[self.algo_index]
        algorithm.set_params(**hyperparameters)
        ftr: list = self.inner_data.train.columns
        ftr.remove(self.inner_data.target)
        ftr.remove(self.inner_data.id_colm)
        algorithm.model.fit(
            self.inner_data.train,
            key=self.inner_data.id_colm,
            features=ftr,
            label=self.inner_data.target,
            categorical_variable=self.categorical_list,
        )
        acc = algorithm.model.score(
            self.inner_data.valid,
            key=self.inner_data.id_colm,
            label=self.inner_data.target,
        )
        print("Iteration accuracy: " + str(acc))
        self.model = algorithm.model
        return acc

    def get_tuned_params(self):
        """Returns tuned hyperparameters."""

        return {
            "title": self.algo_list[self.algo_index].title,
            "params": self.tuned_params,
        }

    def get_model(self):
        """Returns tuned model."""

        return self.model

    def get_preprocessor_settings(self):
        """Returns tuned preprocessor settings."""

        return {"imputer": self.imputer}

    def tune(self):
        """Starts hyperparameter searching."""

        opt = BayesianOptimization(
            f=self.objective,
            pbounds={
                "algo_index_tuned": (0, len(self.algo_list) - 1),
                "preprocess_method": (0, len(self.imputerstrategy_list) - 1),
            },
            random_state=17,
        )
        opt.maximize(n_iter=self.iter)
        self.tuned_params = opt.max
