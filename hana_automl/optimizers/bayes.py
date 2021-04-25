from bayes_opt.bayesian_optimization import BayesianOptimization

from hana_automl.optimizers.base_optimizer import BaseOptimizer
from hana_automl.pipeline.leaderboard import Leaderboard

from hana_automl.pipeline.modelres import ModelBoard


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
    categorical_features : list
        List of categorical features in dataframe.
    inner_data : Data
        Copy of Data object to prevent from preprocessing data object multiple times while optimizing.
    imputer
        Imputer for preprocessing
    model
        Tuned HANA ML model in algorithm.
    """

    def __init__(
        self, algo_list: list, data, iterations, problem, categorical_features=None
    ):
        self.data = data
        self.algo_list = algo_list
        self.iter = iterations
        self.problem = problem
        self.tuned_params = {}
        self.algo_index = 0
        self.imputerstrategy_list = ["mean", "median", "zero"]
        self.categorical_features = categorical_features
        self.inner_data = None
        self.imputer = None
        self.model = None
        self.leaderboard: Leaderboard = Leaderboard()

    def objective(self, algo_index_tuned, preprocess_method):
        """Main objective function. Optimizer uses it to search for best algorithm and preprocess method.

        Parameters
        ----------
        algo_index_tuned : int
            Index of algorithm in algo_list.
        preprocess_method : in
            Strategy to decode categorical variables.

        Returns
        -------
        Best params
            Optimal hyperparameters.

        Note
        ----
            Total number of child objective iterations is n_iter + init_points!
        """
        self.algo_index = round(algo_index_tuned)
        rounded_preprocess_method = round(preprocess_method)
        imputer = self.imputerstrategy_list[rounded_preprocess_method]
        self.inner_data = self.data.clear(
            num_strategy=imputer,
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
        opt.maximize(n_iter=1, init_points=1)

        algo = self.algo_list[self.algo_index]
        algo.set_params(**opt.max["params"])
        self.fit(algo, self.inner_data)

        self.leaderboard.addmodel(
            ModelBoard(algo, opt.max["target"], {"imputer": imputer})
        )

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
        self.fit(algorithm, self.inner_data)
        acc = algorithm.score(self.inner_data, self.inner_data.test)
        print("Child Iteration accuracy: " + str(acc))
        return acc

    def get_tuned_params(self):
        """Returns tuned hyperparameters."""

        return {
            "title": self.algo_list[self.algo_index].title,
            "accuracy": self.leaderboard.board[0].valid_accuracy,
            "info": self.tuned_params,
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
        opt.maximize(n_iter=self.iter, init_points=1)
        self.tuned_params = opt.max
        self.imputer = self.imputerstrategy_list[
            round(opt.max["params"]["preprocess_method"])
        ]

        for member in self.leaderboard.board:
            data = self.data.clear(
                num_strategy=member.preprocessor["imputer"],
                cat_strategy=None,
                dropempty=False,
                categorical_list=None,
            )
            acc = member.algorithm.score(data=data, df=data.valid)
            member.add_valid_acc(acc)

        self.leaderboard.board.sort(
            key=lambda member: member.valid_accuracy + member.train_accuracy,
            reverse=True,
        )
        self.model = self.leaderboard.board[0].algorithm.model

    def fit(self, algo, data):
        """Fits given model from data. Small method to reduce code repeating."""
        ftr: list = data.train.columns
        ftr.remove(data.target)
        ftr.remove(data.id_colm)
        algo.fit(data, ftr, self.categorical_features)
