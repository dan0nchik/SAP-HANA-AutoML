import time

from hana_automl.optimizers.bayes import BayesianOptimizer
from hana_automl.optimizers.optuna_optimizer import OptunaOptimizer
from hana_automl.pipeline.data import Data
from hana_automl.preprocess.preprocessor import Preprocessor

from hana_automl.utils.error import PipelineError


class Pipeline:
    """The 'director' of the whole hyperparameter searching process.

    Attributes
    ----------
    data : Data
        Input data.
    iter : int
        Number of iterations.
    opt
        Optimizer.
    time_limit: int
        In seconds
    verbose
        Level of output.
    """

    def __init__(
        self,
        data: Data,
        steps: int,
        task: str,
        time_limit: int = None,
        verbose=2,
        tuning_metric=None,
    ):
        self.data = data
        self.iter = steps
        self.task = task
        self.time_limit = time_limit
        self.opt = None
        self.verbose = verbose
        self.tuning_metric = tuning_metric

    def train(self, categorical_features: list = None, optimizer: str = None):
        """Preprocesses data and starts optimization.

        Parameters
        ----------
        categorical_features : list
            List of categorical features.
        optimizer : string
            Optimizer for searching for hyperparameters.
            Currently supported: "OptunaSearch" (default), "BayesianOptimizer" (unstable)

        Returns
        -------
        opt
            Optimizer.

        """
        pr = Preprocessor()
        algo_list, self.task, algo_dict = pr.set_task(
            self.data, target=self.data.target, task=self.task
        )
        if self.verbose > 0:
            print("Task:", self.task)
        if self.task == "reg":
            if self.tuning_metric is None:
                self.tuning_metric = "r2_score"
            if self.tuning_metric not in ["r2_score", "mse", "rmse", "mae"]:
                raise PipelineError(f"Wrong {self.task} task metric error")
        if self.task == "cls":
            if self.tuning_metric is None:
                self.tuning_metric = "accuracy"
            if self.tuning_metric not in ["accuracy"]:
                raise PipelineError(f"Wrong {self.task} task metric error")
        if self.verbose > 0:
            print("Tuning metric:", self.tuning_metric)
        if optimizer == "BayesianOptimizer":
            self.opt = BayesianOptimizer(
                algo_list=algo_list,
                data=self.data,
                iterations=self.iter,
                time_limit=self.time_limit,
                categorical_features=categorical_features,
                problem=self.task,
                verbose=self.verbose,
                tuning_metric=self.tuning_metric,
            )
        elif optimizer == "OptunaSearch":
            self.opt = OptunaOptimizer(
                algo_list=algo_list,
                data=self.data,
                problem=self.task,
                iterations=self.iter,
                time_limit=self.time_limit,
                algo_dict=algo_dict,
                categorical_features=categorical_features,
                verbose=self.verbose,
                tuning_metric=self.tuning_metric,
            )
        else:
            raise PipelineError("Optimizer not found!")
        self.opt.tune()
        time.sleep(1)
        return self.opt
