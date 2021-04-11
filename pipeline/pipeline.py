import time

from optimizers.bayes import BayesianOptimizer
from optimizers.optuna_optimizer import OptunaOptimizer
from pipeline.data import Data
from preprocess.preprocessor import Preprocessor

from utils.error import PipelineError


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
    """

    def __init__(self, data: Data, steps):
        self.data = data
        self.iter = steps
        self.opt = None

    def train(self, categorical_features=None, optimizer=None):
        """Preprocesses data and starts optimizer.

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
        algo_list, task, algo_dict = pr.set_task(self.data, target=self.data.target)
        print("Task:", task)
        if optimizer == "BayesianOptimizer":
            self.opt = BayesianOptimizer(
                algo_list=algo_list,
                data=self.data,
                iterations=self.iter,
                categorical_list=categorical_features,
                problem=task,
            )
        elif optimizer == "OptunaSearch":
            self.opt = OptunaOptimizer(
                algo_list=algo_list,
                data=self.data,
                problem=task,
                iterations=self.iter,
                algo_dict=algo_dict,
                categorical_features=categorical_features,
            )
        else:
            raise PipelineError("Optimizer not found!")
        self.opt.tune()
        time.sleep(1)
        return self.opt
