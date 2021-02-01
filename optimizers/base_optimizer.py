from bayes_opt.bayesian_optimization import BayesianOptimization
from pipeline.validator import Validate
from pipeline.fit import Fit
from preprocess.preprocessor import Preprocessor
from algorithms import base_algo
from pipeline.data import Data
import pandas as pd
import numpy as np
import copy
from abc import ABC, abstractmethod


class BaseOptimizer(ABC):
    @abstractmethod
    def objective():
        """Implement the objective function here.
        It must take hyperparameters to be tuned in optimizer"""
        pass

    @abstractmethod
    def get_tuned_params():
        """Return hyperparameters that your optimizer has tuned"""
        pass
