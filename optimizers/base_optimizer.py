from abc import ABC, abstractmethod


class BaseOptimizer(ABC):
    @abstractmethod
    def objective(self):
        """Implement the objective function here.
        It must take hyperparameters to be tuned in optimizer"""
        pass

    @abstractmethod
    def get_tuned_params(self):
        """Return hyperparameters that your optimizer has tuned"""
        pass

    @abstractmethod
    def tune(self):
        """Tune hyperparameters here"""
        pass

    @abstractmethod
    def get_model(self):
        """Return tuned HANA PAL model"""

    @abstractmethod
    def get_preprocessor_settings(self):
        """Return a dictionary with preprocessor settings"""
