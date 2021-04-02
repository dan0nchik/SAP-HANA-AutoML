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

    @abstractmethod
    def tune():
        """Tune hyperparameters here"""
        pass

    @abstractmethod
    def get_model():
        """Return tuned HANA PAL model"""
