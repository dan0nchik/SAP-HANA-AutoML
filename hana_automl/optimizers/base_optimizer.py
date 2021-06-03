from abc import ABC, abstractmethod


class BaseOptimizer(ABC):
    """Base optimizer class. Inherit from it to create custom optimizers."""

    def __init__(self):
        self.leaderboard = list()

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
    def get_algorithm(self):
        """Return tuned AutoML algorithm"""

    @abstractmethod
    def get_preprocessor_settings(self):
        """Return a :meth:`PreprocessorSettings` object with preprocessor settings"""

    def print_leaderboard(self, metric):
        print("\033[33m {}".format("Leaderboard (top best algorithms):\n"))
        place = 1
        for member in self.leaderboard:
            print(
                "\033[33m {}".format(
                    str(place)
                    + ".  "
                    + str(member.algorithm.model)
                    + f"\n Test {metric} score: "
                    + str(member.train_score)
                    + f"  Holdout {metric} score: "
                    + str(member.valid_score)
                )
            )
            print("\033[0m {}".format(""))
            place += 1
