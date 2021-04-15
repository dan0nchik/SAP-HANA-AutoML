from abc import ABC, abstractmethod

from pipeline.leaderboard import Leaderboard


class BaseOptimizer(ABC):
    """Base optimizer class. Inherit from it to create custom optimizer."""

    def __init__(self):
        self.leaderboard: Leaderboard = Leaderboard()

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

    def print_leaderboard(self):
        print("\033[33m {}".format("Leaderboard (top best algorithms):\n"))
        place = 1
        for member in self.leaderboard.board:
            print(
                "\033[33m {}".format(
                    str(place)
                    + ".  "
                    + str(member.algorithm.model)
                    + "\n Train accuracy: "
                    + str(member.train_accuracy)
                    + "  Validation accuracy: "
                    + str(member.valid_accuracy)
                )
            )
            print("\033[0m {}".format(""))
            place += 1
