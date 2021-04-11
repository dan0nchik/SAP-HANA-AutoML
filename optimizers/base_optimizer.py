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
        print("\033[33m {}".format("Leaderboard:\n"))
        num = len(self.leaderboard.board)
        if num > 10:
            num = 10
        for i in range(num):
            print(
                "\033[33m {}".format(
                    str(i + 1)
                    + ".  "
                    + str(self.leaderboard.board[i].model)
                    + "\n Accuracy: "
                    + str(self.leaderboard.board[i].accuracy)
                )
            )
            print("\033[0m {}".format(""))
