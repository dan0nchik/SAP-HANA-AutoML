import hana_ml


class Leaderboard:
    """This class allows you to see best models after hyperparameter tuning and evaluation."""

    def __init__(self):
        self.board = list()

    def addmodel(self, model: hana_ml.algorithms.pal.pal_base):
        """Adds new model to model list."""
        self.board.append(model)
