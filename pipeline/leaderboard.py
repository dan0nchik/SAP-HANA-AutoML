from pipeline.modelres import ModelBoard


class Leaderboard:
    def __init__(self):
        self.board = list()

    def addmodel(self, model):
        self.board.append(model)

    def accSort(self, model: ModelBoard):
        return model.valid_accuracy
