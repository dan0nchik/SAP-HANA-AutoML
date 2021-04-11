from pipeline.modelres import ModelBoard


class Leaderboard:
    def __init__(self):
        self.board = list()

    def addmodel(self, model):
        self.board.append(model)
        self.board.sort(key=self.accSort, reverse=True)

    def accSort(self, model: ModelBoard):
        return model.accuracy
