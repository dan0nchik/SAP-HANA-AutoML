from pipeline.modelres import ModelBoard
from algorithms.classification.decisiontreecls import DecisionTreeCls


def test_init():
    algo = DecisionTreeCls()
    board = ModelBoard(algo.model, 40)
    assert board.model is not None and board.accuracy is not None
