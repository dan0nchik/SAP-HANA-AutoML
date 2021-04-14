from pipeline.input import Input
from preprocess.preprocessor import Preprocessor
import pandas as pd
from utils.connection import connection_context
import pytest


@pytest.mark.parametrize("name", ["cls", "reg"])
def test_tasks(name):
    multiple_tasks(name)


def multiple_tasks(name):
    if name == "cls":
        df = pd.read_csv("../../data/train.csv")
    if name == "reg":
        df = pd.read_csv("../../data/reg.csv")

    inputted = Input(connection_context, df, table_name="test")
    inputted.load_data()
    data = inputted.split_data()
    pr = Preprocessor()

    for i in ["zero", "mean", "median"]:
        print("IMPUTER", i)
        auto_imput(i, data, pr)

    if name == "cls":
        assert pr.set_task(data, "Survived")[1] == name
    if name == "reg":
        assert pr.set_task(data, "Все 18+_TVR")[1] == name


def auto_imput(imputer, data, pr):
    data.train = pr.autoimput(data.train, imputer, None)
    assert data.train.hasna() is False
