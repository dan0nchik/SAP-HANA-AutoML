from hana_automl.pipeline.input import Input
from hana_automl.preprocess.preprocessor import Preprocessor
import pandas as pd
from ..connection import connection_context
import pytest


@pytest.mark.parametrize("name", ["cls", "reg"])
def test_tasks(name):
    multiple_tasks(name)


def multiple_tasks(name):
    if name == "cls":
        df = pd.read_csv("data/train.csv")
        id_col = "PassengerId"
    if name == "reg":
        df = pd.read_csv("data/reg.csv")
        id_col = "ID"

    inputted = Input(connection_context, df, table_name="test", id_col=id_col)
    inputted.load_data()
    data = inputted.split_data(cat_list=None, perform_drop=False)
    pr = Preprocessor()

    for i in ["zero", "mean", "median"]:
        print("IMPUTER", i)
        auto_imput(i, data, pr)

    if name == "cls":
        assert pr.set_task(data, "Survived", task="cls")[1] == name
    if name == "reg":
        assert pr.set_task(data, "Все 18+_TVR", task="reg")[1] == name


def auto_imput(imputer, data, pr):
    data.train = pr.autoimput(data.train, imputer, None)
    assert data.train.hasna() is False
