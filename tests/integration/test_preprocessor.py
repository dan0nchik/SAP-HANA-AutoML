from hana_automl.pipeline.input import Input
from hana_automl.preprocess.preprocessor import Preprocessor
import pandas as pd
from ..connection import connection_context
import pytest
import time


@pytest.mark.parametrize("name", ["cls", "reg"])
def test_tasks(name):
    from benchmarks.cleanup import clean

    clean(connection_context)
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
    data = inputted.split_data()
    pr = Preprocessor()

    if name == "cls":
        start_time = time.time()
        print(data.train.shape)
        data.train = pr.drop_outers(
            data.train,
            "PASSENGERID",
            "Survived",
            ["Name", "Sex", "Ticket", "Cabin", "Embarked"],
        )
        print(data.train.shape)
        print(time.time() - start_time)
        assert pr.set_task(data, "Survived", task="cls")[1] == name
        start_time = time.time()
        data.train = pr.autoimput(data.train, "Survived", "PASSENGERID")
        print(time.time() - start_time)
        start_time = time.time()
        data.train = pr.removecolumns(["Cabin"], data.train)
        print(time.time() - start_time)
        start_time = time.time()
        data.train = pr.normalize(data.train, "min-max", "PASSENGERID", "Survived")
        print(time.time() - start_time)

    if name == "reg":
        assert pr.set_task(data, "Все 18+_TVR", task="reg")[1] == name
        start_time = time.time()
        print(data.train.shape)
        data.train = pr.drop_outers(data.train, "ID", "Все 18+_TVR", [])
        print(data.train.shape)
        print(time.time() - start_time)
        start_time = time.time()
        data.train = pr.autoimput(data.train, "Все 18+_TVR", "ID")
        print(time.time() - start_time)
        start_time = time.time()
        data.train = pr.removecolumns(["Канал_ПЕРВЫЙ КАНАЛ"], data.train)
        print(time.time() - start_time)
        start_time = time.time()
        data.train = pr.normalize(data.train, "min-max", "ID", "Все 18+_TVR")
        print(time.time() - start_time)
