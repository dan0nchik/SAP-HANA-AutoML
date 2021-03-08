import pytest
from pipeline.input import Input
import pandas as pd
from pipeline.data import Data
from sklearn.model_selection import train_test_split


def test_url():
    i = Input()
    df = i.load_from_url(
        "https://gist.githubusercontent.com/netj/8836201/raw"
        "/6f9306ad21398ea43cba4f7d537619d0e07d5ae3 "
        "/iris.csv"
    )
    assert type(pd.DataFrame()) == type(df)


def test_from_file():
    i = Input()
    df = i.read_from_file("tests/ads.csv")
    assert type(pd.DataFrame()) == type(df)


# def test_handle_data():
#     i = Input(file_path="tests/ads.csv", target='Survived')
#     assert type(Data) == type(i.handle_data())
