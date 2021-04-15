import sys, os
import pytest
from automl import AutoML
from utils.connection import connection_context

m = AutoML(connection_context)


@pytest.mark.parametrize("optimizer", ["OptunaSearch", "BayesianOptimizer"])
def test_regression(optimizer):
    m.fit(
        table_name="test_reg",
        file_path="../../data/reg.csv",
        target="Все 18+_TVR",
        id_column="ID",
        categorical_features=[
            "Канал_ПЕРВЫЙ КАНАЛ",
            "Канал_РЕН ТВ",
            "Канал_РОССИЯ 1",
            "Канал_СТС",
            "Канал_ТНТ",
            "day",
            "year",
            "month",
            "hour",
            "holidays",
        ],
        steps=10,
        optimizer=optimizer,
        output_leaderboard=True
    )
    assert m.best_params["accuracy"] > 0.70


@pytest.mark.parametrize("optimizer", ["OptunaSearch", "BayesianOptimizer"])
def test_classification(optimizer):
    m.fit(
        table_name="test_cls",
        file_path="../../data/train.csv",
        target="Survived",
        id_column="PassengerId",
        categorical_features=["Survived", "Sex"],
        columns_to_remove=["Name", "Ticket", "Cabin", "Embarked"],
        steps=10,
        output_leaderboard=True,
        optimizer=optimizer,
    )
    assert m.best_params["accuracy"] > 0.70
