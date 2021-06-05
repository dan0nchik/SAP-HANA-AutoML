import pytest
from hana_automl.automl import AutoML
from ..connection import connection_context

m = AutoML(connection_context)


@pytest.mark.parametrize("optimizer", ["OptunaSearch", "BayesianOptimizer"])
def test_regression(optimizer):
    m.fit(
        file_path="data/boston_data.csv",
        target="medv",
        id_column="ID",
        steps=5,
        optimizer=optimizer,
        output_leaderboard=True,
        task="reg",
    )
    assert m.best_params["accuracy"] > 0.50


@pytest.mark.parametrize("optimizer", ["OptunaSearch", "BayesianOptimizer"])
def test_classification(optimizer):
    m.fit(
        file_path="data/cleaned_train.csv",
        target="Survived",
        id_column="PassengerId",
        categorical_features=["Survived"],
        steps=5,
        optimizer=optimizer,
        task="cls",
        output_leaderboard=True,
    )
    assert m.best_params["accuracy"] > 0.50
