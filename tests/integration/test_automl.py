import pytest
from automl import AutoML
from utils.connection import connection_context

m = AutoML(connection_context)


@pytest.mark.parametrize("optimizer", ["OptunaSearch", "BayesianOptimizer"])
def test_regression(optimizer):
    m.fit(
        file_path="../../data/boston_data.csv",
        target="medv",
        id_column="ID",
        steps=30,
        optimizer=optimizer,
        output_leaderboard=True,
    )
    assert m.best_params["accuracy"] > 0.50


@pytest.mark.parametrize("optimizer", ["OptunaSearch", "BayesianOptimizer"])
def test_classification(optimizer):
    m.fit(
        file_path="../../data/bank.csv",
        target="y",
        id_column="ID",
        categorical_features=["job", "marital", 'education', 'default', 'housing', 'loan', 'contact', 'month', 'y'],
        columns_to_remove=["poutcome"],
        steps=30,
        output_leaderboard=True,
        optimizer=optimizer,
    )
    assert m.best_params["accuracy"] > 0.50
