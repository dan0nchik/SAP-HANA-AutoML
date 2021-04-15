import os

from automl import AutoML
from utils.connection import connection_context
import pytest


@pytest.mark.parametrize("optimizer", ["OptunaSearch", "BayesianOptimizer"])
def test_main(optimizer, tmpdir):
    m = AutoML(connection_context)

    m.fit(
        table_name="test_when_clean",
        file_path="../../data/cleaned_train.csv",
        target="Survived",
        id_column="PassengerId",
        steps=3,
        categorical_features=["Survived"],
        optimizer=optimizer,
    )
    assert m.best_params["accuracy"] > 0.50
    m.predict(
        file_path="../../data/test_cleaned_train.csv",
        table_name="test_when_clean",
        id_column="PassengerId",
    )
    m.save_results_as_csv("res.csv")
    m.save_stats_as_csv("stats.csv")
    m.save_preprocessor("prep.json")
    m.predict(
        file_path="../../data/test_cleaned_train.csv",
        table_name="test_when_clean",
        id_column="PassengerId",
        preprocessor_file="prep.json",
    )
    for i in ["res.csv", "stats.csv", "prep.json"]:
        with open(i) as file:
            assert file.readlines is not None
            os.remove(i)
