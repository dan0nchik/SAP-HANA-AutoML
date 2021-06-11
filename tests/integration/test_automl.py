import pandas as pd
import pytest
from hana_automl.automl import AutoML
from hana_automl.storage import Storage
from ..connection import connection_context, schema
from benchmarks.cleanup import clean

clean(connection_context, schema=schema)

m = AutoML(connection_context)
storage = Storage(connection_context, schema)
verbose = 0


@pytest.mark.parametrize("optimizer", ["OptunaSearch", "BayesianOptimizer"])
def test_regression(optimizer):
    m.fit(
        table_name="TESTING_REG",
        file_path="data/boston_data.csv",
        target="medv",
        id_column="ID",
        steps=5,
        optimizer=optimizer,
        output_leaderboard=True,
        task="reg",
        verbose=verbose,
    )
    assert m.best_params["accuracy"] > 0.50
    print(
        "SCOOOOOOOORE",
        m.score(df=pd.read_csv("data/boston_data.csv"), target="medv", id_column="ID"),
    )
    m.model.name = "TESTING_MODEL_REG"
    storage.save_model(m)
    storage.save_leaderboard(m.leaderboard, "TESTING_LEADERBOARD_REG")
    new = storage.load_model("TESTING_MODEL_REG", version=1)
    assert new.predict(file_path="./data/boston_test_data.csv").empty is False
    # assert new.score(df=pd.read_csv("data/boston_data.csv"), target='medv', id_column="ID") > 0.50
    assert storage.list_preprocessors("TESTING_MODEL_REG").empty is False
    assert storage.list_leaderboards().empty is False
    storage.delete_model("TESTING_MODEL_REG", version=1)
    storage.delete_leaderboard("TESTING_LEADERBOARD_REG")


@pytest.mark.parametrize("optimizer", ["OptunaSearch", "BayesianOptimizer"])
def test_classification(optimizer):
    m.fit(
        table_name="TESTING_CLS",
        file_path="data/cleaned_train.csv",
        target="Survived",
        id_column="PASSENGERID",
        categorical_features=["Survived"],
        steps=5,
        optimizer=optimizer,
        task="cls",
        output_leaderboard=True,
        verbose=verbose,
    )
    assert m.best_params["accuracy"] > 0.50
    print(
        "SCOOOOOOOOORE",
        m.score(
            df=pd.read_csv("data/cleaned_train.csv"),
            target="Survived",
            id_column="PASSENGERID",
        ),
    )
    m.model.name = "TESTING_MODEL_CLS"
    storage.save_model(m)
    storage.save_leaderboard(m.leaderboard, "TESTING_LEADERBOARD_CLS")
    new = storage.load_model("TESTING_MODEL_CLS", version=1)
    assert (
        new.predict(
            file_path="./data/test_cleaned_train.csv", id_column="PassengerId"
        ).empty
        is False
    )
    assert (
        new.score(
            df=pd.read_csv("data/cleaned_train.csv")[:100],
            target="Survived",
            id_column="PASSENGERID",
        )
        > 0.50
    )
    assert storage.list_preprocessors("TESTING_MODEL_CLS").empty is False
    assert storage.list_leaderboards().empty is False
    storage.delete_model("TESTING_MODEL_CLS", version=1)
    storage.delete_leaderboard("TESTING_LEADERBOARD_CLS")


@pytest.mark.parametrize("task", ["cls", "reg"])
def test_ensembles(task):
    if task == "reg":
        m.fit(
            table_name="TESTING_REG",
            file_path="data/boston_data.csv",
            target="medv",
            id_column="ID",
            steps=5,
            ensemble=True,
            output_leaderboard=True,
            task="reg",
            verbose=verbose,
        )
        assert m.best_params["accuracy"] > 0.50
        m.model.name = "ENSEMBLE_REG"
        storage.save_model(m)
        new = storage.load_model("ENSEMBLE_REG", version=1)
        assert new.predict(file_path="./data/boston_test_data.csv").empty is False
        assert storage.list_preprocessors("ENSEMBLE_REG").empty is False
        storage.delete_model("ENSEMBLE_REG", version=1)
        assert storage.list_models("ENSEMBLE_REG").empty is True
    else:
        m.fit(
            table_name="TESTING_CLS",
            file_path="data/cleaned_train.csv",
            target="Survived",
            id_column="PASSENGERID",
            categorical_features=["Survived"],
            steps=5,
            ensemble=True,
            task="cls",
            output_leaderboard=True,
            verbose=verbose,
        )
        assert m.best_params["accuracy"] > 0.50
        m.model.name = "ENSEMBLE_CLS"
        storage.save_model(m)
        new = storage.load_model("ENSEMBLE_CLS", version=1)
        assert (
            new.score(
                df=pd.read_csv("data/cleaned_train.csv")[:100],
                target="Survived",
                id_column="PASSENGERID",
            )
            > 0.50
        )
        assert (
            new.predict(
                file_path="./data/test_cleaned_train.csv", id_column="PassengerId"
            ).empty
            is False
        )
        assert storage.list_preprocessors("ENSEMBLE_CLS").empty is False
        storage.delete_model("ENSEMBLE_CLS", version=1)
        assert storage.list_models("ENSEMBLE_CLS").empty is True
