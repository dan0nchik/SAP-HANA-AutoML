from automl import AutoML


def main():

    m = AutoML()
    m.fit(
        file_path="data/train.csv",
        target="Survived",
        columns_to_remove=["PassengerId"],
        categorical_features=["Sex", "Embarked"],
    )
    print(m.best_params)


if __name__ == "__main__":
    main()
