import sklearn

from automl import AutoML


def main():
    m = AutoML()
    print(sklearn.__version__)
    m.fit(
        file_path="data/train.csv",
        target="Survived",
        columns_to_remove=["PassengerId"],
        categorical_features=["Sex", "Embarked"],
        optimizer='OptunaSearch'
    )
    # m.fit(file_path="data/reg.csv", target="Все 18+_TVR")
    # m.fit(file_path='data/cleaned_train.csv', target='Survived')


if __name__ == "__main__":
    main()
