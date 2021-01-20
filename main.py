from automl import AutoML


def main():
    m = AutoML()
    m.fit(file_path='data/train.csv', target='Survived', columns_to_remove=['PassengerId'], categorical_features=[
    'Sex', 'Embarked'], optimizer='GridSearch')
    m.fit(file_path='data/reg.csv', target='Все 18+_TVR')


if __name__ == "__main__":
    main()
