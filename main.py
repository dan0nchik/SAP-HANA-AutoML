from automl import AutoML


def main():
    m = AutoML()
    m.fit(file_path='data/train.csv', target='Survived', colmnsforremv=['PassengerId'], categorical=['Sex', 'Embarked'])
    # m.fit(file_path='data/reg.csv', target='Все 18+_TVR')


if __name__ == "__main__":
    main()
