from automl import AutoML


def main():
    m = AutoML()
    m.fit(file_path='data/train.csv', target='Survived', columns_to_remove=['PassengerId'],
          categorical_features=['Sex', 'Embarked'])
    print(m.optimizer())
    print(m.best_params())
    m.fit(file_path='data/train.csv', target='Survived', columns_to_remove=['PassengerId'],
          categorical_features=['Sex', 'Embarked'], optimizer='OptunaSearch')
    print(m.optimizer())
    print(m.best_params())



if __name__ == "__main__":
    main()
