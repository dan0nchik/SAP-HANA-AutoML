import sklearn

from automl import AutoML


def main():
    m = AutoML()

    m.fit(file_path="data/reg.csv", target="Все 18+_TVR", optimizer="OptunaSearch")
    # m.fit(file_path='data/cleaned_train.csv', target='Survived')


if __name__ == "__main__":
    main()
