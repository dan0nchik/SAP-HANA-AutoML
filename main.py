from automl import AutoML


def main():
    m = AutoML()
    m.fit(file_path='data/cleaned_train.csv', target='Survived')
    m.fit(file_path='data/reg.csv', target='Все 18+_TVR')


if __name__ == "__main__":
    main()
