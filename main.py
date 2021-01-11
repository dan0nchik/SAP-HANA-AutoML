from models import Model


def main():
    m = Model()
    m.automl(file_path='data/cleaned_train.csv', target='Survived')
    m.automl(file_path='data/reg.csv', target='Все 18+_TVR')


if __name__ == "__main__":
    main()
