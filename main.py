from automl import AutoML


def main():

    m = AutoML()
    m.fit(file_path="data/reg.csv")


if __name__ == "__main__":
    main()
