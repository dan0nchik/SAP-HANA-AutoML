from automl import AutoML, Storage
from utils.connection import connection_context


def main():
    m = AutoML()
    m.fit(
        table_name="AUTOML505f62ca-1c99-405b-b9d5-8912920038ec",
        file_path="data/cleaned_train.csv",
        target="Survived",
        id_column="PassengerId",
        columns_to_remove=[],
        categorical_features=["Survived"],
        steps=1,
        # optimizer="BayesianOptimizer",
    )

    # m.fit(
    #     table_name="AUTOML505f62ca-1c99-405b-b9d5-8912920038ec",
    #     file_path="data/reg.csv",
    #     target="Все 18+_TVR",
    #     id_column="ID",
    #     categorical_features=[
    #         "Канал_ПЕРВЫЙ КАНАЛ",
    #         "Канал_РЕН ТВ",
    #         "Канал_РОССИЯ 1",
    #         "Канал_СТС",
    #         "Канал_ТНТ",
    #         "day",
    #         "year",
    #         "month",
    #         "hour",
    #         "holidays",
    #     ],
    #     optimizer="OptunaSearch",
    # )
    print("Model: ", m.get_model())
    print(m.best_params)
    m.predict(
        table_name="AUTOML08df3e64-b749-489b-8f75-84eeee340342",
        file_path="data/test_cleaned_train.csv",
        id_column="PassengerId",
    )
    m.save_results_as_csv("results.csv")
    m.save_preprocessor("prep.json")
    m.predict(
        table_name="AUTOML08df3e64-b749-489b-8f75-84eeee340342",
        file_path="data/test_cleaned_train.csv",
        id_column="PassengerId",
        preprocessor_file="prep.json",
    )


if __name__ == "__main__":
    main()
