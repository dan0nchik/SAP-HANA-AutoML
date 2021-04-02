from automl import AutoML
import hydra
from omegaconf import DictConfig, OmegaConf
from hdbcli import dbapi


def main():
    m = AutoML()
    m.fit(
        # table_name="AUTOML9bc6ad7a-427e-4782-9190-de2c6b1db96d",
        file_path="data/cleaned_train.csv",
        target="Survived",
        id_column="PassengerId",
        columns_to_remove=[],
        categorical_features=["Survived"],
        steps=1,
        optimizer="OptunaSearch",
    )
    print("Model: ", m.get_model())
    print(m.best_params)
    m.predict(
        table_name="AUTOML08df3e64-b749-489b-8f75-84eeee340342",
        file_path="data/test_cleaned_train.csv",
        id_column="PassengerId",
    )
    # m.fit(
    #     table_name="AUTOML3c13e97d-630b-4620-8f12-7d0ff8601069",
    #     target="Все 18+_TVR",
    #     id_column="ID",
    #     categorical_features=['Канал_ПЕРВЫЙ КАНАЛ', 'Канал_РЕН ТВ', 'Канал_РОССИЯ 1', 'Канал_СТС', 'Канал_ТНТ', 'day',
    #                           'year', 'month', 'hour', 'holidays']
    # )


if __name__ == "__main__":
    main()
