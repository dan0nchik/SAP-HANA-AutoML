from automl import AutoML
import hydra
from omegaconf import DictConfig, OmegaConf


def main():
    m = AutoML()
    '''
    m.fit(
        file_path="data/train.csv",
        table_name="AUTOML0ee8453b-03f5-4fe0-a1c6-3c63226899dd",
        target="Survived",
        id_column="PassengerId",
        columns_to_remove=["Name", "Cabin"],
        categorical_features=["Survived"],
        optimizer="OptunaSearch"
    )
    '''
    m.fit(
        file_path="data/reg.csv",
        target="Все 18+_TVR",
        id_column="ID",
        categorical_features=['Канал_ПЕРВЫЙ КАНАЛ', 'Канал_РЕН ТВ', 'Канал_РОССИЯ 1', 'Канал_СТС', 'Канал_ТНТ', 'day',
                              'year', 'month', 'hour', 'holidays'],
        optimizer="OptunaSearch"
    )
    print(m.best_params)


if __name__ == "__main__":
    main()
