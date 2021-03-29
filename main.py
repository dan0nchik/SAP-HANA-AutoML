from automl import AutoML
import hydra
from omegaconf import DictConfig, OmegaConf


def main():
    m = AutoML()
    m.fit(
        file_path="data/train.csv",
        target="Survived",
        id_column="PassengerId",
        columns_to_remove=["Name", "Cabin"],
        categorical_features=["Survived"],
    )
    """
    m.fit(
        table_name="AUTOML99131b50-a220-427f-afbe-c936deafbfbc",
        file_path="data/reg.csv",
        target="Все 18+_TVR",
        id_column="ID",
        categorical_features=['Канал_ПЕРВЫЙ КАНАЛ', 'Канал_РЕН ТВ', 'Канал_РОССИЯ 1', 'Канал_СТС', 'Канал_ТНТ', 'day',
                              'year', 'month', 'hour', 'holidays']
    )
     """
    print(m.best_params)


if __name__ == "__main__":
    main()
