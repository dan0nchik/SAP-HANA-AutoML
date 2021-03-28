from automl import AutoML
import hydra
from omegaconf import DictConfig, OmegaConf


def main():
    m = AutoML()
    m.fit(
        file_path="data/train.csv",
        target="Survived",
        columns_to_remove=["PassengerId"],
        categorical_features=["Sex", "Embarked"],
    )


if __name__ == "__main__":
    main()
