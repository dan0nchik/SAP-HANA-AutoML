from automl import AutoML
import hydra
from omegaconf import DictConfig, OmegaConf


def main():
    m = AutoML()
    m.fit(file_path="data/train.csv", table_name="AUTOML0ee8453b-03f5-4fe0-a1c6-3c63226899dd")


if __name__ == "__main__":
    main()
