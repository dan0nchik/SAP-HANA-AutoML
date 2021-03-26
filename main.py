from automl import AutoML
import hydra
from omegaconf import DictConfig, OmegaConf


def main():
    m = AutoML()
    m.fit(file_path="data/train.csv")


if __name__ == "__main__":
    main()
