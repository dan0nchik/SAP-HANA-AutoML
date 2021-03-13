from automl import AutoML
import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(config_path="config", config_name="run")
def main(cfg: DictConfig):
    print(cfg)
    config = OmegaConf.create(cfg)
    m = AutoML()
    m.fit(url=config.internet.url)


if __name__ == "__main__":
    main()
