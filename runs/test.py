import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base=None, config_name="config", config_path="config")
def run(config: DictConfig) -> None:
    cfg = OmegaConf.to_container(config)
    print(cfg)
    with open("cfg.yaml", "w") as f:
        OmegaConf.save(config=cfg, f=f)


if __name__ == "__main__":
    run()
