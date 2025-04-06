import hydra
import torch
import wandb
from omegaconf import DictConfig, OmegaConf

from dviforbml.utils.load_dvi import load_dvi
from dviforbml.utils.helper import get_name_dvi, upload_run_dvi


@hydra.main(version_base=None, config_name="cfg", config_path="config")
def run(cfg: DictConfig) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, trainer, dataloader = load_dvi(cfg=cfg, device=device)

    name = get_name_dvi(cfg)

    if cfg.wandb.logging:
        wandb.init(
            name=name,
            project=cfg.wandb.project,
            config=OmegaConf.to_container(cfg),
        )

    trainer.train(
        num_epochs=cfg.training.num_epochs,
        max_clip_norm=cfg.training.max_clip_norm,
        alpha=cfg.training.alpha,
        validate=False,
    )

    if cfg.wandb.logging and wandb.run is not None:
        upload_run_dvi(cfg, model, trainer, device, dataloader)


if __name__ == "__main__":
    run()
