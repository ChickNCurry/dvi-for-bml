import hydra
import torch
import wandb
from omegaconf import DictConfig, OmegaConf

from dviforbml.utils.helper import get_name_dvinp, upload_run_np
from dviforbml.utils.load_dvinp import load_dvinp


@hydra.main(version_base=None, config_name="cfg", config_path="config")
def run(cfg: DictConfig) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, trainer, _, val_loader = load_dvinp(cfg=cfg, device=device)

    if cfg.wandb.logging:
        wandb.init(
            name=get_name_dvinp(cfg),
            project=cfg.wandb.project,
            config=OmegaConf.to_container(cfg),
        )

    trainer.train(
        num_epochs=cfg.training.num_epochs,
        max_clip_norm=cfg.training.max_clip_norm,
        alpha=cfg.training.alpha,
        validate=True,
    )

    if cfg.wandb.logging and wandb.run is not None:
        upload_run_np(cfg, model, trainer, val_loader, device)


if __name__ == "__main__":
    run()
