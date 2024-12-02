import os

import hydra
import torch
import wandb
from omegaconf import DictConfig, OmegaConf

from src.components.cdvi import load_cdvi_for_bml
from src.utils.train import BetterBMLTrainer


@hydra.main(version_base=None, config_name="config", config_path="config")
def run(cfg: DictConfig) -> None:

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cdvi, dataloader, optimizer = load_cdvi_for_bml(cfg, device)

    if cfg.wandb.logging:
        wandb.init(
            project=cfg.wandb.project,
            config=OmegaConf.to_container(cfg),  # type: ignore
        )

    trainer = BetterBMLTrainer(
        device=device,
        cdvi=cdvi,
        dataloader=dataloader,
        optimizer=optimizer,
        scheduler=None,
        wandb_logging=cfg.wandb.logging,
    )

    trainer.train(
        num_epochs=cfg.training.num_epochs,
        max_clip_norm=cfg.training.max_clip_norm,
        alpha=cfg.training.alpha,
    )

    if cfg.wandb.logging and wandb.run is not None:

        if not os.path.exists("models"):
            os.mkdir("models")

        dir = os.path.join("models", wandb.run.name)
        os.mkdir(dir)

        model_path = os.path.join(dir, "cdvi.pth")
        optim_path = os.path.join(dir, "optim.pth")
        cfg_path = os.path.join(dir, "config.yaml")

        torch.save(cdvi.state_dict(), model_path)
        torch.save(optimizer.state_dict(), optim_path)
        with open(cfg_path, "w") as f:
            OmegaConf.save(cfg, f)

        if cfg.wandb.logging and wandb.run is not None:
            wandb.run.log_model(path=model_path, name=f"{wandb.run.name}_cdvi.pth")
            wandb.run.log_model(path=optim_path, name=f"{wandb.run.name}_optim.pth")
            wandb.run.log_artifact(path=cfg_path, name=f"{wandb.run.name}_cfg.yaml")


if __name__ == "__main__":
    run()
