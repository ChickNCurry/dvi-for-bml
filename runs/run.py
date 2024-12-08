import os
from typing import Any

import hydra
import torch
import wandb
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from src.train.train_bml import BetterBMLTrainer
from src.train.train_bml_alternating import AlternatingBMLTrainer
from src.utils.load import load_cdvi_for_bml


@hydra.main(version_base=None, config_name="cfg", config_path="config")
def run(cfg: DictConfig) -> None:

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cdvi, optimizer, train_loader, val_loader = load_cdvi_for_bml(
        cfg=cfg, alternating_ratio=cfg.training.alternating_ratio, device=device
    )

    if cfg.wandb.logging:
        wandb.init(
            project=cfg.wandb.project,
            config=OmegaConf.to_container(cfg),  # type: ignore
        )

    if cfg.training.alternating_ratio is None:
        # assert type(train_loader) == DataLoader[Any]

        trainer = BetterBMLTrainer(
            device=device,
            cdvi=cdvi,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            scheduler=None,
            wandb_logging=cfg.wandb.logging,
        )

    else:
        # assert type(train_loader) == tuple[DataLoader[Any], DataLoader[Any]]

        trainer = AlternatingBMLTrainer(
            device=device,
            cdvi=cdvi,
            train_decoder_loader=train_loader[0],
            train_cdvi_loader=train_loader[1],
            val_loader=val_loader,
            optimizer=optimizer,
            wandb_logging=cfg.wandb.logging,
        )

    trainer.train(
        num_epochs=cfg.training.num_epochs,
        max_clip_norm=cfg.training.max_clip_norm,
        alpha=cfg.training.alpha,
        validate=True,
    )

    if cfg.wandb.logging and wandb.run is not None:

        if not os.path.exists("models"):
            os.mkdir("models")

        dir = os.path.join("models", wandb.run.name)
        os.mkdir(dir)

        model_path = os.path.join(dir, "cdvi.pth")
        optim_path = os.path.join(dir, "optim.pth")
        cfg_path = os.path.join(dir, "cfg.yaml")

        torch.save(cdvi.state_dict(), model_path)
        torch.save(optimizer.state_dict(), optim_path)
        with open(cfg_path, "w") as f:
            OmegaConf.save(cfg, f)

        if cfg.wandb.logging and wandb.run is not None:
            wandb.run.log_model(path=model_path, name=f"{wandb.run.name}_cdvi.pth")
            wandb.run.log_model(path=optim_path, name=f"{wandb.run.name}_optim.pth")
            wandb.run.log_model(path=cfg_path, name=f"{wandb.run.name}_cfg.yaml")


if __name__ == "__main__":
    run()
