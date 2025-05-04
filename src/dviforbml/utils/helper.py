import os
from pathlib import Path
from typing import Tuple

import torch
import wandb
from omegaconf import DictConfig, OmegaConf

from dviforbml.architectures.dvi import DVI
from dviforbml.architectures.np import NP
from dviforbml.evaluation.num_eval_np import num_eval_np
from dviforbml.evaluation.visualization.visualize_dvi_contour import (
    visualize_dvi_2d_contour_all,
)
from dviforbml.training.abstract_trainer import AbstractTrainer


def get_name_np(cfg: DictConfig) -> str:
    model_keys = [
        "model_variant",
        "context_variant",
        "self_attn_num_heads",
        "z_dim",
        "max_context_size",
    ]
    training_keys = ["trainer_variant", "seed"]

    model_values = [f"{v}" for k, v in cfg.model.items() if k in model_keys]
    training_values = [f"{v}" for k, v in cfg.training.items() if k in training_keys]

    return "-".join(model_values + training_values)


def get_name_dvinp(cfg: DictConfig) -> str:
    model_keys = [
        "num_steps",
        "model_variant",
        "context_variant",
        "noise_variant",
        "self_attn_num_heads",
        "contextual_schedules",
        "max_context_size",
    ]
    training_keys = ["trainer_variant", "seed"]

    model_values = [f"{v}" for k, v in cfg.model.items() if k in model_keys]
    training_values = [f"{v}" for k, v in cfg.training.items() if k in training_keys]

    return "-".join(model_values + training_values)


def get_name_dvi(cfg: DictConfig) -> str:
    model_keys = [
        "num_steps",
        "model_variant",
        "context_variant",
        "noise_variant",
        "self_attn_num_heads",
        "contextual_schedules",
        "max_context_size",
    ]

    model_values = [f"{v}" for k, v in cfg.model.items() if k in model_keys]

    return "-".join(model_values)


def upload_run_np(
    cfg: DictConfig,
    model: NP,
    trainer: AbstractTrainer,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> None:
    assert wandb.run is not None

    if not os.path.exists("models"):
        os.mkdir("models")

    dir = os.path.join("models", wandb.run.name)
    os.mkdir(dir)

    model_path = os.path.join(dir, "model.pth")
    decoder_path = os.path.join(dir, "decoder.pth")
    optim_path = os.path.join(dir, "optim.pth")
    cfg_path = os.path.join(dir, "cfg.yaml")
    metrics_path = os.path.join(dir, "metrics.csv")

    torch.save(model.state_dict(), model_path)
    torch.save(model.decoder.state_dict(), decoder_path)
    torch.save(trainer.optimizer.state_dict(), optim_path)

    with open(cfg_path, "w") as f:
        OmegaConf.save(cfg, f)

    num_eval_np(model, val_loader, device, 1024, metrics_path)

    wandb.run.log_model(path=model_path, name=f"{wandb.run.name}_model.pth")
    wandb.run.log_model(path=decoder_path, name=f"{wandb.run.name}_decoder.pth")
    wandb.run.log_model(path=optim_path, name=f"{wandb.run.name}_optim.pth")
    wandb.run.log_model(path=cfg_path, name=f"{wandb.run.name}_cfg.yaml")
    wandb.run.log_model(path=metrics_path, name=f"{wandb.run.name}_metrics.csv")


def upload_run_dvi(
    cfg: DictConfig,
    model: DVI,
    trainer: AbstractTrainer,
    device: torch.device,
    dataloader: torch.utils.data.DataLoader,
) -> None:
    assert wandb.run is not None

    if not os.path.exists("models"):
        os.mkdir("models")

    dir = os.path.join("models", wandb.run.name)
    os.mkdir(dir)

    model_path = os.path.join(dir, "model.pth")
    optim_path = os.path.join(dir, "optim.pth")
    cfg_path = os.path.join(dir, "cfg.yaml")
    vis_path = os.path.join(dir, "vis.pdf")
    metrics_path = os.path.join(dir, "metrics.csv")

    torch.save(model.state_dict(), model_path)
    torch.save(trainer.optimizer.state_dict(), optim_path)

    with open(cfg_path, "w") as f:
        OmegaConf.save(cfg, f)

    visualize_dvi_2d_contour_all(
        device=device,
        model=model,
        dataset=dataloader.dataset,
        save_fig_path=vis_path,
        save_csv_path=metrics_path,
    )

    wandb.run.log_model(path=model_path, name=f"{wandb.run.name}_model.pth")
    wandb.run.log_model(path=optim_path, name=f"{wandb.run.name}_optim.pth")
    wandb.run.log_model(path=cfg_path, name=f"{wandb.run.name}_cfg.yaml")
    wandb.run.log_model(path=vis_path, name=f"{wandb.run.name}_vis.pdf")
    wandb.run.log_model(path=metrics_path, name=f"{wandb.run.name}_metrics.csv")


def download_run_np(project: str, name: str) -> str:
    dir = f"./models/{project}/{name}"

    if not os.path.exists(dir):
        api = wandb.Api()

        # "optim.pth:v0", "decoder.pth:v0"]:
        for type in [
            "model.pth:v1",
            "metrics.csv:v1",
            "cfg.yaml:v1",
            "optim.pth:v1",
            "decoder.pth:v1",
        ]:
            artifact = api.artifact(f"{project}/{name}_{type}")
            artifact.download(root=dir)

    return dir


def download_run_dvi(project: str, name: str) -> str:
    dir = f"./models/{project}/{name}"

    if not os.path.exists(dir):
        api = wandb.Api()

        for type in [
            "model.pth:v0",
            "optim.pth:v0",
            "cfg.yaml:v0",
            "vis.pdf:v0",
            "metrics.csv:v0",
        ]:
            artifact = api.artifact(f"{project}/{name}_{type}")
            artifact.download(root=dir)

    return dir


def load_state_dicts_np(
    dir: str, model: NP, trainer: AbstractTrainer, load_decoder_encoder_only: bool
) -> Tuple[NP, AbstractTrainer]:
    model_path = f"{dir}/model.pth"
    decoder_path = f"{dir}/decoder.pth"
    encoder_path = f"{dir}/encoder.pth"
    optim_path = f"{dir}/optim.pth"

    if load_decoder_encoder_only:
        if os.path.exists(decoder_path):
            decoder_state_dict = torch.load(
                decoder_path, map_location=torch.device("cpu"), weights_only=False
            )
            model.decoder.load_state_dict(decoder_state_dict)
            print(f"loaded decoder from {decoder_path}")
        else:
            print(f"decoder not found at {decoder_path}")

        if os.path.exists(encoder_path):
            encoder_state_dict = torch.load(
                encoder_path, map_location=torch.device("cpu"), weights_only=False
            )
            model.encoder.load_state_dict(encoder_state_dict)
            print(f"loaded encoder from {encoder_path}")
        else:
            print(f"encoder not found at {encoder_path}")

    else:
        if os.path.exists(model_path):
            model_state_dict = torch.load(
                model_path, map_location=torch.device("cpu"), weights_only=False
            )
            model.load_state_dict(model_state_dict)
            print(f"loaded model from {model_path}")
        else:
            print(f"model not found at {model_path}")

        if os.path.exists(optim_path):
            optim_state_dict = torch.load(
                optim_path, map_location=torch.device("cpu"), weights_only=False
            )
            trainer.optimizer.load_state_dict(optim_state_dict)
            print(f"loaded optim from {optim_path}")
        else:
            print(f"optim not found at {optim_path}")

    return model, trainer
