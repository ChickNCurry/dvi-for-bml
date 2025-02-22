from typing import Any, Tuple

import torch
from hydra import compose, initialize
from omegaconf import DictConfig
from torch.utils.data import DataLoader

import wandb
from src.architectures.cnp import AggrCNP, BcaCNP
from src.architectures.dvinp import DVINP
from src.architectures.lnp import AggrLNP, BcaLNP
from src.training.cnp_trainer import CNPTrainer
from src.training.dvinp_trainer import DVINPTrainer
from src.training.lnp_trainer import LNPTrainer
from src.utils.load_dvinp import load_dvinp
from src.utils.load_np import load_np


def download_run(project: str, name: str) -> str:

    api = wandb.Api()

    for type in ["model.pth:v0", "optim.pth:v0", "cfg.yaml:v0"]:
        artifact = api.artifact(f"{project}/{name}_{type}")
        artifact.download(root=f"../models/{project}/{name}")

    dir = f"../models/{project}/{name}"

    return dir


def get_np(
    dir: str,
    device: torch.device,
) -> Tuple[
    AggrLNP | BcaLNP | AggrCNP | BcaCNP, LNPTrainer | CNPTrainer, DataLoader[Any]
]:

    with initialize(version_base=None, config_path=dir):
        cfg = compose(config_name="cfg")

    model, trainer, test_loader = load_np(cfg=cfg, device=device, dir=dir)

    return model, trainer, test_loader


def get_dvinp(
    dir: str,
    device: torch.device,
) -> Tuple[DVINP, DVINPTrainer, DataLoader[Any]]:

    with initialize(version_base=None, config_path=dir):
        cfg = compose(config_name="cfg")

    model, trainer, test_loader = load_dvinp(cfg=cfg, device=device, dir=dir)

    return model, trainer, test_loader


def get_name_np(cfg: DictConfig) -> str:
    model_keys = ["model_variant", "context_variant", "self_attn_num_heads"]
    training_keys = ["trainer_variant", "max_clip_norm", "seed"]

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
        "use_score",
    ]
    training_keys = ["trainer_variant", "max_clip_norm", "seed"]

    model_values = [f"{v}" for k, v in cfg.model.items() if k in model_keys]
    training_values = [f"{v}" for k, v in cfg.training.items() if k in training_keys]

    return "-".join(model_values + training_values)
