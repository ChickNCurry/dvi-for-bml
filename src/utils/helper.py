import os

import wandb
from omegaconf import DictConfig


def download_run(project: str, name: str) -> str:
    dir = f"models/{project}/{name}"

    if not os.path.exists(dir):
        api = wandb.Api()

        for type in ["model.pth:v0", "optim.pth:v0", "cfg.yaml:v0"]:
            artifact = api.artifact(f"{project}/{name}_{type}")
            artifact.download(root=f"models/{project}/{name}")

    return dir


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
