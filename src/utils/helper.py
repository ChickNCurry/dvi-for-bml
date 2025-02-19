from omegaconf import DictConfig


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
