from dataclasses import dataclass
from typing import List

import torch
from hydra import compose, initialize

from src.architectures.np import NP
from src.evaluation.common import ModelInfo, ModelType
from src.evaluation.predictive.num_pred_eval import num_pred_eval
from src.utils.helper import download_run
from src.utils.load_dvinp import load_dvinp
from src.utils.load_np import load_np


def run(model_infos: List[ModelInfo], num_samples: int, save_dir: str) -> None:
    try:
        import torch_directml  # type: ignore

        device = torch_directml.device()
    except ImportError:
        device = torch.device("cpu")

    for model_info in model_infos:

        dir = download_run(model_info.project, model_info.name)

        model: NP

        with initialize(version_base=None, config_path=f"../../{dir}"):
            cfg = compose(config_name="cfg")

            if model_info.type == ModelType.LNP or model_info.type == ModelType.CNP:
                model, _, _, val_loader = load_np(
                    cfg=cfg,
                    device=device,
                    dir=dir,
                )
            elif model_info.type == ModelType.DVINP:
                model, _, _, val_loader = load_dvinp(
                    cfg=cfg,
                    device=device,
                    dir=dir,
                )

        model_info.model = model
        model_info.val_loader = val_loader

    num_pred_eval(
        model_infos=model_infos,
        num_samples=num_samples,
        device=device,
        save_dir=save_dir,
    )


if __name__ == "__main__":

    infos = [
        ModelInfo(
            name="16-1-bca-free-dis-True-True-forwardandcontext-1.0-0",
            project="cluster-dvinp-score2",
            type=ModelType.DVINP,
        ),
        ModelInfo(
            name="16-1-bca-free-dis-True-False-forwardandcontext-1.0-0",
            project="cluster-dvinp-noscore",
            type=ModelType.DVINP,
        ),
        ModelInfo(
            name="16-1-bca-free-dis-True-False-context-1.0-0",
            project="cluster-dvinp-noscore",
            type=ModelType.DVINP,
        ),
        ModelInfo(
            name="1-mean-lnp-data-1.0-0",
            project="cluster-np",
            type=ModelType.LNP,
        ),
        ModelInfo(
            name="1-mean-cnp-data-None-0",
            project="cluster-np",
            type=ModelType.CNP,
        ),
    ]

    save_dir = "scripts/evaluation/plots"

    run(model_infos=infos, num_samples=512, save_dir=save_dir)
