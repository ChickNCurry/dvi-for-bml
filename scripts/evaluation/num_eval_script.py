from dataclasses import dataclass
from typing import List

import torch
from hydra import compose, initialize

from dviforbml.evaluation.taskposterior.num_tp_eval import num_tp_eval
from dviforbml.architectures.np import NP
from dviforbml.evaluation.common import ModelInfo, ModelType
from dviforbml.evaluation.predictive.num_pred_eval import num_pred_eval
from dviforbml.utils.helper import download_run
from dviforbml.utils.load_dvinp import load_dvinp
from dviforbml.utils.load_np import load_np


def run(model_infos: List[ModelInfo], num_samples: int, save_dir: str) -> None:
    try:
        import torch_directml

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

    num_tp_eval(
        model_infos=model_infos,
        num_samples=num_samples,
        ranges=[(-5, 5), (-5, 5)],
        device=device,
        save_dir=save_dir,
    )


if __name__ == "__main__":
    infos = [
        ModelInfo(
            name="1-bca-free-dis-True-cntxt",
            project="cluster-dvinp-sine",
            type=ModelType.DVINP,
        ),
        ModelInfo(
            name="1-bca-free-dis_score-True-cntxt",
            project="cluster-dvinp-sine",
            type=ModelType.DVINP,
        ),
        ModelInfo(
            name="1-bca-free-dis_score-True-fwdcntxt",
            project="cluster-dvinp-sine-2",
            type=ModelType.DVINP,
        ),
        ModelInfo(
            name="1-bca-free-dis-True-fwdcntxt",
            project="cluster-dvinp-sine-2",
            type=ModelType.DVINP,
        ),
    ]

    save_dir = "scripts/evaluation/plots"

    run(model_infos=infos, num_samples=400, save_dir=save_dir)
