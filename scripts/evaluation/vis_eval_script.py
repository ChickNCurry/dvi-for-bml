from typing import List

import torch
from hydra import compose, initialize

from dviforbml.architectures.np import NP
from dviforbml.evaluation.common import ModelInfo, ModelType
from dviforbml.evaluation.predictive.vis_pred_eval import vis_pred_eval
from dviforbml.evaluation.taskposterior.vis_tp_eval import vis_tp_eval
from dviforbml.utils.helper import download_run_np
from dviforbml.utils.load_dvinp import load_dvinp
from dviforbml.utils.load_np import load_np


def run(
    model_infos: List[ModelInfo],
    num_tasks: int,
    num_samples: int,
    max_context_size: int,
) -> None:
    try:
        import torch_directml

        device = torch_directml.device()
    except ImportError:
        device = torch.device("cpu")

    for info in model_infos:
        dir = download_run_np(info.project, info.name)

        model: NP

        with initialize(version_base=None, config_path=f"../../{dir}"):
            cfg = compose(config_name="cfg")

            if info.type == ModelType.LNP or info.type == ModelType.CNP:
                model, _, test_loader, _ = load_np(
                    cfg=cfg,
                    device=device,
                    dir=dir,
                    load_model=True,
                )
            elif info.type == ModelType.DVINP:
                model, _, test_loader, _ = load_dvinp(
                    cfg=cfg, device=device, dir=dir, load_model=True
                )

        info.model = model

    for index, batch in enumerate(test_loader):
        if index == num_tasks:
            break

        vis_pred_eval(
            model_infos,
            batch,
            num_samples,
            device,
            max_context_size,
            save_dir,
            index=index,
        )

        vis_tp_eval(
            model_infos,
            batch,
            num_samples,
            device,
            max_context_size,
            save_dir,
            show_score=True,
            index=index,
        )


if __name__ == "__main__":
    infos = [
        ModelInfo(
            name="16-None-bca-dis-free-True-cntxt-0",
            project="DVINP-NEW",
            type=ModelType.DVINP,
        ),
        ModelInfo(
            name="16-None-mean-dis-free-True-cntxt-0",
            project="DVINP-NEW",
            type=ModelType.DVINP,
        ),
        ModelInfo(
            name="16-None-mean-dis-free-True-fwdcntxt-0",
            project="DVINP-NEW",
            type=ModelType.DVINP,
        ),
    ]

    save_dir = "scripts/evaluation/plots"

    run(model_infos=infos, num_tasks=1, num_samples=1600, max_context_size=5)
