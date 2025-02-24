from dataclasses import dataclass
from typing import List

import torch
from hydra import compose, initialize

from src.architectures.np import NP
from src.evaluation.visual.vis_pred import vis_pred
from src.evaluation.visual.vis_tp import vis_tp
from src.utils.helper import download_run
from src.utils.load_dvinp import load_dvinp
from src.utils.load_np import load_np


@dataclass
class ModelInfo:
    name: str
    project: str
    type: str


def run(
    infos: List[ModelInfo], num_tasks: int, num_samples: int, max_context_size: int
) -> None:
    try:
        import torch_directml  # type: ignore

        device = torch_directml.device()
    except ImportError:
        device = torch.device("cpu")

    models: List[NP] = []

    for info in infos:

        dir = download_run(info.project, info.name)

        model: NP

        with initialize(version_base=None, config_path=f"../../{dir}"):
            cfg = compose(config_name="cfg")

            if info.type == "np":
                model, _, test_loader, _ = load_np(
                    cfg=cfg,
                    device=device,
                    dir=dir,
                )
            elif info.type == "dvinp":
                model, _, test_loader, _ = load_dvinp(
                    cfg=cfg,
                    device=device,
                    dir=dir,
                )

        models.append(model)

    for index, batch in enumerate(test_loader):

        if index == num_tasks:
            break

        vis_pred(
            models,
            batch,
            num_samples,
            device,
            max_context_size,
            save_dir,
            names=[info.name for info in infos],
        )

        vis_tp(
            models,
            batch,
            num_samples,
            device,
            max_context_size,
            save_dir,
            names=[info.name for info in infos],
        )


if __name__ == "__main__":

    infos = [
        ModelInfo(
            name="1-mean-lnp-data-None-0",
            project="cluster-np",
            type="np",
        ),
        ModelInfo(
            name="16-1-bca-free-dis-True-False-forwardandcontext-1.0-0",
            project="cluster-dvinp-noscore",
            type="dvinp",
        ),
    ]

    save_dir = "scripts/evaluation/plots"

    run(infos=infos, num_tasks=1, num_samples=512, max_context_size=4)
