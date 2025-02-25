from dataclasses import dataclass
from typing import Any, List

import torch
from hydra import compose, initialize
from torch.utils.data import DataLoader

from src.architectures.np import NP
from src.evaluation.predictive.num_pred_eval import num_pred_eval
from src.utils.helper import download_run
from src.utils.load_dvinp import load_dvinp
from src.utils.load_np import load_np


@dataclass
class ModelInfo:
    name: str
    project: str
    type: str


def run(infos: List[ModelInfo], num_samples: int, save_dir: str) -> None:
    try:
        import torch_directml  # type: ignore

        device = torch_directml.device()
    except ImportError:
        device = torch.device("cpu")

    names: List[str] = []
    models: List[NP] = []
    val_loaders: List[DataLoader[Any]] = []

    for info in infos:

        dir = download_run(info.project, info.name)

        model: NP

        with initialize(version_base=None, config_path=f"../../{dir}"):
            cfg = compose(config_name="cfg")

            if info.type == "np":
                model, _, _, val_loader = load_np(
                    cfg=cfg,
                    device=device,
                    dir=dir,
                )
            elif info.type == "dvinp":
                model, _, _, val_loader = load_dvinp(
                    cfg=cfg,
                    device=device,
                    dir=dir,
                )

        names.append(info.name)
        models.append(model)
        val_loaders.append(val_loader)

    num_pred_eval(
        names=names,
        models=models,
        val_loaders=val_loaders,
        num_samples=num_samples,
        device=device,
        save_dir=save_dir,
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

    run(infos=infos, num_samples=512, save_dir=save_dir)
