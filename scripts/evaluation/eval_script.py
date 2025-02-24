from dataclasses import dataclass

import torch
from hydra import compose, initialize

from src.architectures.np import NP
from src.evaluation.predictive.eval import eval_np
from src.utils.helper import download_run
from src.utils.load_dvinp import load_dvinp
from src.utils.load_np import load_np


@dataclass
class ModelInfo:
    name: str
    project: str
    type: str


def run(info: ModelInfo, num_samples: int) -> None:
    try:
        import torch_directml  # type: ignore

        device = torch_directml.device()
    except ImportError:
        device = torch.device("cpu")

    dir = download_run(info.project, info.name)

    model: NP

    with initialize(version_base=None, config_path=dir):
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

    df = eval_np(model, val_loader, num_samples, device)

    file = f"scripts/eval/{info.name}.csv"
    df.to_csv(file)


if __name__ == "__main__":

    # info = ModelInfo(name="1-mean-lnp-data-None-0", project="cluster-np", type="np")
    info = ModelInfo(
        name="16-1-bca-free-dis-True-False-forwardandcontext-1.0-0",
        project="cluster-dvinp-noscore",
        type="dvinp",
    )

    run(info, 2)
