from typing import List

from matplotlib import pyplot as plt
import pandas as pd
import torch
from hydra import compose, initialize

from dviforbml.architectures.np import NP
from dviforbml.evaluation.common import ModelInfo, ModelType
from dviforbml.evaluation.predictive.vis_pred_eval import (
    vis_pred_eval,
    vis_pred_eval_gt,
)
from dviforbml.evaluation.taskposterior.vis_tp_eval import vis_tp_eval
from dviforbml.utils.helper import download_run_np
from dviforbml.utils.load_dvinp import load_dvinp
from dviforbml.utils.load_np import load_np


def vis_metrics(model_infos: List[ModelInfo], save_dir: str) -> None:
    metrics = ["lmpl", "mse", "jsd"]

    fig, ax = plt.subplots(
        nrows=1,
        ncols=len(metrics),
        figsize=(4 * len(metrics), 3),
    )

    for i, metric in enumerate(metrics):
        for info in model_infos:
            df = pd.read_csv(f"{info.dir}/metrics.csv")

            if (df[metric] == 0).all():
                continue

            df = df.rename(columns={df.columns[0]: "index"})
            ax[i].plot(df["index"], df[metric], label=info.name, marker="o")

        ax[i].set_xlabel("Context Size")
        ax[i].set_ylabel(metric.upper())
        ax[i].legend(fontsize="small")
        ax[i].grid(True)

    plt.tight_layout()
    plt.savefig(f"{save_dir}/metrics.pdf")


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
                    cfg=cfg,
                    device=device,
                    dir=dir,
                    load_model=True,
                    load_decoder_encoder_only=True if info.project == "ula" else False,
                )

        info.model = model
        info.dir = dir

    # model_infos.append(model_infos[0])
    # print(len(model_infos))

    vis_metrics(model_infos, save_dir=save_dir)

    # for index, batch in enumerate(test_loader):
    #     if index == num_tasks:
    #         break

    #     # vis_pred_eval_gt(
    #     #     model_infos,
    #     #     batch,
    #     #     num_samples,
    #     #     device,
    #     #     max_context_size,
    #     #     save_dir,
    #     #     index=index,
    #     # )

    #     # vis_pred_eval(
    #     #     model_infos,
    #     #     batch,
    #     #     num_samples,
    #     #     device,
    #     #     max_context_size,
    #     #     save_dir,
    #     #     index=index,
    #     # )

    #     vis_tp_eval(
    #         model_infos,
    #         batch,
    #         num_samples,
    #         device,
    #         max_context_size,
    #         save_dir,
    #         show_score=True,
    #         index=index,
    #     )


if __name__ == "__main__":
    infos = [
        ModelInfo(
            name="16-8-None-bca-dis-free-True-fwdcntxt-11",
            project="DVINP-HOPE-SINE",
            type=ModelType.DVINP,
        ),
        ModelInfo(
            name="128-8-None-bca-ula-constr-False-fwdcntxt-11",
            project="ula",
            type=ModelType.DVINP,
        ),
    ]

    save_dir = "scripts/evaluation/plots"

    # run(model_infos=infos, num_tasks=10, num_samples=100, max_context_size=5)
    run(model_infos=infos, num_tasks=10, num_samples=1600, max_context_size=5)


# ModelInfo(
#             name="2-8-16-bca-cnp-data-11",
#             project="NP-HOPE",
#             type=ModelType.CNP,
#         ),
#         ModelInfo(
#             name="2-8-16-bca-lnp-data-11",
#             project="NP-HOPE",
#             type=ModelType.LNP,
#         ),
#         ModelInfo(
#             name="16-8-None-bca-dis-free-True-fwdcntxt-11",
#             project="DVINP-HOPE-NEW",
#             type=ModelType.DVINP,
#         ),
#         ModelInfo(
#             name="128-8-None-bca-ula-constr-False-fwdcntxt-11",
#             project="ula",
#             type=ModelType.DVINP,
#         ),
