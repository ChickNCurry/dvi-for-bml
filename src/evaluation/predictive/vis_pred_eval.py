from typing import List, Tuple

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import Tensor

from src.evaluation.common import ModelInfo, ModelType
from src.utils.datasets import hash_tensor


def vis_pred_eval(
    model_infos: List[ModelInfo],
    batch: Tuple[Tensor, Tensor],
    num_samples: int,
    device: torch.device,
    max_context_size: int,
    save_dir: str,
    index: int = 0,
) -> None:
    x_data, y_data = batch
    task_hash = hash_tensor(x_data)  # + hash_tensor(y_data)

    x_data = x_data.to(device)
    y_data = y_data.to(device)
    # (1, data_size, x_dim)
    # (1, data_size, y_dim)

    x_data = x_data.expand(1, num_samples, -1, -1)
    y_data = y_data.expand(1, num_samples, -1, -1)
    # (1, num_samples, data_size, x_dim)
    # (1, num_samples, data_size, y_dim)

    sorted, indices = x_data.squeeze(0).squeeze(-1).sort(dim=1)
    x_data_sorted = sorted.detach().cpu().numpy()
    y_data_sorted = (
        y_data.squeeze(0).squeeze(-1).gather(1, indices).detach().cpu().numpy()
    )  # (num_samples, data_size)

    fig, axs = plt.subplots(
        nrows=max_context_size,
        ncols=len(model_infos),
        figsize=(4 * len(model_infos), 2 * max_context_size),
    )

    fig.text(
        0,
        1,
        task_hash,
        fontsize=12,
        color="blue",
        ha="left",
        va="top",
    )

    if len(model_infos) == 1:
        axs = np.expand_dims(axs, axis=1)

    for row in range(max_context_size):
        context_size = row + 1

        axs[row, 0].set_ylabel(f"Context Size: {context_size}", fontsize=8)

        x_context = x_data[:, :, :context_size, :]
        y_context = y_data[:, :, :context_size, :]
        # (1, num_samples, context_size, x_dim)
        # (1, num_samples, context_size, y_dim)

        x_context_np = x_context.squeeze(0).squeeze(-1).detach().cpu().numpy()
        y_context_np = y_context.squeeze(0).squeeze(-1).detach().cpu().numpy()
        # (num_samples, data_size)

        for col, model_info in enumerate(model_infos):
            assert model_info.model is not None

            axs[0, col].set_title(model_info.name, fontsize=8)

            ax = axs[row, col]

            if model_info.type == ModelType.DVINP:
                y_dist_data, _ = model_info.model.inference(
                    x_context, y_context, None, x_data
                )
            else:
                with torch.no_grad():
                    y_dist_data, _ = model_info.model.inference(
                        x_context, y_context, None, x_data
                    )
            # (1, num_samples, data_size, y_dim)

            y_mu_sorted = (
                y_dist_data.mean.squeeze(0)
                .squeeze(-1)
                .gather(1, indices)
                .detach()
                .cpu()
                .numpy()
            )  # (num_samples, data_size)
            y_sigma_sorted = (
                y_dist_data.scale.squeeze(0)
                .squeeze(-1)
                .gather(1, indices)
                .detach()
                .cpu()
                .numpy()
            )  # (num_samples, data_size)

            ax.scatter(
                x_data_sorted[0], y_data_sorted[0], marker="o", c="black", zorder=2
            )
            ax.scatter(
                x_context_np[0], y_context_np[0], marker="X", c="red", s=100, zorder=3
            )

            for k in range(num_samples):
                ax.plot(
                    x_data_sorted[k], y_mu_sorted[k], alpha=0.8, c="tab:blue", zorder=1
                )

                if model_info.type == ModelType.CNP:
                    ax.fill_between(
                        x_data_sorted[0],
                        y_mu_sorted[k] - y_sigma_sorted[k],
                        y_mu_sorted[k] + y_sigma_sorted[k],
                        color="tab:purple",
                        alpha=0.2,
                        zorder=0,
                    )

    plt.savefig(f"{save_dir}/pred_{index}.pdf")
    plt.close()
