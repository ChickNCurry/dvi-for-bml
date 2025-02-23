from typing import List, Tuple

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import Tensor
from torch.utils.data import DataLoader

from src.architectures.np import NP


def vis_pred(
    models: List[NP],
    test_loader: DataLoader[Tuple[Tensor, Tensor]],
    num_samples: int,
    device: torch.device,
    max_context_size: int,
    show_sigma: bool = False,
    names: List[str] | None = None,
) -> None:
    x_data, y_data = next(iter(test_loader))
    x_data = x_data.to(device)
    y_data = y_data.to(device)
    # (1, data_size, x_dim)
    # (1, data_size, y_dim)

    x_data = x_data.expand(1, num_samples, -1, -1)
    y_data = y_data.expand(1, num_samples, -1, -1)
    # (1, num_samples, data_size, x_dim)
    # (1, num_samples, data_size, y_dim)

    x_data_sorted, indices = x_data.squeeze(0).squeeze(-1).sort(dim=1)
    x_data_sorted = x_data_sorted.detach().cpu().numpy()
    y_data_sorted = (
        y_data.squeeze(0).squeeze(-1).gather(1, indices).detach().cpu().numpy()
    )
    # (num_samples, data_size)

    fig, axs = plt.subplots(
        nrows=max_context_size,
        ncols=len(models),
        figsize=(4 * len(models), 2 * max_context_size),
    )

    if len(models) == 1:
        axs = np.expand_dims(axs, axis=1)

    if names is not None:
        for i, name in enumerate(names):
            axs[0, i].set_title(name)

    for row in range(max_context_size):
        context_size = row + 1

        fig.text(
            0.02,  # X-coordinate for left margin
            1 - (row + 0.5) / max_context_size,  # Y-coordinate for row center
            f"Context Size: {context_size}",
            va="center",
            ha="center",
            fontsize=12,
            rotation=90,
        )

        x_context = x_data[:, :, :context_size, :]
        y_context = y_data[:, :, :context_size, :]
        # (1, num_samples, context_size, x_dim)
        # (1, num_samples, context_size, y_dim)

        x_context_np = x_context.squeeze(0).squeeze(-1).detach().cpu().numpy()
        y_context_np = y_context.squeeze(0).squeeze(-1).detach().cpu().numpy()
        # (num_samples, data_size)

        for column, model in enumerate(models):
            ax = axs[row, column]

            with torch.no_grad():
                y_dist_data, _ = model.inference(x_context, y_context, None, x_data)
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

                if show_sigma:
                    ax.fill_between(
                        x_data_sorted[0],
                        y_mu_sorted[k] - y_sigma_sorted[k],
                        y_mu_sorted[k] + y_sigma_sorted[k],
                        color="tab:purple",
                        alpha=0.2,
                        zorder=0,
                    )

    plt.tight_layout()
    plt.show()
    plt.close()
