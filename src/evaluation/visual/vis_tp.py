from typing import List, Tuple

import numpy as np
import torch
from matplotlib import cm
from matplotlib import pyplot as plt
from torch import Tensor
from torch.utils.data import DataLoader

from src.architectures.np import NP
from src.evaluation.taskposterior.grid import create_grid, eval_hist_on_grid


def vis_tp(
    models: List[NP],
    test_loader: DataLoader[Tuple[Tensor, Tensor]],
    num_samples: int,
    device: torch.device,
    max_context_size: int,
    ranges: List[Tuple[float, float]] = [(-5, 5), (-5, 5)],
    show_score: bool = True,
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

    fig, axs = plt.subplots(
        nrows=max_context_size,
        ncols=len(models),
        figsize=(2 * len(models), 2 * max_context_size),
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

        for column, model in enumerate(models):
            ax = axs[row, column]

            with torch.no_grad():
                _, z = model.inference(x_context, y_context, None, x_data)
                # (1, num_samples, z_dim)

            assert z is not None

            num_cells = int(np.sqrt(x_data.shape[0] * x_data.shape[1]))
            grid = create_grid(ranges, num_cells)

            log_probs = eval_hist_on_grid(
                z.reshape(-1, z[-1].shape[-1]).cpu().detach().numpy(),
                ranges,
                num_cells,
            )

            ax.contourf(grid[:, :, 0], grid[:, :, 1], np.exp(log_probs), cmap=cm.coolwarm)  # type: ignore

    plt.tight_layout()
    plt.show()
    plt.close()
