from typing import List, Tuple

import numpy as np
import torch
from matplotlib import cm
from matplotlib import pyplot as plt
from numpy.typing import NDArray
from torch import Tensor
from torch.distributions import Distribution
from torch.utils.data import DataLoader

from dviforbml.architectures.dvinp import DVINP
from dviforbml.evaluation.taskposterior.grid import (
    create_grid,
    eval_dist_on_grid,
    eval_hist_on_grid,
    eval_score_on_grid,
    sample_from_log_probs,
)
from dviforbml.utils.datasets import hash_tensor


def visualize_dvinp(
    device: torch.device,
    dvinp: DVINP,
    dataloader: DataLoader[Tuple[Tensor, Tensor]],
    num_samples: int,
    max_context_size: int,
    ranges: List[Tuple[int, int]],
    show_score: bool = True,
    save_dir: str | None = None,
) -> Tuple[List[Distribution], List[NDArray[np.float32]]]:
    assert dvinp.decoder is not None

    x_data, y_data = next(iter(dataloader))
    task_hash = hash_tensor(x_data)

    x_data = x_data.to(device)
    y_data = y_data.to(device)
    # (1, context_size, x_dim)
    # (1, context_size, y_dim)

    x_data = x_data.expand(1, num_samples, -1, -1)
    y_data = y_data.expand(1, num_samples, -1, -1)
    # (1, num_samples, context_size, x_dim)
    # (1, num_samples, context_size, y_dim)

    x_data_sorted, indices = x_data.sort(dim=2)
    x_data_sorted = x_data_sorted.cpu().squeeze(0).detach().numpy()
    y_data_sorted = y_data.gather(2, indices).squeeze(0).cpu().detach().numpy()
    # (num_samples, context_size, x_dim)
    # (num_samples, context_size, y_dim)

    fig = plt.figure(figsize=(15, 3 * max_context_size), constrained_layout=True)
    subfigs = fig.subfigures(nrows=max_context_size, ncols=1)

    fig.text(
        0,
        1,
        task_hash,
        fontsize=12,
        color="blue",
        ha="left",
        va="top",
    )

    targets = []
    samples = []

    for row, subfig in enumerate(subfigs):
        subfig.suptitle(f"context size: {row + 1}")
        ax = subfig.subplots(nrows=1, ncols=4, width_ratios=[2, 1, 1, 2])

        context_size = row + 1

        x_context = x_data[:, :, :context_size, :]
        y_context = y_data[:, :, :context_size, :]
        # (1, num_samples, context_size, x_dim)
        # (1, num_samples, context_size, y_dim)

        y_dist, z = dvinp.inference(x_context, y_context, None, x_data)
        target_dist = dvinp.get_target_dist(x_context, y_context, None)

        tp_samples = z.cpu().detach().numpy()

        targets.append(target_dist)
        samples.append(tp_samples)

        # y_dist: Distribution = dvinp.decoder(z_samples[-1], x_data)

        y_mu_sorted = y_dist.mean.gather(2, indices).squeeze(0).cpu().detach().numpy()
        # (num_samples, target_size, y_dim)

        num_cells = int(np.sqrt(x_data.shape[0] * x_data.shape[1]))
        grid = create_grid(ranges, num_cells)

        # dvi_vals = eval_kde_on_grid(grid, z_samples[-1].cpu().detach().numpy())
        dvi_log_probs = eval_hist_on_grid(
            z.reshape(-1, z.shape[-1]).cpu().detach().numpy(),
            ranges,
            num_cells,
        )

        target_log_probs = eval_dist_on_grid(grid, target_dist, device=device).squeeze(
            0
        )

        if show_score:
            score_vals = eval_score_on_grid(grid, target_dist, device=device)

        target_samples_np = sample_from_log_probs(grid, target_log_probs, num_samples)
        target_samples = torch.from_numpy(target_samples_np).unsqueeze(0).to(device)

        y_dist_test: Distribution = dvinp.decoder(target_samples, x_data)

        y_mu_test_sorted = (
            y_dist_test.mean.gather(2, indices).squeeze(0).cpu().detach().numpy()
        )

        ax[0].set_title("$\mu_{1:M}$ of $p_{\\theta}(y_{1:M}|x_{1:M},z_T)$")
        ax[0].scatter(x_data_sorted, y_data_sorted, marker="o", c="black", zorder=1)
        ax[0].scatter(
            x_context.cpu().detach().numpy(),
            y_context.cpu().detach().numpy(),
            marker="X",
            c="red",
            s=100,
            zorder=2,
        )
        for k in range(num_samples):
            ax[0].plot(
                x_data_sorted[k].squeeze(-1),
                y_mu_sorted[k].squeeze(-1),
                alpha=0.2,
                c="tab:blue",
                zorder=0,
            )

        ax[1].set_title("$q_\phi(z_T|z_{0:T-1}, D^c)$")
        ax[1].contourf(
            grid[:, :, 0], grid[:, :, 1], np.exp(dvi_log_probs), cmap=cm.coolwarm
        )
        ax[1].grid(True)

        ax[2].set_title("$p_\\theta(y_{1:N}|x_{1:N},z_T)p_\\theta(z_T)$")
        ax[2].contourf(
            grid[:, :, 0], grid[:, :, 1], np.exp(target_log_probs), cmap=cm.coolwarm
        )
        ax[2].grid(True)

        if show_score:
            ax[1].quiver(
                grid[:, :, 0],
                grid[:, :, 1],
                score_vals[:, :, 0],
                score_vals[:, :, 1],
                scale_units="xy",
            )
            ax[2].quiver(
                grid[:, :, 0],
                grid[:, :, 1],
                score_vals[:, :, 0],
                score_vals[:, :, 1],
                scale_units="xy",
            )

        ax[3].set_title("$\mu_{1:M}$ of $p_{\\theta}(y_{1:M}|x_{1:M},z_T)$")
        ax[3].scatter(x_data_sorted, y_data_sorted, marker="o", c="black", zorder=1)
        ax[3].scatter(
            x_context.cpu().detach().numpy(),
            y_context.cpu().detach().numpy(),
            marker="X",
            c="red",
            s=100,
            zorder=2,
        )
        for k in range(num_samples):
            ax[3].plot(
                x_data_sorted[k].squeeze(-1),
                y_mu_test_sorted[k].squeeze(-1),
                alpha=0.2,
                c="tab:orange",
                zorder=0,
            )

        # for a in ax:
        #     a.set_xticks([])
        #     a.set_yticks([])

    if save_dir is not None:
        plt.savefig(f"{save_dir}/dvinp.png")
    else:
        plt.show()

    return targets, tp_samples
