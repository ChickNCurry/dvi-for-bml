from typing import List, Tuple

import numpy as np
import torch
from matplotlib import cm
from matplotlib import pyplot as plt
from numpy.typing import NDArray
from omegaconf import DictConfig
from torch import Tensor
from torch.distributions import Distribution
from torch.utils.data import DataLoader

from src.architectures.cnp import AggrCNP, BcaCNP
from src.architectures.dvinp import DVINP
from src.architectures.lnp import AggrLNP, BcaLNP
from src.components.decoder.decoder_times_prior import DecoderTimesPrior
from src.evaluation.taskposterior.grid import (
    create_grid,
    eval_dist_on_grid,
    eval_hist_on_grid,
    eval_score_on_grid,
    sample_from_log_probs,
)
from src.utils.datasets import hash_tensor


def visualize_dvinp_both(
    device: torch.device,
    dvinp: DVINP,
    dataloader: DataLoader[Tuple[Tensor, Tensor]],
    num_samples: int,
    max_context_size: int,
    ranges: List[Tuple[float, float]],
    show_score: bool = True,
) -> Tuple[List[Distribution], List[NDArray[np.float32]]]:

    assert dvinp.decoder is not None

    x_data, y_data = next(iter(dataloader))
    task_hash = hash_tensor(x_data)  # + hash_tensor(y_data)

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

        context = torch.cat([x_context, y_context], dim=-1)
        # (1, num_samples, context_size, x_dim + y_dim)

        r = dvinp.encoder(context, None)

        target = DecoderTimesPrior(
            decoder=dvinp.decoder,
            x=x_context,
            y=y_context,
            mask=None,
        )

        _, z_samples = dvinp.cdvi.run_both_processes(target, r, None)

        tp_samples = z_samples[-1].cpu().detach().numpy()

        targets.append(target)
        samples.append(tp_samples)

        y_dist: Distribution = dvinp.decoder(z_samples[-1], x_data)

        y_mu_sorted = y_dist.mean.gather(2, indices).squeeze(0).cpu().detach().numpy()
        # (num_samples, target_size, y_dim)

        num_cells = int(np.sqrt(x_data.shape[0] * x_data.shape[1]))
        grid = create_grid(ranges, num_cells)

        # dvi_vals = eval_kde_on_grid(grid, z_samples[-1].cpu().detach().numpy())
        dvi_log_probs = eval_hist_on_grid(
            z_samples[-1].reshape(-1, z_samples[-1].shape[-1]).cpu().detach().numpy(),
            ranges,
            num_cells,
        )

        target_log_probs = eval_dist_on_grid(grid, target, device=device).squeeze(0)

        if show_score:
            score_vals = eval_score_on_grid(grid, target, device=device)

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
        ax[1].contourf(grid[:, :, 0], grid[:, :, 1], np.exp(dvi_log_probs), cmap=cm.coolwarm)  # type: ignore

        ax[2].set_title("$p_\\theta(y_{1:N}|x_{1:N},z_T)p_\\theta(z_T)$")
        ax[2].contourf(grid[:, :, 0], grid[:, :, 1], np.exp(target_log_probs), cmap=cm.coolwarm)  # type: ignore

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

        for a in ax:
            a.set_xticks([])
            a.set_yticks([])

    plt.show()

    return targets, tp_samples  # type: ignore


def visualize_np(
    model: AggrLNP | BcaLNP | AggrCNP | BcaCNP,
    device: torch.device,
    dataloader: DataLoader[Tuple[Tensor, Tensor]],
    num_samples: int,
    max_context_size: int,
) -> None:
    x_data, y_data = next(iter(dataloader))
    task_hash = hash_tensor(x_data)  # + hash_tensor(y_data)

    x_data = x_data.to(device)
    y_data = y_data.to(device)
    # (1, context_size, x_dim)
    # (1, context_size, y_dim)

    x_data = x_data.expand(1, num_samples, -1, -1)
    y_data = y_data.expand(1, num_samples, -1, -1)
    # (1, num_samples, context_size, x_dim)
    # (1, num_samples, context_size, y_dim)

    x_data_sorted, indices = x_data.sort(dim=2)
    x_data_sorted = x_data_sorted.squeeze(0).squeeze(-1).detach().cpu().numpy()
    y_data_sorted = (
        y_data.gather(2, indices).squeeze(0).squeeze(-1).detach().cpu().numpy()
    )
    # (num_samples, context_size, x_dim)
    # (num_samples, context_size, y_dim)

    fig = plt.figure(figsize=(6, 3 * max_context_size), constrained_layout=True)
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

    for row, subfig in enumerate(subfigs):
        subfig.suptitle(f"context size: {row + 1}")
        ax = subfig.subplots(nrows=1, ncols=1, width_ratios=[1])

        context_size = row + 1

        x_context = x_data[:, :, :context_size, :]
        y_context = y_data[:, :, :context_size, :]
        # (1, num_samples, context_size, x_dim)
        # (1, num_samples, context_size, y_dim)

        y_dist, _ = model.inference(x_context, y_context, None, x_data)

        y_mu_sorted = (
            y_dist.mean.gather(2, indices).squeeze(0).squeeze(-1).detach().cpu().numpy()
        )
        y_sigma_sorted = (
            y_dist.scale.gather(2, indices)
            .squeeze(0)
            .squeeze(-1)
            .detach()
            .cpu()
            .numpy()
        )
        # (num_samples, target_size, y_dim)

        ax.scatter(x_data_sorted, y_data_sorted, marker="o", c="black", zorder=2)
        ax.scatter(
            x_context.detach().cpu().numpy(),
            y_context.detach().cpu().numpy(),
            marker="X",
            c="red",
            s=100,
            zorder=3,
        )

        for k in range(num_samples):
            ax.plot(
                x_data_sorted[k],
                y_mu_sorted[k],
                alpha=0.8,
                c="tab:blue",
                zorder=1,
            )
            ax.fill_between(
                x_data_sorted[k],
                y_mu_sorted[k] - y_sigma_sorted[k],  # Lower bound
                y_mu_sorted[k] + y_sigma_sorted[k],  # Upper bound
                color="tab:purple",
                alpha=0.2,
                zorder=0,
            )

    plt.show()


def visualize_samples_1d(
    samples: NDArray[np.float32], bins: int, range: Tuple[float, float]
) -> None:
    # (num_samples, z_dim)

    plt.hist(samples[:, 0], bins=bins, density=True, range=range)
    plt.show()


def visualize_samples_2d(
    samples: NDArray[np.float32],
    bins: int,
    range: List[Tuple[float, float]],
) -> None:
    # (num_samples, z_dim)

    plt.hist2d(samples[:, 0], samples[:, 1], density=True, bins=bins, range=range)
    plt.show()


def visualize_vals_on_grid_1d(
    grid: NDArray[np.float32], vals: NDArray[np.float32]
) -> None:
    # (dim1, dim2, ..., z_dim)
    # (dim1, dim2, ...)

    grid_flat = grid.reshape(-1, grid.shape[-1])
    # (dim1 * dim2 * ..., z_dim)

    vals_flat = vals.reshape(-1)
    # (dim1 * dim2 * ...)

    plt.plot(grid_flat, vals_flat)
    plt.show()


def visualize_vals_on_grid_2d(
    grid: NDArray[np.float32],
    vals: NDArray[np.float32],
) -> None:
    # (dim1, dim2, ..., z_dim)
    # (dim1, dim2, ...)

    fig = plt.figure(figsize=(8, 4))

    ax1 = fig.add_subplot(121)
    ax1.contourf(grid[:, :, 0], grid[:, :, 1], vals, cmap=cm.coolwarm)  # type: ignore

    ax2 = fig.add_subplot(122, projection="3d")
    ax2.plot_surface(grid[:, :, 0], grid[:, :, 1], vals, cmap=cm.coolwarm)  # type: ignore

    plt.tight_layout()
    plt.show()
