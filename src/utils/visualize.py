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

from src.components.nn.decoder import DecoderTimesPrior
from src.components.dvinp import DVINP
from src.utils.grid import (
    create_grid,
    eval_dist_on_grid,
    eval_hist_on_grid,
    eval_kde_on_grid,
    sample_from_vals,
)


def visualize_dvinp_both(
    device: torch.device,
    dvinp: DVINP,
    dataloader: DataLoader[Tuple[Tensor, Tensor]],
    config: DictConfig,
    num_samples: int,
    max_context_size: int,
    ranges: List[Tuple[float, float]],
) -> Tuple[List[Distribution], List[NDArray[np.float32]]]:

    x_data, y_data = next(iter(dataloader))

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

    fig = plt.figure(
        figsize=(12, 3 * max_context_size * 2),
        constrained_layout=True,
    )
    subfigs = fig.subfigures(nrows=max_context_size, ncols=1)

    targets = []
    samples = []

    for row, subfig in enumerate(subfigs):
        subfig.suptitle(f"context size: {row + 1}")
        ax = subfig.subplots(nrows=2, ncols=2, width_ratios=[3, 1])

        context_size = row + 1

        x_context = x_data[:, :, :context_size, :]
        y_context = y_data[:, :, :context_size, :]
        # (1, num_samples, context_size, x_dim)
        # (1, num_samples, context_size, y_dim)

        context = torch.cat([x_context, y_context], dim=-1)
        # (1, num_samples, context_size, x_dim + y_dim)

        r_aggr, r_non_aggr = dvinp.encoder(context, None)

        target = DecoderTimesPrior(
            decoder=dvinp.decoder,
            x_target=x_context,
            y_target=y_context,
            x_context=x_context,
            r_aggr=r_aggr,
            r_non_aggr=r_non_aggr,
            mask=None,
        )

        targets.append(target)

        _, _, z_samples = dvinp.cdvi.run_chain(
            target=target,
            r_aggr=r_aggr,
            r_non_aggr=r_non_aggr,
            mask=None,
        )

        tp_samples = z_samples[-1].cpu().detach().numpy()
        samples.append(tp_samples)

        y_dist: Distribution = dvinp.decoder(
            z_samples[-1],
            x_data,
            x_context,
            r_aggr,
            r_non_aggr,
            None,
        )

        y_mu_sorted = y_dist.mean.gather(2, indices).squeeze(0).cpu().detach().numpy()
        # (num_samples, target_size, y_dim)

        for k in range(num_samples):
            ax[0][0].plot(
                x_data_sorted[k].squeeze(-1),
                y_mu_sorted[k].squeeze(-1),
                alpha=0.2,
                c="tab:blue",
                zorder=0,
            )

        ax[0][0].scatter(x_data_sorted, y_data_sorted, marker="o", c="black", zorder=1)
        ax[0][0].scatter(
            x_context.cpu().detach().numpy(),
            y_context.cpu().detach().numpy(),
            marker="X",
            c="red",
            s=100,
            zorder=2,
        )

        ax[0][0].set_title("$\mu_y$ of $p_{\\theta}(y|x,z_T)$")

        num_cells = int(np.sqrt(x_data.shape[0] * x_data.shape[1]))
        grid = create_grid(ranges, num_cells)

        # dvi_vals = eval_kde_on_grid(grid, z_samples[-1].cpu().detach().numpy())
        dvi_vals = eval_hist_on_grid(
            z_samples[-1].reshape(-1, z_samples[-1].shape[-1]).cpu().detach().numpy(),
            ranges,
            num_cells,
        )

        target_vals = eval_dist_on_grid(
            grid, target, x_data.shape[0], x_data.shape[1], device=device
        )
        target_samples = sample_from_vals(grid, target_vals, num_samples)
        target_samples = torch.from_numpy(target_samples).unsqueeze(0).to(device)

        y_dist_test: Distribution = dvinp.decoder(
            target_samples,
            x_data,
            x_context,
            r_aggr,
            r_non_aggr,
            None,
        )

        y_mu_test_sorted = (
            y_dist_test.mean.gather(2, indices).squeeze(0).cpu().detach().numpy()
        )

        for k in range(num_samples):
            ax[1][0].plot(
                x_data_sorted[k].squeeze(-1),
                y_mu_test_sorted[k].squeeze(-1),
                alpha=0.2,
                c="tab:orange",
                zorder=0,
            )

        ax[1][0].scatter(x_data_sorted, y_data_sorted, marker="o", c="black", zorder=1)
        ax[1][0].scatter(
            x_context.cpu().detach().numpy(),
            y_context.cpu().detach().numpy(),
            marker="X",
            c="red",
            s=100,
            zorder=2,
        )

        ax[1][0].set_title("$\mu_y$ of $p_\\theta(y|x,z_T)$")

        ax[0][1].contourf(grid[:, :, 0], grid[:, :, 1], dvi_vals, cmap=cm.coolwarm)  # type: ignore
        ax[0][1].set_title("$q_\phi(z_T|z_{0:T-1}, D^c)$")

        ax[1][1].contourf(grid[:, :, 0], grid[:, :, 1], target_vals, cmap=cm.coolwarm)  # type: ignore
        ax[1][1].set_title("$p_\\theta(y_k|x_k,z_T)p_\\theta(z_T)$")

        for a in ax:
            for b in a:
                b.set_xticks([])
                b.set_yticks([])

    plt.show()

    return targets, tp_samples  # type: ignore


def visualize_dvinp(
    device: torch.device,
    dvinp: DVINP,
    dataloader: DataLoader[Tuple[Tensor, Tensor]],
    config: DictConfig,
    num_samples: int,
    max_context_size: int,
    ranges: List[Tuple[float, float]],
) -> Tuple[List[Distribution], List[NDArray[np.float32]]]:

    x_data, y_data = next(iter(dataloader))

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

    fig = plt.figure(
        figsize=(12, 3 * max_context_size * 1),
        constrained_layout=True,
    )
    subfigs = fig.subfigures(nrows=max_context_size, ncols=1)

    targets = []
    samples = []

    for row, subfig in enumerate(subfigs):
        subfig.suptitle(f"context size: {row + 1}")
        ax = subfig.subplots(nrows=1, ncols=2, width_ratios=[3, 1])

        context_size = row + 1

        x_context = x_data[:, :, :context_size, :]
        y_context = y_data[:, :, :context_size, :]
        # (1, num_samples, context_size, x_dim)
        # (1, num_samples, context_size, y_dim)

        context = torch.cat([x_context, y_context], dim=-1)
        # (1, num_samples, context_size, x_dim + y_dim)

        r_aggr, r_non_aggr = dvinp.encoder(context, None)

        target = DecoderTimesPrior(
            decoder=dvinp.decoder,
            x_target=x_context,
            y_target=y_context,
            x_context=x_context,
            r_aggr=r_aggr,
            r_non_aggr=r_non_aggr,
            mask=None,
        )

        targets.append(target)

        _, _, z_samples = dvinp.cdvi.run_chain(
            target,
            r_aggr,
            r_non_aggr,
            None,
        )

        tp_samples = z_samples[-1].cpu().detach().numpy()
        samples.append(tp_samples)

        y_dist: Distribution = dvinp.decoder(
            z_samples[-1],
            x_data,
            x_context,
            r_aggr,
            r_non_aggr,
            None,
        )

        y_mu_sorted = y_dist.mean.gather(2, indices).squeeze(0).cpu().detach().numpy()
        # (num_samples, target_size, y_dim)

        for k in range(num_samples):
            ax[0].plot(
                x_data_sorted[k].squeeze(-1),
                y_mu_sorted[k].squeeze(-1),
                alpha=0.2,
                c="tab:blue",
                zorder=0,
            )

        ax[0].scatter(x_data_sorted, y_data_sorted, marker="o", c="black", zorder=1)
        ax[0].scatter(
            x_context.cpu().detach().numpy(),
            y_context.cpu().detach().numpy(),
            marker="X",
            c="red",
            s=100,
            zorder=2,
        )

        ax[0].set_title("$\mu_y$ of $p_{\\theta}(y|x,z_T)$")

        num_cells = int(np.sqrt(x_data.shape[0] * x_data.shape[1]))
        grid = create_grid(ranges, num_cells)

        # dvi_vals = eval_kde_on_grid(grid, z_samples[-1].cpu().detach().numpy())
        dvi_vals = eval_hist_on_grid(
            z_samples[-1].reshape(-1, z_samples[-1].shape[-1]).cpu().detach().numpy(),
            ranges,
            num_cells,
        )

        ax[1].contourf(grid[:, :, 0], grid[:, :, 1], dvi_vals, cmap=cm.coolwarm)  # type: ignore
        ax[1].set_title("$q_\phi(z_T|z_{0:T-1}, D^c)$")

        for a in ax:
            a.set_yticks([])
            a.set_xticks([])

    plt.show()

    return targets, tp_samples  # type: ignore


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
