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

from src.components.cdvi import ContextualDVI
from src.components.decoder import LikelihoodTimesPrior
from src.utils.grid import (
    create_grid,
    eval_dist_on_grid,
    eval_hist_on_grid,
    eval_kde_on_grid,
    sample_from_vals,
)


def visualize_cdvi_for_bml(
    device: torch.device,
    cdvi: ContextualDVI,
    dataloader: DataLoader[Tuple[Tensor, Tensor]],
    config: DictConfig,
    num_samples: int,
    max_context_size: int,
) -> Tuple[List[Distribution], List[NDArray[np.float32]]]:

    x_data, y_data = next(iter(dataloader))
    x_data, y_data = x_data.to(device), y_data.to(device)
    x_data, y_data = x_data.expand(num_samples, -1, -1), y_data.expand(
        num_samples, -1, -1
    )

    x_data_sorted, indices = x_data.sort(dim=1)
    indices = indices[1].squeeze(1)
    x_data_sorted = x_data_sorted.cpu().detach().numpy()
    y_data_sorted = y_data[:, indices, :].cpu().detach().numpy()

    fig = plt.figure(figsize=(9, 3 * max_context_size), constrained_layout=True)
    subfigs = fig.subfigures(nrows=max_context_size, ncols=1)

    targets = []
    samples = []

    for row, subfig in enumerate(subfigs):
        subfig.suptitle(f"Context Size: {row + 1}")
        ax = subfig.subplots(nrows=1, ncols=2, width_ratios=[3, 1])

        context_size = row + 1
        x_context = x_data[:, :context_size, :]
        y_context = y_data[:, :context_size, :]
        context = torch.cat([x_context, y_context], dim=-1)

        aggregated, non_aggregated = cdvi.encoder(context, None)

        p_z_T = LikelihoodTimesPrior(
            decoder=cdvi.decoder,
            x_target=x_context,
            y_target=y_context,
            mask=None,
            context_embedding=(
                non_aggregated if config.decoder.is_cross_attentive else aggregated
            ),
        )

        targets.append(p_z_T)

        _, _, z_samples = cdvi.dvi_process.run_chain(
            p_z_T,
            (
                non_aggregated
                if cdvi.dvi_process.control.is_cross_attentive
                else aggregated
            ),
            None,
        )

        samples.append(z_samples[-1].cpu().detach().numpy())

        y_dist: Distribution = cdvi.decoder(
            x_data,
            z_samples[-1],
            (
                non_aggregated
                if cdvi.dvi_process.control.is_cross_attentive
                else aggregated
            ),
            None,
        )

        y_mu_sorted = y_dist.mean[:, indices, :].cpu().detach().numpy()

        for k in range(num_samples):
            ax[0].plot(
                x_data_sorted[k].squeeze(1),
                y_mu_sorted[k].squeeze(1),
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

        ax[1].hist2d(
            z_samples[-1][:, 0].cpu().detach().numpy(),
            z_samples[-1][:, 1].cpu().detach().numpy(),
            bins=50,
            range=((-3, 3), (-3, 3)),
        )

    plt.show()

    return targets, samples  # type: ignore


def visualize_cdvi_for_bml_test(
    device: torch.device,
    cdvi: ContextualDVI,
    dataloader: DataLoader[Tuple[Tensor, Tensor]],
    config: DictConfig,
    num_samples: int,
    max_context_size: int,
    ranges: List[Tuple[float, float]],
    num_cells: int,
) -> Tuple[List[Distribution], List[NDArray[np.float32]]]:

    x_data, y_data = next(iter(dataloader))
    x_data, y_data = x_data.to(device), y_data.to(device)
    x_data, y_data = x_data.expand(num_samples, -1, -1), y_data.expand(
        num_samples, -1, -1
    )

    x_data_sorted, indices = x_data.sort(dim=1)
    indices = indices[1].squeeze(1)
    x_data_sorted = x_data_sorted.cpu().detach().numpy()
    y_data_sorted = y_data[:, indices, :].cpu().detach().numpy()

    fig = plt.figure(figsize=(12, 3 * max_context_size * 2), constrained_layout=True)
    subfigs = fig.subfigures(nrows=max_context_size, ncols=1)

    targets = []
    samples = []

    for row, subfig in enumerate(subfigs):
        subfig.suptitle(f"Context Size: {row + 1}")
        ax = subfig.subplots(nrows=2, ncols=2, width_ratios=[3, 1])

        context_size = row + 1
        x_context = x_data[:, :context_size, :]
        y_context = y_data[:, :context_size, :]
        context = torch.cat([x_context, y_context], dim=-1)

        aggregated, non_aggregated = cdvi.encoder(context, None)

        p_z_T = LikelihoodTimesPrior(
            decoder=cdvi.decoder,
            x_target=x_context,
            y_target=y_context,
            mask=None,
            context_embedding=(
                non_aggregated if config.decoder.is_cross_attentive else aggregated
            ),
        )

        targets.append(p_z_T)

        _, _, z_samples = cdvi.dvi_process.run_chain(
            p_z_T,
            (
                non_aggregated
                if cdvi.dvi_process.control.is_cross_attentive
                else aggregated
            ),
            None,
        )

        samples.append(z_samples[-1].cpu().detach().numpy())

        y_dist: Distribution = cdvi.decoder(
            x_data,
            z_samples[-1],
            (
                non_aggregated
                if cdvi.dvi_process.control.is_cross_attentive
                else aggregated
            ),
            None,
        )

        y_mu_sorted = y_dist.mean[:, indices, :].cpu().detach().numpy()

        for k in range(num_samples):
            ax[0][0].plot(
                x_data_sorted[k].squeeze(1),
                y_mu_sorted[k].squeeze(1),
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

        ax[0][0].set_title("Predictions from Task Posterior")

        grid = create_grid(ranges, num_cells)

        # dvi_vals = eval_kde_on_grid(grid, z_samples[-1].cpu().detach().numpy())
        dvi_vals = eval_hist_on_grid(
            z_samples[-1].cpu().detach().numpy(), ranges, num_cells
        )

        target_vals = eval_dist_on_grid(grid, p_z_T, device=device)
        target_samples = sample_from_vals(grid, target_vals, num_samples)
        target_samples = torch.from_numpy(target_samples).to(device)

        y_dist_test: Distribution = cdvi.decoder(
            x_data,
            target_samples,
            (
                non_aggregated
                if cdvi.dvi_process.control.is_cross_attentive
                else aggregated
            ),
            None,
        )

        y_mu_test_sorted = y_dist_test.mean[:, indices, :].cpu().detach().numpy()

        for k in range(num_samples):
            ax[1][0].plot(
                x_data_sorted[k].squeeze(1),
                y_mu_test_sorted[k].squeeze(1),
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

        ax[1][0].set_title("Predictions from Likelihood Times Prior")

        ax[0][1].contourf(grid[:, :, 0], grid[:, :, 1], dvi_vals, cmap=cm.coolwarm)  # type: ignore
        ax[1][1].contourf(grid[:, :, 0], grid[:, :, 1], target_vals, cmap=cm.coolwarm)  # type: ignore

        ax[0][1].set_title("Task Posterior")
        ax[1][1].set_title("Likelihood Times Prior")

    plt.show()

    return targets, samples  # type: ignore


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
