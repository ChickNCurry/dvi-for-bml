from typing import List, Tuple

import torch
from matplotlib import pyplot as plt
from omegaconf import DictConfig
from torch import Tensor
from torch.distributions import Distribution
from torch.utils.data import DataLoader
import numpy as np
from numpy.typing import NDArray
from matplotlib import cm

from src.components.control import Control
from src.components.decoder import Decoder, LikelihoodTimesPrior
from src.components.dvi_process import DiffusionVIProcess
from src.components.encoder import SetEncoder


def visualize(
    device: torch.device,
    dvi_process: DiffusionVIProcess,
    set_encoder: SetEncoder,
    decoder: Decoder,
    dataloader: DataLoader[Tuple[Tensor, Tensor]],
    control: Control,
    config: DictConfig,
    num_samples: int,
    max_context_size: int,
) -> Tuple[List[Distribution], List[NDArray[np.float64]]]:

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

        aggregated, non_aggregated = set_encoder(context, None)

        p_z_T = LikelihoodTimesPrior(
            decoder=decoder,
            x_target=x_context,
            y_target=y_context,
            mask=None,
            context_embedding=(
                non_aggregated
                if config.decoder.value.is_cross_attentive
                else aggregated
            ),
        )

        targets.append(p_z_T)

        _, z_samples = dvi_process.run_chain(
            p_z_T,
            non_aggregated if control.is_cross_attentive else aggregated,
            None,
        )

        samples.append(z_samples[-1].cpu().detach().numpy())

        y_dist: Distribution = decoder(
            x_data,
            z_samples[-1],
            non_aggregated if control.is_cross_attentive else aggregated,
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


def visualize_vals_on_grid_1d(
    grid: NDArray[np.float64], vals: NDArray[np.float64]
) -> None:
    # (dim1, dim2, ..., z_dim)
    # (dim1, dim2, ...)

    plt.plot(grid[:, 0], vals)
    plt.show()


def visualize_vals_on_grid_2d(
    grid: NDArray[np.float64],
    vals: NDArray[np.float64],
    range: List[Tuple[float, float]] = [(-5, 5), (-5, 5)],
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


def visualize_samples_1d(
    samples: NDArray[np.float64], bins: int = 50, range: Tuple[float, float] = (-5, 5)
) -> None:
    # (num_samples, z_dim)

    plt.hist(samples[:, 0], bins=bins, range=range, density=True)
    plt.show()


def visualize_samples_2d(
    samples: NDArray[np.float64],
    bins: int = 50,
    range: List[Tuple[float, float]] = [(-5, 5), (-5, 5)],
) -> None:
    # (num_samples, z_dim)

    plt.hist2d(samples[:, 0], samples[:, 1], density=True, bins=bins, range=range)
    plt.show()
