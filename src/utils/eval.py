import torch
from torch import Tensor
from typing import Any, List, Tuple
import numpy as np
from numpy.typing import NDArray

from scipy.stats import gaussian_kde  # type: ignore
from matplotlib import pyplot as plt
from matplotlib import cm
from torch.distributions import Distribution


def create_grid(mins: List[float], maxs: List[float], num: int) -> NDArray[np.float64]:
    dims = [np.linspace(min, max, num) for min, max in zip(mins, maxs)]
    # (dims, nums)

    grid: NDArray[np.float64] = np.stack(np.meshgrid(*dims), axis=-1)
    # (dim1, dim2, ..., z_dim)

    return grid


def eval_kde_on_grid(
    grid: NDArray[np.float64], samples: NDArray[np.float64]
) -> NDArray[np.float64]:
    # (x_grid, y_grid, ..., z_dim), (num_samples, z_dim)

    kde = gaussian_kde(samples.T)

    grid_flat = grid.reshape(-1, grid.shape[-1]).T
    # (z_dim, x_grid * y_grid * ...)

    vals: NDArray[np.float64] = kde(grid_flat).reshape(grid.shape[: grid.shape[-1]])
    # (dim1, dim2, ...)

    return vals


def eval_dist_on_grid(
    grid: NDArray[np.float64], dist: Distribution
) -> NDArray[np.float64]:
    # (x_grid, y_grid, ..., z_dim)

    grid_tensor = torch.from_numpy(grid)
    # (dim1, dim2, ..., z_dim)

    vals: NDArray[np.float64] = dist.log_prob(grid_tensor).sum(dim=-1).exp().numpy()
    # (dim1, dim2, ...)

    return vals


def normalize_vals(
    vals: NDArray[np.float64], mins: List[float], maxs: List[float], num: int
) -> NDArray[np.float64]:
    # (dim1, dim2, ...)

    spacings = [(max - min) / num for min, max in zip(mins, maxs)]

    vals = vals / (vals.sum() * np.prod(spacings))

    return vals


def visualize_vals_on_grid_2d(
    grid: NDArray[np.float64], vals: NDArray[np.float64]
) -> None:
    # (dim1, dim2, ..., z_dim)
    # (dim1, dim2, ...)

    plt.contourf(grid[:, :, 0], grid[:, :, 1], vals, cmap=cm.coolwarm)  # type: ignore
    plt.colorbar()
    plt.show()

    ax = plt.axes(projection="3d")
    ax.plot_surface(grid[:, :, 0], grid[:, :, 1], vals, cmap=cm.coolwarm)  # type: ignore
    plt.show()


def visualize_vals_on_grid_1d(
    grid: NDArray[np.float64], vals: NDArray[np.float64]
) -> None:
    # (dim1, dim2, ...)

    plt.plot(grid[:, 0], vals)
    plt.show()


def visualize_samples_2d(
    samples: NDArray[np.float64],
    bins: int = 50,
    range: List[Tuple[float, float]] = [(-5, 5), (-5, 5)],
) -> None:
    # (num_samples, z_dim)

    plt.hist2d(samples[:, 0], samples[:, 1], density=True, bins=bins, range=range)
    plt.show()


def visualize_samples_1d(
    samples: NDArray[np.float64], bins: int = 50, range: Tuple[float, float] = (-5, 5)
) -> None:
    # (num_samples, z_dim)

    plt.hist(samples[:, 0], bins=bins, range=range, density=True)
    plt.show()


def compute_jsd(p_vals: NDArray[np.float64], q_vals: NDArray[np.float64]) -> Any:
    # (dim1, dim2, ...)
    # (dim1, dim2, ...)

    eps = 1e-10
    p_vals = p_vals + eps
    q_vals = q_vals + eps

    m_vals = 0.5 * (p_vals + q_vals)
    jsd = 0.5 * np.sum(
        p_vals * (np.log(p_vals) - np.log(m_vals))
        + q_vals * (np.log(q_vals) - np.log(m_vals))
    )

    return jsd


def compute_bd(p_vals: NDArray[np.float64], q_vals: NDArray[np.float64]) -> Any:
    # (dim1, dim2, ...)
    # (dim1, dim2, ...)

    p_vals = p_vals / np.sum(p_vals)
    q_vals = q_vals / np.sum(q_vals)

    bc = np.sum(np.sqrt(p_vals * q_vals))
    bd = -np.log(bc)

    return bd
