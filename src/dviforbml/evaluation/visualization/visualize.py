from typing import List, Tuple
import numpy as np
from matplotlib import cm
from matplotlib import pyplot as plt
from numpy.typing import NDArray


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
