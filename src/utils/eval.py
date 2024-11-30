from typing import Any, List, Tuple

import numpy as np
import torch
from numpy.typing import NDArray
from scipy.stats import gaussian_kde  # type: ignore
from torch.distributions import Distribution


def create_grid(intervals: List[Tuple[float, float]], num: int) -> NDArray[np.float32]:
    dims = [np.linspace(min, max, num) for min, max in intervals]
    # (dims, nums)

    grid: NDArray[np.float32] = np.stack(np.meshgrid(*dims), axis=-1)
    # (dim1, dim2, ..., z_dim)

    return grid


def eval_kde_on_grid(
    grid: NDArray[np.float32],
    samples: NDArray[np.float32],
    bw_method: str | None = None,
) -> NDArray[np.float32]:
    # (dim1, dim2, ..., z_dim), (num_samples, z_dim)

    grid_flat = grid.reshape(-1, grid.shape[-1])
    # (dim1 * dim2 * ..., z_dim)

    kde = gaussian_kde(samples.T, bw_method=bw_method)
    vals: NDArray[np.float32] = kde(grid_flat.T)
    # (dim1 * dim2 * ...)

    return vals


def eval_hist_on_grid(
    samples: NDArray[np.float32], intervals: List[Tuple[float, float]], num: int
) -> NDArray[np.float32]:
    # (num_samples, z_dim)

    step_sizes = [(max - min) / num for min, max in intervals]
    outer_edges = [
        (min - step_size, max + step_size)
        for (min, max), step_size in zip(intervals, step_sizes)
    ]

    vals, _ = np.histogramdd(samples, bins=num, range=outer_edges, density=True)
    # (dim1, dim2, ...)

    vals = vals.T.reshape(-1).astype(np.float32)
    # (dim1 * dim2 * ...)

    return vals


def eval_dist_on_grid(
    grid: NDArray[np.float32], dist: Distribution, device: torch.device
) -> NDArray[np.float32]:
    # (dim1, dim2, ..., z_dim)

    grid_flat = grid.reshape(-1, grid.shape[-1])
    # (dim1 * dim2 * ..., z_dim)

    grid_tensor = torch.from_numpy(grid_flat).float().to(device)
    # (dim1 * dim2 * ..., z_dim)

    vals = dist.log_prob(grid_tensor).sum(-1).exp().detach().cpu().numpy()
    # (dim1 * dim2 * ...)

    return vals


def normalize_vals_on_grid(
    vals: NDArray[np.float32], intervals: List[Tuple[float, float]], num: int
) -> NDArray[np.float32]:
    # (dim1 * dim2 * ...)

    spacings = [(max - min) / num for min, max in intervals]
    normalizer = vals.sum() * np.prod(spacings)

    vals = vals / normalizer if normalizer != 0 else vals

    return vals


def compute_jsd(p_vals: NDArray[np.float32], q_vals: NDArray[np.float32]) -> Any:
    # (dim1 * dim2 * ...)
    # (dim1 * dim2 * ...)

    eps = 1e-10
    p_vals = p_vals + eps
    q_vals = q_vals + eps

    m_vals = 0.5 * (p_vals + q_vals)
    jsd = 0.5 * np.sum(
        p_vals * (np.log(p_vals) - np.log(m_vals))
        + q_vals * (np.log(q_vals) - np.log(m_vals))
    )

    return jsd


def compute_bd(p_vals: NDArray[np.float32], q_vals: NDArray[np.float32]) -> Any:
    # (dim1 * dim2 * ...)
    # (dim1 * dim2 * ...)

    p_vals = p_vals / np.sum(p_vals) if p_vals.sum() != 0 else p_vals
    q_vals = q_vals / np.sum(q_vals) if q_vals.sum() != 0 else q_vals

    bc = np.sum(np.sqrt(p_vals * q_vals))
    bd = -np.log(bc)

    return bd
