import torch
from typing import Any, List
import numpy as np
from numpy.typing import NDArray

from scipy.stats import gaussian_kde  # type: ignore
from torch.distributions import Distribution


def create_grid(mins: List[float], maxs: List[float], num: int) -> NDArray[np.float32]:
    dims = [np.linspace(min, max, num) for min, max in zip(mins, maxs)]
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

    kde = gaussian_kde(samples.T, bw_method=bw_method)

    grid_flat = grid.reshape(-1, grid.shape[-1]).T
    # (z_dim, dim1 * dim2 * ...)

    vals: NDArray[np.float32] = kde(grid_flat)
    vals = vals.reshape(grid.shape[: grid.shape[-1]])
    # (dim1, dim2, ...)

    return vals


def eval_dist_on_grid(
    grid: NDArray[np.float32], dist: Distribution
) -> NDArray[np.float32]:
    # (dim1, dim2, ..., z_dim)

    grid_flat = grid.reshape(-1, grid.shape[-1])
    grid_tensor = torch.from_numpy(grid_flat).float()
    # (z_dim, dim1 * dim2 * ...)

    log_prob = getattr(dist, "log_prob_test", None)

    if log_prob is not None:
        vals: NDArray[np.float32] = (
            log_prob(grid_tensor).sum(dim=-1).exp().detach().cpu().numpy()
        )
    else:
        vals = dist.log_prob(grid_tensor).sum(dim=-1).exp().detach().cpu().numpy()

    vals = vals.reshape(grid.shape[: grid.shape[-1]])
    # (dim1, dim2, ...)

    return vals


def normalize_vals_on_grid(
    vals: NDArray[np.float32], mins: List[float], maxs: List[float], num: int
) -> NDArray[np.float32]:
    # (dim1, dim2, ...)

    spacings = [(max - min) / num for min, max in zip(mins, maxs)]

    normalizer = vals.sum() * np.prod(spacings)

    vals = vals / normalizer if normalizer != 0 else vals

    return vals


def compute_jsd(p_vals: NDArray[np.float32], q_vals: NDArray[np.float32]) -> Any:
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


def compute_bd(p_vals: NDArray[np.float32], q_vals: NDArray[np.float32]) -> Any:
    # (dim1, dim2, ...)
    # (dim1, dim2, ...)

    p_vals = p_vals / np.sum(p_vals)
    q_vals = q_vals / np.sum(q_vals)

    bc = np.sum(np.sqrt(p_vals * q_vals))
    bd = -np.log(bc)

    return bd
