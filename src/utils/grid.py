from typing import Any, List, Tuple

import numpy as np
import torch
from numpy.typing import NDArray
from scipy.special import logsumexp  # type: ignore
from scipy.stats import gaussian_kde  # type: ignore
from torch.distributions import Distribution


def create_grid(
    ranges: List[Tuple[float, float]], num_cells: int
) -> NDArray[np.float32]:
    dims = [np.linspace(min, max, num_cells) for min, max in ranges]
    # (dims, nums)

    grid: NDArray[np.float32] = np.stack(np.meshgrid(*dims), axis=-1).astype(np.float32)
    # (dim1, dim2, ..., z_dim)

    return grid


def eval_kde_on_grid(
    grid: NDArray[np.float32],
    samples: NDArray[np.float32],
) -> NDArray[np.float32]:
    # (dim1, dim2, ..., z_dim), (num_samples, z_dim)

    grid_flat = grid.reshape(-1, grid.shape[-1])
    # (dim1 * dim2 * ..., z_dim)

    kde = gaussian_kde(samples.T)
    vals: NDArray[np.float32] = kde(grid_flat.T)
    vals = vals / np.sum(vals) if np.sum(vals) != 0 else vals
    # (dim1 * dim2 * ...)

    vals = vals.reshape(grid.shape[:-1])
    # (dim1, dim2, ...)

    return vals


def eval_hist_on_grid(
    samples: NDArray[np.float32], ranges: List[Tuple[float, float]], num_cells: int
) -> NDArray[np.float32]:
    # (num_samples, z_dim)

    step_sizes = [(max - min) / num_cells for min, max in ranges]
    outer_edges = [
        (min - step_size, max + step_size)
        for (min, max), step_size in zip(ranges, step_sizes)
    ]

    vals, _ = np.histogramdd(samples, bins=num_cells, range=outer_edges, density=True)
    vals = vals / np.sum(vals) if np.sum(vals) != 0 else vals
    # (dim1, dim2, ...)

    vals = vals.T  # TODO: why transpose needed?

    return vals


def eval_dist_on_grid(
    grid: NDArray[np.float32],
    dist: Distribution,
    batch_size: int,
    sample_size: int,
    device: torch.device,
) -> NDArray[np.float32]:
    # (dim1, dim2, ..., z_dim)

    grid_flat = grid.reshape(-1, grid.shape[-1])
    grid_tensor = torch.from_numpy(grid_flat).float().to(device)
    # (dim1 * dim2 * ..., z_dim)

    grid_tensor = grid_tensor.reshape(batch_size, sample_size, -1)
    # (batch_size, sample_size, z_dim)

    vals = dist.log_prob(grid_tensor).sum(-1).exp().detach().cpu().numpy()
    vals = vals / np.sum(vals) if np.sum(vals) != 0 else vals
    # (dim1 * dim2 * ...)

    vals = vals.reshape(grid.shape[:-1])
    # (dim1, dim2, ...)

    return vals


def sample_from_vals(
    grid: NDArray[np.float32], vals: NDArray[np.float32], num_samples: int
) -> NDArray[np.float32]:
    # (dim1, dim2, ...)

    flat_vals = vals.reshape(-1)
    # (dim1 * dim2 * ...)

    flat_indices = np.random.choice(flat_vals.shape[0], size=num_samples, p=flat_vals)
    # (num_samples)

    grid_indices = np.unravel_index(flat_indices, vals.shape)
    samples = grid[grid_indices]
    # (num_samples, z_dim)

    return samples


def compute_jsd(
    p_vals: NDArray[np.float32],
    q_vals: NDArray[np.float32],
) -> Any:
    # (dim1, dim2, ...)
    # (dim1, dim2, ...)

    eps = 1e-10

    m_vals = 0.5 * (p_vals + q_vals) + eps

    jsd = 0.5 * np.sum(
        p_vals * np.log((p_vals / m_vals) + eps)
        + q_vals * np.log((q_vals / m_vals) + eps)
    )

    return jsd


def compute_bd(
    p_vals: NDArray[np.float32],
    q_vals: NDArray[np.float32],
) -> Any:
    # (dim1, dim2, ...)
    # (dim1, dim2, ...)

    eps = 1e-10

    bc = np.sum(np.sqrt(p_vals * q_vals))
    bd = -np.log(bc + eps)

    return bd


def compute_lmpl(
    grid: NDArray[np.float32],
    dist: Distribution,
    device: torch.device,
) -> Any:
    # (dim1, dim2, ..., z_dim)

    grid_flat = grid.reshape(-1, grid.shape[-1])
    grid_tensor = torch.from_numpy(grid_flat).float().to(device)
    # (dim1 * dim2 * ..., z_dim)

    log_probs = dist.log_prob(grid_tensor).sum(-1).detach().cpu().numpy()
    # (dim1 * dim2 * ...)

    lmpl = logsumexp(log_probs)  # scales with context size

    return lmpl
