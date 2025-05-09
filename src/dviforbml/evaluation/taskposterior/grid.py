from typing import List, Tuple

import numpy as np
import torch
from numpy.typing import NDArray
from scipy.special import logsumexp
from scipy.stats import gaussian_kde
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
    probs: NDArray[np.float32] = kde(grid_flat.T)
    log_probs = np.log(np.clip(probs, a_min=1e-300, a_max=None))
    log_probs = log_probs - logsumexp(log_probs)
    # (dim1 * dim2 * ...)

    log_probs = log_probs.reshape(grid.shape[:-1])
    # (dim1, dim2, ...)

    return log_probs


def eval_hist_on_grid(
    samples: NDArray[np.float32], ranges: List[Tuple[float, float]], num_cells: int
) -> NDArray[np.float32]:
    # (num_samples, z_dim)

    step_sizes = [(max - min) / num_cells for min, max in ranges]
    outer_edges = [
        (min - step_size, max + step_size)
        for (min, max), step_size in zip(ranges, step_sizes)
    ]

    probs, _ = np.histogramdd(samples, bins=num_cells, range=outer_edges, density=True)

    if np.isnan(probs).any():
        probs = np.nan_to_num(probs)

    log_probs: NDArray[np.float32] = np.log(np.clip(probs, a_min=1e-300, a_max=None))
    log_probs = log_probs - logsumexp(log_probs)
    # (dim1, dim2, ...)

    log_probs = log_probs.T  # NEEDED

    return log_probs


def eval_dist_on_grid(
    grid: NDArray[np.float32],
    dist: Distribution,
    device: torch.device,
    batch_size: int = 1,
) -> NDArray[np.float32]:
    # (dim1, dim2, ..., z_dim)

    grid_flat = grid.reshape(-1, grid.shape[-1])
    grid_tensor = (
        torch.from_numpy(grid_flat)
        .float()
        .to(device)
        .unsqueeze(0)
        .expand(batch_size, -1, -1)
    )  # (batch_size, dim1 * dim2 * ..., z_dim)

    log_probs = dist.log_prob(grid_tensor).sum(-1).detach().cpu().numpy()
    log_probs = log_probs - logsumexp(log_probs)
    # (batch_size, dim1 * dim2 * ...)

    log_probs = log_probs.reshape(log_probs.shape[0], *grid.shape[:-1])
    # (batch_size, dim1, dim2, ...)

    return log_probs


def eval_score_on_grid(
    grid: NDArray[np.float32],
    dist: Distribution,
    device: torch.device,
) -> NDArray[np.float32]:
    # (dim1, dim2, ..., z_dim)

    grid_flat = grid.reshape(-1, grid.shape[-1])
    grid_tensor = torch.from_numpy(grid_flat).float().to(device).unsqueeze(0)
    # (1, dim1 * dim2 * ..., z_dim)

    grid_tensor = grid_tensor.requires_grad_(True)

    log_prob = dist.log_prob(grid_tensor)

    grad = (
        torch.autograd.grad(
            outputs=log_prob,
            inputs=grid_tensor,
            grad_outputs=torch.ones_like(log_prob),
        )[0]
        .detach()
        .cpu()
        .numpy()
    )  # (batch_size, dim1 * dim2 * ..., z_dim)

    grad = grad.reshape(grid.shape)
    # (batch_size, dim1, dim2, ..., z_dim)

    return grad


def sample_from_log_probs(
    grid: NDArray[np.float32], log_probs: NDArray[np.float32], num_samples: int
) -> NDArray[np.float32]:
    # (dim1, dim2, ...)

    flat_log_probs = log_probs.reshape(-1)
    flat_probs = np.exp(flat_log_probs)
    # (dim1 * dim2 * ...)

    flat_indices = np.random.choice(
        flat_log_probs.shape[0], size=num_samples, p=flat_probs
    )
    # (num_samples)

    grid_indices = np.unravel_index(flat_indices, log_probs.shape)
    samples = grid[grid_indices]
    # (num_samples, z_dim)

    return samples
