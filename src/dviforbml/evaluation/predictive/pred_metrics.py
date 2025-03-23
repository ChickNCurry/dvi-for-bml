import numpy as np
import torch
from torch import Tensor
from torch.distributions.normal import Normal


def compute_mse(y_dist_data: Normal, y_data: Tensor) -> Tensor:
    # (batch_size, num_subtasks, data_size, y_dim)

    y_pred: Tensor = y_dist_data.mean
    # (batch_size, num_subtasks, data_size, y_dim)

    mse: Tensor = torch.median(((y_pred - y_data) ** 2).sum((3, 2)).mean(1))
    # (1)

    return mse


def compute_mse_over_samples(y_dist_data: Normal, y_data: Tensor) -> Tensor:
    # (num_val_tasks, num_samples, data_size, y_dim)

    y_pred = y_dist_data.mean
    # (num_val_tasks, num_samples, data_size, y_dim)

    mse = torch.median(((y_pred - y_data).mean(1) ** 2).sum(-1).mean(1))
    # (1)

    return mse


def compute_lmpl(y_dist_data: Normal, y_data: Tensor) -> Tensor:
    # (batch_size, num_subtasks, data_size, y_dim)

    lmpl: Tensor = y_dist_data.log_prob(y_data)  # type: ignore
    # (batch_size, num_subtasks, data_size, y_dim)

    lmpl = torch.median(
        -np.log(y_data.shape[1]) + torch.logsumexp(lmpl.sum((3, 2)), dim=1)
    )  # (1)

    return lmpl


def compute_lmpl_over_samples(y_dist_data: Normal, y_data: Tensor) -> Tensor:
    # (num_val_tasks, num_samples, data_size, y_dim)

    lmpl: Tensor = y_dist_data.log_prob(y_data)  # type: ignore
    # (num_val_tasks, num_samples, data_size, y_dim)

    lmpl = torch.median(
        -np.log(y_data.shape[1]) + torch.logsumexp(lmpl.sum((3, 2)), dim=1)
    )  # (1)

    return lmpl
