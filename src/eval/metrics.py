import numpy as np
import torch
from torch import Tensor

from src.components.decoder.decoder import Decoder


def compute_mse(
    decoder: Decoder,
    z: Tensor,
    x_data: Tensor,
    y_data: Tensor,
) -> Tensor:
    # (batch_size, num_subtasks, z_dim)
    # (batch_size, num_subtasks, target_size, x_dim)
    # (batch_size, num_subtasks, target_size, y_dim)

    y_pred: Tensor = decoder(z, x_data).mean
    # (batch_size, num_subtasks, target_size, y_dim)

    mse: Tensor = torch.median(((y_pred - y_data) ** 2).sum(3).sum(2).mean(1))
    # (1)

    return mse


def compute_lmpl(
    decoder: Decoder,
    z: Tensor,
    x_data: Tensor,
    y_data: Tensor,
) -> Tensor:
    # (batch_size, num_subtasks, z_dim)
    # (batch_size, num_subtasks, target_size, x_dim)
    # (batch_size, num_subtasks, target_size, y_dim)

    lmpl: Tensor = decoder(z, x_data).log_prob(y_data)
    # (batch_size, num_subtasks, target_size, y_dim)

    lmpl = torch.median(
        torch.logsumexp(lmpl.sum(3).sum(2), dim=1) - np.log(x_data.shape[1])
    )  # (1)

    return lmpl
