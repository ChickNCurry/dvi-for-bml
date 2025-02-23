from typing import Any, Tuple

import pandas as pd  # type: ignore
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from src.evaluation.predictive.metrics import (
    compute_lmpl,
    compute_lmpl_over_samples,
    compute_mse,
    compute_mse_over_samples,
)
from src.architectures.np import NP
from src.utils.datasets import MetaLearningDataset


def eval_np_for_fixed_context_size_test(
    model: NP, context_size: int, x_data: Tensor, y_data: Tensor, device: torch.device
) -> Tuple[float, float]:
    data = torch.cat([x_data, y_data], dim=-1)
    # (batch_size, data_size, x_dim + y_dim)

    # context_sizes = (
    #     torch.ones(
    #         size=(data.shape[0], data.shape[1], 1),
    #         device=device,
    #     )
    #     * context_size
    # )
    # # (batch_size, num_subtasks, 1)

    # pos_indices = (
    #     torch.arange(data.shape[2], device=device)
    #     .unsqueeze(0)
    #     .unsqueeze(0)
    #     .expand(data.shape[0], data.shape[1], -1)
    # )  # (batch_size, sample_size, data_size)

    rand_context_sizes = torch.randint(
        low=1,
        high=data.shape[2] + 1,
        size=(data.shape[0], data.shape[1], 1),
        device=device,
        # generator=self.generator,
    )  # (batch_size, num_subtasks, 1)

    pos_indices = (
        torch.arange(data.shape[2], device=device)
        .unsqueeze(0)
        .unsqueeze(0)
        .expand(data.shape[0], data.shape[1], -1)
    )  # (batch_size, num_subtasks, data_size)

    mask = (pos_indices < rand_context_sizes).float()
    # (batch_size, sample_size, data_size)

    context = data * mask.unsqueeze(-1).expand(-1, -1, -1, data.shape[-1])
    # (batch_size, num_subtasks, data_size, x_dim + y_dim)

    x_context = context[:, :, :, 0 : x_data.shape[-1]]
    y_context = context[:, :, :, x_data.shape[-1] : data.shape[-1]]
    # (batch_size, num_subtasks, data_size, x_dim)
    # (batch_size, num_subtasks, data_size, y_dim)

    with torch.no_grad():

        y_dist_data, _ = model.inference(x_context, y_context, mask, x_data)
        # (num_val_tasks, num_samples, data_size, y_dim)

    lmpl = compute_lmpl(y_dist_data, y_data)
    mse = compute_mse(y_dist_data, y_data)

    del x_context, y_context, y_dist_data

    return lmpl.item(), mse.item()


def eval_np_for_fixed_context_size(
    model: NP,
    context_size: int,
    x_data: Tensor,
    y_data: Tensor,
) -> Tuple[float, float]:
    x_context = x_data[:, :, :context_size, :]
    y_context = y_data[:, :, :context_size, :]
    # (num_val_tasks, num_samples, context_size, x_dim)
    # (num_val_tasks, num_samples, context_size, y_dim)

    with torch.no_grad():

        y_dist_data, _ = model.inference(x_context, y_context, None, x_data)
        # (num_val_tasks, num_samples, data_size, y_dim)

    lmpl = compute_lmpl_over_samples(y_dist_data, y_data)
    mse = compute_mse_over_samples(y_dist_data, y_data)

    del x_context, y_context, y_dist_data

    return lmpl.item(), mse.item()


def eval_np(
    model: NP, val_loader: Any, num_samples: int, device: torch.device
) -> pd.DataFrame:
    assert isinstance(val_loader.dataset.dataset, MetaLearningDataset)

    context_sizes = range(1, val_loader.dataset.dataset.max_context_size + 1)

    x_data, y_data = next(iter(val_loader))

    x_data = x_data.to(device)
    y_data = y_data.to(device)
    # (num_val_tasks, data_size, x_dim)
    # (num_val_tasks, data_size, y_dim)

    x_data = x_data.unsqueeze(1).expand(-1, num_samples, -1, -1)
    y_data = y_data.unsqueeze(1).expand(-1, num_samples, -1, -1)
    # (num_val_tasks, num_samples, data_size, x_dim)
    # (num_val_tasks, num_samples, data_size, y_dim)

    lmpls = []
    mses = []

    for context_size in context_sizes:
        print(f"Evaluating for context size {context_size}")

        # lmpl, mse = eval_np_for_fixed_context_size_test(
        #     model, context_size, x_data, y_data, device
        # )

        lmpl, mse = eval_np_for_fixed_context_size(model, context_size, x_data, y_data)

        lmpls.append(lmpl)
        mses.append(mse)

    df = pd.DataFrame({"lmpl": lmpls, "mse": mses}, index=[c for c in context_sizes])

    return df
