from typing import Any, Tuple

import pandas as pd  # type: ignore
import torch
from torch.utils.data import DataLoader

from src.architectures.np import NP
from src.eval.metrics import compute_lmpl, compute_mse
from src.utils.datasets import MetaLearningDataset


def eval_np_for_fixed_context_size(
    model: NP,
    val_loader: DataLoader[Any],
    context_size: int,
    num_samples: int,
    device: torch.device,
) -> Tuple[float, float]:
    x_data, y_data = next(iter(val_loader))

    x_data = x_data.to(device)
    y_data = y_data.to(device)
    # (num_val_tasks, data_size, x_dim)
    # (num_val_tasks, data_size, y_dim)

    x_data = x_data.expand(1, num_samples, -1, -1)
    y_data = y_data.expand(1, num_samples, -1, -1)
    # (num_val_tasks, num_samples, data_size, x_dim)
    # (num_val_tasks, num_samples, data_size, y_dim)

    data = torch.cat([x_data, y_data], dim=-1)
    # (num_val_tasks, num_samples, data_size, x_dim + y_dim)

    x_context = x_data[:, :, :context_size, :]
    y_context = y_data[:, :, :context_size, :]
    # (num_val_tasks, num_samples, context_size, x_dim)
    # (num_val_tasks, num_samples, context_size, y_dim)

    y_dist_data = model.inference(x_context, y_context, data, x_data)
    # (num_val_tasks, num_samples, data_size, y_dim)

    lmpl = compute_lmpl(y_dist_data, y_data)
    mse = compute_mse(y_dist_data, y_data)

    return lmpl.item(), mse.item()


def eval_np(
    model: NP, val_loader: DataLoader[Any], num_samples: int, device: torch.device
) -> pd.DataFrame:
    assert isinstance(val_loader.dataset, MetaLearningDataset)

    context_sizes = range(1, val_loader.dataset.max_context_size + 1)

    lmpls = []
    mses = []

    for context_size in context_sizes:
        lmpl, mse = eval_np_for_fixed_context_size(
            model, val_loader, context_size, num_samples, device
        )
        lmpls.append(lmpl)
        mses.append(mse)

    df = pd.DataFrame({"lmpl": lmpls, "mse": mses}, index=[c for c in context_sizes])

    return df
