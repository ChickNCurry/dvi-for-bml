from typing import List, Tuple

import pandas as pd  # type: ignore
import torch
from matplotlib import pyplot as plt
from torch import Tensor

from src.architectures.np import NP
from src.evaluation.common import ModelInfo
from src.evaluation.predictive.pred_metrics import (
    compute_lmpl_over_samples,
    compute_mse_over_samples,
)
from src.utils.datasets import MetaLearningDataset


def num_pred_eval(
    model_infos: List[ModelInfo],
    num_samples: int,
    device: torch.device,
    save_dir: str,
) -> None:

    num_metrics = 2

    _, axs = plt.subplots(
        nrows=1,
        ncols=num_metrics,
        figsize=(4 * num_metrics, 3),
    )

    for model_info in model_infos:
        # assert isinstance(model_info.val_loader.dataset.dataset, MetaLearningDataset)
        # context_sizes = range(1, model_info.val_loader.dataset.dataset.max_context_size + 1)

        context_sizes = range(1, 17)

        x_data, y_data = next(iter(model_info.val_loader))

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
            print(f"Evaluating {model_info.name} for context size {context_size}")

            lmpl, mse = num_pred_eval_for_fixed_context_size(
                model_info.model, context_size, x_data, y_data
            )

            # lmpl, mse = num_pred_eval_for_fixed_context_size_mask(
            #     model, context_size, x_data, y_data, device
            # )

            lmpls.append(lmpl)
            mses.append(mse)

        df = pd.DataFrame(
            {"lmpl": lmpls, "mse": mses}, index=[c for c in context_sizes]
        )
        df.to_csv(f"{save_dir}/pred_metrics_{model_info.name}.csv")

        axs[0].set_title("Log Marginal Predictive Likelihood (LMPL)", fontsize=12)
        axs[0].set_xlabel("context size", fontsize=8)
        axs[0].plot(context_sizes, lmpls, label=model_info.name)
        axs[0].legend(prop={"size": 5})
        axs[0].grid(True)

        axs[1].set_title("Mean Squared Error (MSE)", fontsize=12)
        axs[1].set_xlabel("context size", fontsize=8)
        axs[1].plot(context_sizes, mses, label=model_info.name)
        axs[1].legend(prop={"size": 5})
        axs[1].grid(True)

    plt.tight_layout()
    plt.savefig(f"{save_dir}/pred_metrics.pdf")
    plt.close()


def num_pred_eval_for_fixed_context_size(
    model: NP,
    context_size: int,
    x_data: Tensor,
    y_data: Tensor,
) -> Tuple[float, float]:
    x_context = x_data[:, :, :context_size, :]
    y_context = y_data[:, :, :context_size, :]
    # (num_val_tasks, num_samples, context_size, x_dim)
    # (num_val_tasks, num_samples, context_size, y_dim)

    # with torch.no_grad():
    y_dist_data, _ = model.inference(x_context, y_context, None, x_data)
    # (num_val_tasks, num_samples, data_size, y_dim)

    lmpl = compute_lmpl_over_samples(y_dist_data, y_data)
    mse = compute_mse_over_samples(y_dist_data, y_data)

    del x_context, y_context, y_dist_data

    return lmpl.item(), mse.item()


def num_pred_eval_for_fixed_context_size_mask(
    model: NP, context_size: int, x_data: Tensor, y_data: Tensor, device: torch.device
) -> Tuple[float, float]:
    data = torch.cat([x_data, y_data], dim=-1)
    # (batch_size, data_size, x_dim + y_dim)

    context_sizes = (
        torch.ones(
            size=(data.shape[0], data.shape[1], 1),
            device=device,
        )
        * context_size
    )  # (batch_size, num_subtasks, 1)

    pos_indices = (
        torch.arange(data.shape[2], device=device)
        .unsqueeze(0)
        .unsqueeze(0)
        .expand(data.shape[0], data.shape[1], -1)
    )  # (batch_size, sample_size, data_size)

    mask = (pos_indices < context_sizes).float()
    # (batch_size, sample_size, data_size)

    context = data * mask.unsqueeze(-1).expand(-1, -1, -1, data.shape[-1])
    # (batch_size, num_subtasks, data_size, x_dim + y_dim)

    x_context = context[:, :, :, 0 : x_data.shape[-1]]
    y_context = context[:, :, :, x_data.shape[-1] : data.shape[-1]]
    # (batch_size, num_subtasks, data_size, x_dim)
    # (batch_size, num_subtasks, data_size, y_dim)

    # with torch.no_grad():
    y_dist_data, _ = model.inference(x_context, y_context, mask, x_data)
    # (num_val_tasks, num_samples, data_size, y_dim)

    lmpl = compute_lmpl_over_samples(y_dist_data, y_data)
    mse = compute_mse_over_samples(y_dist_data, y_data)

    del x_context, y_context, y_dist_data

    return lmpl.item(), mse.item()
