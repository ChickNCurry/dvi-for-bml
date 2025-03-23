from typing import List, Tuple

import numpy as np
import pandas as pd  # type: ignore
import torch
from matplotlib import pyplot as plt
from torch import Tensor

from dviforbml.architectures.np import NP
from dviforbml.evaluation.common import ModelInfo, ModelType
from dviforbml.evaluation.taskposterior.grid import (
    create_grid,
    eval_dist_on_grid,
    eval_hist_on_grid,
)
from dviforbml.evaluation.taskposterior.tp_metrics import compute_bd, compute_jsd


def num_tp_eval(
    model_infos: List[ModelInfo],
    num_samples: int,
    ranges: List[Tuple[float, float]],
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

        if model_info.type == ModelType.CNP or model_info.type == ModelType.LNP:
            continue

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

        jsds = []
        bds = []

        for context_size in context_sizes:
            print(f"Evaluating {model_info.name} for context size {context_size}")

            jsd, bd = num_tp_eval_for_fixed_context_size(
                model_info.model,
                context_size,
                x_data,
                y_data,
                num_samples,
                ranges,
                device,
                model_info.type == ModelType.DVINP,
            )

            jsds.append(jsd)
            bds.append(bd)

        df = pd.DataFrame({"jsd": jsds, "bd": bds}, index=[c for c in context_sizes])
        df.to_csv(f"{save_dir}/tp_metrics_{model_info.name}.csv")

        axs[0].set_title("Jensen–Shannon Divergence", fontsize=12)
        axs[0].set_xlabel("context size", fontsize=8)
        axs[0].plot(context_sizes, jsds, label=model_info.name)
        axs[0].legend(prop={"size": 5})

        axs[1].set_title("Bhattacharyya Distance", fontsize=12)
        axs[1].set_xlabel("context size", fontsize=8)
        axs[1].plot(context_sizes, bds, label=model_info.name)
        axs[1].legend(prop={"size": 5})

    plt.tight_layout()
    plt.savefig(f"{save_dir}/tp_metrics.pdf")
    plt.close()


def num_tp_eval_for_fixed_context_size(
    model: NP,
    context_size: int,
    x_data: Tensor,
    y_data: Tensor,
    num_samples: int,
    ranges: List[Tuple[float, float]],
    device: torch.device,
    enable_grad: bool,
) -> Tuple[float, float]:
    x_context = x_data[:, :, :context_size, :]
    y_context = y_data[:, :, :context_size, :]
    # (num_val_tasks, num_samples, context_size, x_dim)
    # (num_val_tasks, num_samples, context_size, y_dim)

    if enable_grad:
        _, z = model.inference(x_context, y_context, None, x_data)
    else:
        with torch.no_grad():
            _, z = model.inference(x_context, y_context, None, x_data)
    # (num_val_tasks, num_samples, data_size, y_dim)

    tp_samples = z.detach().cpu().numpy()

    sqrt = np.sqrt(num_samples)
    assert sqrt.is_integer()
    num_cells = int(sqrt)
    grid = create_grid(ranges, num_cells)

    target_dist = model.get_target_dist(x_context, y_context, None)

    target_log_probs = eval_dist_on_grid(
        grid, target_dist, device, batch_size=x_data.shape[0]
    )

    jsds = []
    bds = []

    for i in range(tp_samples.shape[0]):
        tp_log_probs = eval_hist_on_grid(tp_samples[i], ranges, num_cells)

        jsd = compute_jsd(target_log_probs[i], tp_log_probs)
        bd = compute_bd(target_log_probs[i], tp_log_probs)

        jsds.append(jsd)
        bds.append(bd)

    jsd = np.median(jsds)
    bd = np.median(bds)

    return jsd, bd
