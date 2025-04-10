from typing import List, Tuple
import pandas as pd
import torch
from torch.utils.data import DataLoader

from dviforbml.architectures.np import NP
from dviforbml.evaluation.predictive.num_pred_eval import (
    num_pred_eval_for_fixed_context_size,
)
from dviforbml.evaluation.taskposterior.num_tp_eval import (
    num_tp_eval_for_fixed_context_size,
)


def num_eval_np(
    model: NP,
    val_loader: DataLoader,
    device: torch.device,
    num_samples: int,
    save_path: str,
    ranges: List[Tuple[float, float]] = [(-6, 6), (-6, 6)],
):
    context_sizes = range(1, 17)

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
    jsds = []
    bds = []

    for context_size in context_sizes:
        lmpl, mse = num_pred_eval_for_fixed_context_size(
            model, context_size, x_data, y_data, True
        )

        jsd, bd = num_tp_eval_for_fixed_context_size(
            model, context_size, x_data, y_data, num_samples, ranges, device, True
        )

        lmpls.append(lmpl)
        mses.append(mse)
        jsds.append(jsd)
        bds.append(bd)

    df = pd.DataFrame(
        {"lmpl": lmpls, "mse": mses, "jsd": jsds, "bd": bds},
        index=[c for c in context_sizes],
    )
    df.to_csv(save_path)
