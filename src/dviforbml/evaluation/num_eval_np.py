from typing import List, Tuple
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dviforbml.architectures.dvinp import DVINP
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
    old_device: torch.device,
    num_samples: int,
    save_path: str,
    ranges: List[Tuple[float, float]] = [(-6, 6), (-6, 6)],
):
    context_sizes = range(1, 9)

    device = "cpu"
    model.to(device)

    model.freeze()
    model.eval()

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

    for context_size in tqdm(context_sizes):
        lmpl, mse = num_pred_eval_for_fixed_context_size(
            model, context_size, x_data, y_data, True
        )

        lmpls.append(lmpl)
        mses.append(mse)

        if isinstance(model, DVINP):
            jsd, bd = num_tp_eval_for_fixed_context_size(
                model, context_size, x_data, y_data, num_samples, ranges, device, True
            )
        else:
            jsd, bd = 0, 0

        jsds.append(jsd)
        bds.append(bd)

    model.train()
    model.unfreeze()
    model.to(old_device)

    df = pd.DataFrame(
        {"lmpl": lmpls, "mse": mses, "jsd": jsds, "bd": bds},
        index=[c for c in context_sizes],
    )
    df.to_csv(save_path)


def num_eval_np_test(
    model: NP,
    test_loader: DataLoader,
    old_device: torch.device,
    num_samples: int,
    save_path: str,
    ranges: List[Tuple[float, float]] = [(-6, 6), (-6, 6)],
):
    context_sizes = range(1, 9)

    device = "cpu"
    model.to(device)

    model.freeze()
    model.eval()

    avg_lmpls = []
    avg_mses = []
    avg_jsds = []
    avg_bds = []

    for batch in tqdm(test_loader):
        x_data, y_data = batch
        x_data = x_data.to(device)
        y_data = y_data.to(device)
        # (1, data_size, x_dim)
        # (1, data_size, y_dim)

        x_data = x_data.unsqueeze(1).expand(-1, num_samples, -1, -1)
        y_data = y_data.unsqueeze(1).expand(-1, num_samples, -1, -1)
        # (1, num_samples, data_size, x_dim)
        # (1, num_samples, data_size, y_dim)

        lmpls = []
        mses = []
        jsds = []
        bds = []

        for context_size in tqdm(context_sizes):
            lmpl, mse = num_pred_eval_for_fixed_context_size(
                model, context_size, x_data, y_data, True
            )

            lmpls.append(lmpl)
            mses.append(mse)

            if isinstance(model, DVINP):
                jsd, bd = num_tp_eval_for_fixed_context_size(
                    model,
                    context_size,
                    x_data,
                    y_data,
                    num_samples,
                    ranges,
                    device,
                    True,
                )
            else:
                jsd, bd = 0, 0

            jsds.append(jsd)
            bds.append(bd)

        avg_lmpls.append(lmpls)
        avg_mses.append(mses)
        avg_jsds.append(jsds)
        avg_bds.append(bds)

    # avg.. list have dimensions (num_val_tasks, num_context_sizes)
    # now compute median per context size

    lmpls = [np.median(lmpls) for lmpls in zip(*avg_lmpls)]
    mses = [np.median(mses) for mses in zip(*avg_mses)]
    jsds = [np.median(jsds) for jsds in zip(*avg_jsds)]
    bds = [np.median(bds) for bds in zip(*avg_bds)]

    model.train()
    model.unfreeze()
    model.to(old_device)

    df = pd.DataFrame(
        {"lmpl": lmpls, "mse": mses, "jsd": jsds, "bd": bds},
        index=[c for c in context_sizes],
    )

    df.to_csv(save_path)
