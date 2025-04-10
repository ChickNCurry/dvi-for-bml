from typing import Tuple

import torch
from matplotlib import pyplot as plt
from torch import Tensor
from torch.utils.data import DataLoader

from dviforbml.architectures.cnp import CNP
from dviforbml.architectures.lnp import LNP
from dviforbml.utils.datasets import hash_tensor


def visualize_np(
    model: CNP | LNP,
    device: torch.device,
    dataloader: DataLoader[Tuple[Tensor, Tensor]],
    num_samples: int,
    max_context_size: int,
    show_sigma: bool = False,
) -> None:
    x_data, y_data = next(iter(dataloader))
    # task_hash = hash_tensor(x_data)  # + hash_tensor(y_data)

    x_data = x_data.to(device)
    y_data = y_data.to(device)
    # (1, context_size, x_dim)
    # (1, context_size, y_dim)

    x_data = x_data.expand(1, num_samples, -1, -1)
    y_data = y_data.expand(1, num_samples, -1, -1)
    # (1, num_samples, context_size, x_dim)
    # (1, num_samples, context_size, y_dim)

    x_data_sorted, indices = x_data.sort(dim=2)
    x_data_sorted = x_data_sorted.squeeze(0).squeeze(-1).detach().cpu().numpy()
    y_data_sorted = (
        y_data.gather(2, indices).squeeze(0).squeeze(-1).detach().cpu().numpy()
    )
    # (num_samples, context_size, x_dim)
    # (num_samples, context_size, y_dim)

    fig = plt.figure(figsize=(4, 2 * max_context_size))
    subfigs = fig.subfigures(nrows=max_context_size, ncols=1)

    # fig.text(
    #     0,
    #     1,
    #     task_hash,
    #     fontsize=12,
    #     color="blue",
    #     ha="left",
    #     va="top",
    # )

    for row, subfig in enumerate(subfigs):
        subfig.suptitle(f"context size: {row + 1}")
        ax = subfig.subplots(nrows=1, ncols=1)

        context_size = row + 1

        x_context = x_data[:, :, :context_size, :]
        y_context = y_data[:, :, :context_size, :]
        # (1, num_samples, context_size, x_dim)
        # (1, num_samples, context_size, y_dim)

        y_dist, _ = model.inference(x_context, y_context, None, x_data)

        y_mu_sorted = (
            y_dist.mean.gather(2, indices).squeeze(0).squeeze(-1).detach().cpu().numpy()
        )
        y_sigma_sorted = (
            y_dist.scale.gather(2, indices)
            .squeeze(0)
            .squeeze(-1)
            .detach()
            .cpu()
            .numpy()
        )
        # (num_samples, target_size, y_dim)

        ax.scatter(x_data_sorted, y_data_sorted, marker="o", c="black", zorder=2)
        ax.scatter(
            x_context.detach().cpu().numpy(),
            y_context.detach().cpu().numpy(),
            marker="X",
            c="red",
            s=100,
            zorder=3,
        )

        for k in range(num_samples):
            ax.plot(
                x_data_sorted[k],
                y_mu_sorted[k],
                alpha=0.8,
                c="tab:blue",
                zorder=1,
            )
            if show_sigma:
                ax.fill_between(
                    x_data_sorted[k],
                    y_mu_sorted[k] - y_sigma_sorted[k],  # Lower bound
                    y_mu_sorted[k] + y_sigma_sorted[k],  # Upper bound
                    color="tab:purple",
                    alpha=0.2,
                    zorder=0,
                )

    plt.show()
