from typing import List, Tuple

from matplotlib.patches import Patch
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import Tensor

from dviforbml.evaluation.common import ModelInfo, ModelType
from dviforbml.evaluation.taskposterior.grid import (
    create_grid,
    eval_dist_on_grid,
    sample_from_log_probs,
)
from dviforbml.utils.datasets import hash_tensor
import copy


def vis_pred_eval(
    model_infos: List[ModelInfo],
    batch: Tuple[Tensor, Tensor],
    num_samples: int,
    device: torch.device,
    max_context_size: int,
    save_dir: str,
    index: int = 0,
) -> None:
    x_data, y_data = batch
    # task_hash = hash_tensor(x_data)  # + hash_tensor(y_data)

    x_data = x_data.to(device)
    y_data = y_data.to(device)
    # (1, data_size, x_dim)
    # (1, data_size, y_dim)

    x_data = x_data.expand(1, num_samples, -1, -1)
    y_data = y_data.expand(1, num_samples, -1, -1)
    # (1, num_samples, data_size, x_dim)
    # (1, num_samples, data_size, y_dim)

    sorted, indices = x_data.squeeze(0).squeeze(-1).sort(dim=1)
    x_data_sorted = sorted.detach().cpu().numpy()
    y_data_sorted = (
        y_data.squeeze(0).squeeze(-1).gather(1, indices).detach().cpu().numpy()
    )  # (num_samples, data_size)

    fig, axs = plt.subplots(
        nrows=max_context_size,
        ncols=len(model_infos),
        figsize=(3 * len(model_infos), 1.5 * max_context_size),
    )

    # fig.text(
    #     0,
    #     1,
    #     task_hash,
    #     fontsize=12,
    #     color="blue",
    #     ha="left",
    #     va="top",
    # )

    if len(model_infos) == 1:
        axs = np.expand_dims(axs, axis=1)

    for row in range(max_context_size):
        context_size = row + 1

        axs[row, 0].set_ylabel(f"Context Size: {context_size}", fontsize=8)

        x_context = x_data[:, :, :context_size, :]
        y_context = y_data[:, :, :context_size, :]
        # (1, num_samples, context_size, x_dim)
        # (1, num_samples, context_size, y_dim)

        x_context_np = x_context.squeeze(0).squeeze(-1).detach().cpu().numpy()
        y_context_np = y_context.squeeze(0).squeeze(-1).detach().cpu().numpy()
        # (num_samples, data_size)

        for col, model_info in enumerate(model_infos):
            assert model_info.model is not None

            axs[0, col].set_title(model_info.name, fontsize=8)

            ax = axs[row, col]

            if model_info.type == ModelType.DVINP:
                y_dist_data, _ = model_info.model.inference(
                    x_context, y_context, None, x_data
                )
            else:
                with torch.no_grad():
                    y_dist_data, _ = model_info.model.inference(
                        x_context, y_context, None, x_data
                    )
            # (1, num_samples, data_size, y_dim)

            y_mu_sorted = (
                y_dist_data.mean.squeeze(0)
                .squeeze(-1)
                .gather(1, indices)
                .detach()
                .cpu()
                .numpy()
            )  # (num_samples, data_size)
            y_sigma_sorted = (
                y_dist_data.scale.squeeze(0)
                .squeeze(-1)
                .gather(1, indices)
                .detach()
                .cpu()
                .numpy()
            )  # (num_samples, data_size)

            ax.scatter(
                x_data_sorted[0], y_data_sorted[0], marker="o", c="black", zorder=2
            )
            ax.scatter(
                x_context_np[0], y_context_np[0], marker="X", c="red", s=100, zorder=3
            )

            lmpl = y_dist_data.log_prob(y_data).mean((0, 1)).sum().item()
            mse = ((y_mu_sorted - y_data_sorted) ** 2).mean()

            legend_patch_lmpl = Patch(facecolor="none", label=f"LMPL: {lmpl:.2f}")
            legend_patch_mse = Patch(facecolor="none", label=f"MSE: {mse:.2f}")
            ax.legend(handles=[legend_patch_lmpl, legend_patch_mse], fontsize=8)

            ax.set_ylim(-10, 10)
            ax.set_xticks([])
            ax.set_yticks([])

            for k in range(num_samples):
                ax.plot(
                    x_data_sorted[k], y_mu_sorted[k], alpha=0.8, c="tab:blue", zorder=1
                )

                if model_info.type == ModelType.CNP:
                    ax.fill_between(
                        x_data_sorted[0],
                        y_mu_sorted[k] - y_sigma_sorted[k],
                        y_mu_sorted[k] + y_sigma_sorted[k],
                        color="tab:purple",
                        alpha=0.2,
                        zorder=0,
                    )

    # plt.suptitle("Comparison of different BML models")
    plt.tight_layout()
    plt.savefig(f"{save_dir}/pred_{index}.pdf")  # bbox_inches='tight'
    plt.close()


def vis_pred_eval_gt(
    model_infos: List[ModelInfo],
    batch: Tuple[Tensor, Tensor],
    num_samples: int,
    device: torch.device,
    max_context_size: int,
    save_dir: str,
    index: int = 0,
    ranges: List[Tuple[float, float]] = [(-5, 5), (-5, 5)],
) -> None:
    x_data, y_data = batch
    # task_hash = hash_tensor(x_data)  # + hash_tensor(y_data)

    x_data = x_data.to(device)
    y_data = y_data.to(device)
    # (1, data_size, x_dim)
    # (1, data_size, y_dim)

    x_data = x_data.expand(1, num_samples, -1, -1)
    y_data = y_data.expand(1, num_samples, -1, -1)
    # (1, num_samples, data_size, x_dim)
    # (1, num_samples, data_size, y_dim)

    sorted, indices = x_data.squeeze(0).squeeze(-1).sort(dim=1)
    x_data_sorted = sorted.detach().cpu().numpy()
    y_data_sorted = (
        y_data.squeeze(0).squeeze(-1).gather(1, indices).detach().cpu().numpy()
    )  # (num_samples, data_size)

    fig, axs = plt.subplots(
        nrows=max_context_size,
        ncols=len(model_infos),
        figsize=(3 * len(model_infos), 1.5 * max_context_size),
    )

    if len(model_infos) == 1:
        axs = np.expand_dims(axs, axis=1)

    for row in range(max_context_size):
        context_size = row + 1

        axs[row, 0].set_ylabel(f"Context Size: {context_size}", fontsize=8)

        x_context = x_data[:, :, :context_size, :]
        y_context = y_data[:, :, :context_size, :]
        # (1, num_samples, context_size, x_dim)
        # (1, num_samples, context_size, y_dim)

        x_context_np = x_context.squeeze(0).squeeze(-1).detach().cpu().numpy()
        y_context_np = y_context.squeeze(0).squeeze(-1).detach().cpu().numpy()
        # (num_samples, data_size)

        for col, model_info in enumerate(model_infos):
            assert model_info.model is not None

            axs[0, col].set_title(
                model_info.name if col != len(model_infos) - 1 else "Ground Truth",
                fontsize=8,
            )

            ax = axs[row, col]

            if model_info.type == ModelType.DVINP:
                y_dist_data, _ = model_info.model.inference(
                    x_context, y_context, None, x_data
                )

                if col == len(model_infos) - 1:
                    num_cells = int(np.sqrt(x_data.shape[0] * x_data.shape[1]))
                    grid = create_grid(ranges, num_cells)

                    target_log_probs = eval_dist_on_grid(
                        grid,
                        model_info.model.get_target_dist(x_context, y_context, None),
                        device=device,
                    ).squeeze(0)

                    target_samples_np = sample_from_log_probs(
                        grid, target_log_probs, num_samples
                    )
                    target_samples = (
                        torch.from_numpy(target_samples_np).unsqueeze(0).to(device)
                    )
                    y_dist_data = model_info.model.decoder(target_samples, x_data)

            else:
                with torch.no_grad():
                    y_dist_data, _ = model_info.model.inference(
                        x_context, y_context, None, x_data
                    )
            # (1, num_samples, data_size, y_dim)

            y_mu_sorted = (
                y_dist_data.mean.squeeze(0)
                .squeeze(-1)
                .gather(1, indices)
                .detach()
                .cpu()
                .numpy()
            )  # (num_samples, data_size)
            y_sigma_sorted = (
                y_dist_data.scale.squeeze(0)
                .squeeze(-1)
                .gather(1, indices)
                .detach()
                .cpu()
                .numpy()
            )  # (num_samples, data_size)

            ax.scatter(
                x_data_sorted[0], y_data_sorted[0], marker="o", c="black", zorder=2
            )
            ax.scatter(
                x_context_np[0], y_context_np[0], marker="X", c="red", s=100, zorder=3
            )

            lmpl = y_dist_data.log_prob(y_data).mean((0, 1)).sum().item()
            mse = ((y_mu_sorted - y_data_sorted) ** 2).mean()

            legend_patch_lmpl = Patch(facecolor="none", label=f"LMPL: {lmpl:.2f}")
            legend_patch_mse = Patch(facecolor="none", label=f"MSE: {mse:.2f}")
            ax.legend(handles=[legend_patch_lmpl, legend_patch_mse], fontsize=8)

            ax.set_ylim(-10, 10)
            ax.set_xticks([])
            ax.set_yticks([])

            for k in range(min(num_samples, 100)):
                ax.plot(
                    x_data_sorted[k],
                    y_mu_sorted[k],
                    alpha=0.8,
                    c="tab:blue" if col != len(model_infos) - 1 else "tab:orange",
                    zorder=1,
                )

                if model_info.type == ModelType.CNP:
                    ax.fill_between(
                        x_data_sorted[0],
                        y_mu_sorted[k] - y_sigma_sorted[k],
                        y_mu_sorted[k] + y_sigma_sorted[k],
                        color="tab:purple",
                        alpha=0.2,
                        zorder=0,
                    )

    # plt.suptitle("Comparison of different BML models")
    plt.tight_layout()
    plt.savefig(f"{save_dir}/pred_{index}.pdf")  # bbox_inches='tight'
    plt.close()
