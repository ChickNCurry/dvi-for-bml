from typing import List, Tuple

from matplotlib.patches import Patch
import numpy as np
import torch
from matplotlib import cm
from matplotlib import pyplot as plt
from torch import Tensor

from dviforbml.evaluation.common import ModelInfo, ModelType
from dviforbml.evaluation.taskposterior.grid import (
    create_grid,
    eval_dist_on_grid,
    eval_hist_on_grid,
    eval_score_on_grid,
)
from dviforbml.evaluation.taskposterior.tp_metrics import compute_jsd


def vis_tp_eval(
    model_infos: List[ModelInfo],
    batch: Tuple[Tensor, Tensor],
    num_samples: int,
    device: torch.device,
    max_context_size: int,
    save_dir: str,
    show_score: bool = False,
    ranges: List[Tuple[float, float]] = [(-5, 5), (-5, 5)],
    index: int = 0,
) -> None:
    x_data, y_data = batch
    x_data = x_data.to(device)
    y_data = y_data.to(device)
    # (1, data_size, x_dim)
    # (1, data_size, y_dim)

    x_data = x_data.expand(1, num_samples, -1, -1)
    y_data = y_data.expand(1, num_samples, -1, -1)
    # (1, num_samples, data_size, x_dim)
    # (1, num_samples, data_size, y_dim)

    fig, axs = plt.subplots(
        nrows=max_context_size,
        ncols=2 * len(model_infos),
        figsize=(3 * len(model_infos), 1.5 * max_context_size),
    )

    if len(model_infos) == 1:
        axs = np.expand_dims(axs, axis=1)

    for row in range(max_context_size):
        context_size = row + 1

        axs[row, 0].set_ylabel(f"Context Size: {context_size}", fontsize=8)

        x_context = x_data[:, :, 0:context_size, :]
        y_context = y_data[:, :, 0:context_size, :]
        # (1, num_samples, context_size, x_dim)
        # (1, num_samples, context_size, y_dim)

        for col, model_info in enumerate(model_infos):
            assert model_info.model is not None

            # add 50 white spaces before name

            axs[0, col * 2].set_title(
                f"{' ' * 50}{model_info.name} \n Approximation", fontsize=8
            )
            axs[0, col * 2 + 1].set_title("Target", fontsize=8)

            # bbox1 = axs[0, col * 2].get_position()
            # bbox2 = axs[0, col * 2 + 1].get_position()
            # mid_x = (bbox1.x0 + bbox2.x1) / 2
            # top_y = max(bbox1.y1, bbox2.y1)

            # fig.text(
            #     mid_x,
            #     top_y + 0.02,
            #     model_info.name,
            #     ha="center",
            #     va="bottom",
            #     fontsize=8,
            # )

            ax_model = axs[row, col * 2]
            ax_target = axs[row, col * 2 + 1]

            sqrt = np.sqrt(x_data.shape[0] * x_data.shape[1])
            assert sqrt.is_integer()
            num_cells = int(sqrt)
            grid = create_grid(ranges, num_cells)

            target_dist = model_info.model.get_target_dist(x_context, y_context, None)

            if model_info.type == ModelType.DVINP:
                _, z = model_info.model.inference(x_context, y_context, None, x_data)
            else:
                with torch.no_grad():
                    z = model_info.model.inference(x_context, y_context, None, x_data)
            # (1, num_samples, z_dim)

            log_probs = (
                eval_hist_on_grid(
                    z.reshape(-1, z[-1].shape[-1]).detach().cpu().numpy(),
                    ranges,
                    num_cells,
                )
                if z is not None
                else np.zeros((x_context.shape[1], 2))
            )
            # (num_samples, z_dim)

            # with torch.no_grad():
            target_log_probs = (
                eval_dist_on_grid(grid, target_dist, device=device).squeeze(0)
                if target_dist is not None
                else np.zeros_like(log_probs)
            )  # (num_samples, z_dim)

            ax_model.contourf(
                grid[:, :, 0],
                grid[:, :, 1],
                np.exp(log_probs),
                cmap=cm.coolwarm,
            )
            ax_target.contourf(
                grid[:, :, 0],
                grid[:, :, 1],
                np.exp(target_log_probs),
                cmap=cm.RdYlBu,  # cm.PuOr
            )

            jsd = compute_jsd(log_probs, target_log_probs)

            legend_patch = Patch(facecolor="none", label=f"JSD: {jsd:.2f}")
            ax_model.legend(handles=[legend_patch], fontsize=8)

            # for spine in ax_model.spines.values():
            #     spine.set_edgecolor("tab:green")
            #     spine.set_linewidth(2.0)

            # for spine in ax_target.spines.values():
            #     spine.set_edgecolor("tab:orange")
            #     spine.set_linewidth(2.0)

            for ax in [ax_model, ax_target]:
                ax.set_aspect("equal")
                ax.grid(True)
                ax.set_xticklabels([])
                ax.set_yticklabels([])

            if show_score:
                score_vals = eval_score_on_grid(grid, target_dist, device=device)
                for ax in [ax_model, ax_target]:
                    ax.quiver(
                        grid[:, :, 0],
                        grid[:, :, 1],
                        score_vals[:, :, 0],
                        score_vals[:, :, 1],
                        scale_units="xy",
                    )

    plt.tight_layout()
    plt.savefig(f"{save_dir}/tp_{index}.pdf")
    plt.close()
