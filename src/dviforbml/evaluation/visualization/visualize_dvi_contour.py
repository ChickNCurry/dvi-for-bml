from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt

from dviforbml.architectures.dvi import DVI
from dviforbml.components.cdvi.dds import DDS
from dviforbml.evaluation.taskposterior.grid import (
    create_grid,
    eval_dist_on_grid,
    eval_hist_on_grid,
)
from dviforbml.evaluation.taskposterior.tp_metrics import compute_bd, compute_jsd
from dviforbml.utils.datasets import ContextSetDataset


def visualize_dvi_2d_contour_all(
    device: torch.device,
    model: DVI,
    dataset: ContextSetDataset,
    num_samples: int = 10000,  #  # 8192
    samples_to_plot: int = 200,
    max_context_size: int = 8,
    plot_range: List[Tuple[float, float]] = [(-5, 5), (-5, 5)],
    save_fig_path: str | None = None,
    save_csv_path: str | None = None,
) -> None:
    assert max_context_size <= dataset.max_context_size

    context = dataset.sampling_factor * torch.rand(
        (1, num_samples, max_context_size, 2), device=device
    )

    det_context = (
        dataset.sampling_factor
        * torch.ones((1, num_samples, max_context_size, 2), device=device)
        * 0.5
    )

    ncols = max_context_size

    fig = plt.figure(figsize=(3 * ncols, 16))
    subfigs = fig.subfigures(nrows=1, ncols=ncols)

    jsds = []
    bds = []

    multipliers = [(-1, 1), (1, 1), (1, -1), (-1, -1)]

    for col, subfig in enumerate(subfigs):
        context_size = col + 1

        sub_context = context[:, :, :context_size, :]
        det_sub_context = det_context[:, :, :context_size, :]
        # (1, num_samples, context_size, 2)

        ax = subfig.subplots(nrows=len(multipliers), ncols=1)

        context_size_jsds = []
        context_size_bds = []

        for i, m in enumerate(multipliers):
            mul_sub_context = sub_context * torch.tensor(m, device=device)
            mul_det_sub_context = det_sub_context * torch.tensor(m, device=device)
            # (1, num_samples, context_size, 2)

            target_dist = model.contextual_target(mul_sub_context, None)
            det_target_dist = model.contextual_target(mul_det_sub_context, None)

            r, s = model.encoder(mul_sub_context.to(device), None)
            _, zs = model.cdvi.run_forward_process(target_dist, r, None, s, None)

            assert zs is not None

            z_T = zs[-1].detach().cpu().numpy()
            z_T = z_T.reshape(-1, z_T.shape[-1])

            num_cells = int(np.sqrt(context.shape[1]))
            grid = create_grid(plot_range, num_cells)

            dvi_vals = eval_hist_on_grid(z_T, plot_range, num_cells)
            target_vals = eval_dist_on_grid(grid, target_dist, device)[0]
            det_target_vals = eval_dist_on_grid(grid, det_target_dist, device)[0]

            cs_jsd = compute_jsd(dvi_vals, target_vals)
            cs_bd = compute_bd(dvi_vals, target_vals)

            context_size_jsds.append(cs_jsd)
            context_size_bds.append(cs_bd)

            ax[i].contour(
                grid[:, :, 0],
                grid[:, :, 1],
                np.exp(det_target_vals.reshape(num_cells, num_cells)),
                levels=50,
                zorder=0,
            )

            ax[i].scatter(
                z_T[:samples_to_plot, 0],
                z_T[:samples_to_plot, 1],
                color="red",
                marker="x",
                s=20,
                alpha=0.7,
                zorder=1,
            )

            ax[i].set_xlim(plot_range[0])
            ax[i].set_ylim(plot_range[1])
            ax[i].axis("off")
            ax[i].set_aspect("equal")

        jsd = np.mean(context_size_jsds)
        bd = np.mean(context_size_bds)

        jsds.append(jsd)
        bds.append(bd)

        subfig.suptitle(
            f"Context Size: {context_size} \n JSD: {jsd:.2f} \n BD: {bd:.2f}",
            fontsize=24,
        )

    if save_csv_path is not None:
        df = pd.DataFrame(
            {"jsds": jsds, "bds": bds}, index=[col + 1 for col in range(ncols)]
        )
        df.to_csv(save_csv_path)

    plt.tight_layout()

    if save_fig_path is not None:
        plt.savefig(save_fig_path)
    else:
        plt.show()


def visualize_dvi_2d_contour(
    device: torch.device,
    model: DVI,
    dataset: ContextSetDataset,
    num_samples: int = 1600,  #  # 8192
    samples_to_plot: int = 100,
    plot_range: List[Tuple[float, float]] = [(-5, 5), (-5, 5)],
    multipliers: Tuple[float, float] = (1, 1),
) -> None:
    ncols = dataset.max_context_size - 1

    context = dataset.sampling_factor * torch.rand(
        (1, num_samples, ncols, 2), device=device
    )

    det_context = (
        dataset.sampling_factor
        * torch.ones((1, num_samples, ncols, 2), device=device)
        * 0.5
    )

    context[:, :, :, 0] = context[:, :, :, 0] * multipliers[0]
    context[:, :, :, 1] = context[:, :, :, 1] * multipliers[1]

    det_context[:, :, :, 0] = det_context[:, :, :, 0] * multipliers[0]
    det_context[:, :, :, 1] = det_context[:, :, :, 1] * multipliers[1]

    fig = plt.figure(figsize=(3 * ncols, 4))
    subfigs = fig.subfigures(nrows=1, ncols=ncols)

    jsds = []
    bds = []

    for col, subfig in enumerate(subfigs):
        context_size = col + 1

        ax = subfig.subplots(nrows=1, ncols=1)

        sub_context = context[:, :, :context_size, :]
        det_sub_context = det_context[:, :, :context_size, :]

        target_dist = model.contextual_target(sub_context, None)
        det_target_dist = model.contextual_target(det_sub_context, None)

        r, s = model.encoder(sub_context.to(device), None)
        _, zs = model.cdvi.run_forward_process(target_dist, r, None, s, None)

        assert zs is not None

        # z_0 = zs[0].detach().cpu().numpy()
        z_T = zs[-1].detach().cpu().numpy()
        z_target = target_dist.sample().detach().cpu().numpy()

        # z_0 = z_0.reshape(-1, z_0.shape[-1])
        z_T = z_T.reshape(-1, z_T.shape[-1])
        z_target = z_target.reshape(-1, z_target.shape[-1])

        num_cells = int(np.sqrt(context.shape[1]))
        grid = create_grid(plot_range, num_cells)

        dvi_vals = eval_hist_on_grid(z_T, plot_range, num_cells)
        target_vals = eval_dist_on_grid(grid, target_dist, device=device).squeeze(0)
        det_target_vals = eval_dist_on_grid(
            grid, det_target_dist, device=device
        ).squeeze(0)

        jsd = compute_jsd(dvi_vals, target_vals)
        bd = compute_bd(dvi_vals, target_vals)

        jsds.append(jsd)
        bds.append(bd)

        ax.contourf(
            grid[:, :, 0],
            grid[:, :, 1],
            np.exp(det_target_vals.reshape(num_cells, num_cells)),
            # gaussian_filter(
            #     np.exp(det_target_vals.reshape(num_cells, num_cells)), sigma=1.0
            # ),
            # cmap="autumn",
            levels=10,
            zorder=0,
        )

        # sns.kdeplot(ax=ax, x=z_target[:, 0], y=z_target[:, 1])

        # ax.hist2d(
        #     z_T[:, 0],
        #     z_T[:, 1],
        #     bins=num_cells,
        #     range=plot_range,
        #     cmap="Blues",
        #     zorder=1,
        #     alpha=0.7,
        #     density=True,
        # )

        ax.scatter(
            z_T[:samples_to_plot, 0],
            z_T[:samples_to_plot, 1],
            color="red",
            marker="x",
            s=20,
            alpha=0.7,
        )

        ax.set_xlim(plot_range[0])
        ax.set_ylim(plot_range[1])
        ax.axis("off")
        ax.set_aspect("equal")

        subfig.suptitle(f"Context Size: {context_size} \n JSD: {jsd:.2f}, BD: {bd:.2f}")

        # print(f"context size: {row + 1}, jsd: {jsd}, bd: {bd}")

    # import pandas as pd

    # df = pd.DataFrame({id: jsds}, index=[row + 1 for row in range(nrows)])
    # print(df.head())

    plt.tight_layout()

    plt.show()
