from typing import List, Tuple

import numpy as np
import seaborn as sns
import torch
from matplotlib import pyplot as plt
from numpy.typing import NDArray

from dviforbml.architectures.dvi import DVI
from dviforbml.utils.datasets import ContextSetDataset, ContextTestDataset


def visualize_dvi_1d(
    device: torch.device,
    model: DVI,
    dataset: ContextTestDataset,
    num_samples: int = 256,
) -> None:
    context = dataset.sampling_factor * torch.ones(
        (1, num_samples, 1, 1), device=device
    )
    target_dist = model.contextual_target(context, None)

    r = model.encoder(context.to(device), None)
    _, z_samples = model.cdvi.run_forward_process(target_dist, r, None, None)

    assert z_samples is not None

    z_0_samples = z_samples[0].detach().cpu().numpy()
    z_T_samples = z_samples[-1].detach().cpu().numpy()
    z_trajectories = [
        [z[0, i, 0].detach().cpu().numpy() for z in z_samples]
        for i in range(num_samples)
    ]
    z_target_samples = target_dist.sample().detach().cpu().numpy()

    z_0_samples = z_0_samples.reshape(-1)
    z_T_samples = z_T_samples.reshape(-1)
    z_target_samples = z_target_samples.reshape(-1)

    fig, ax = plt.subplots(
        1, 4, figsize=(14, 3), gridspec_kw={"width_ratios": [1, 2, 1, 1]}
    )

    sns.histplot(z_0_samples, ax=ax[0], stat="density", bins=30, kde=True)
    ax[0].set_title("$z_0\sim q_\phi(z_0)$")
    ax[0].set_ylabel(None)
    ax[0].set_xlim(-8, 8)
    ax[0].set_ylim(0, 0.5)

    for i in range(num_samples):
        ax[1].plot(z_trajectories[i])
    ax[1].set_title("$z_{1:T}\sim q_\phi(z_{1:T}|z_0)$")

    sns.histplot(z_T_samples, ax=ax[2], stat="density", bins=50, kde=True)
    ax[2].set_title("$z_T\sim q_\phi(z_T|z_{0:T-1})$")
    ax[2].set_ylabel(None)
    ax[2].set_xlim(-8, 8)
    ax[2].set_ylim(0, 0.5)

    sns.histplot(z_target_samples, ax=ax[3], stat="density", bins=50, kde=True)
    ax[3].set_title("$z_T\sim p(z_T)$")
    ax[3].set_ylabel(None)
    ax[3].set_xlim(-8, 8)
    ax[3].set_ylim(0, 0.5)

    ax[0].set_yticks([])
    ax[1].set_yticks([])
    ax[2].set_yticks([])
    ax[3].set_yticks([])

    ax[0].set_xticks([])

    ax[2].set_xticks([])
    ax[3].set_xticks([])

    plt.tight_layout()
    plt.show()


def visualize_dvi_2d(
    device: torch.device,
    model: DVI,
    dataset: ContextSetDataset,
    num_samples: int = 1600,  # 8192
    bins: int = 40,
    plot_range: List[Tuple[float, float]] = [(-5, 5), (-5, 5)],
) -> None:
    nrows = dataset.max_context_size - 1

    context = dataset.sampling_factor * torch.rand(
        (1, num_samples, nrows, 2), device=device
    )
    context[:, :, :, 0] = context[:, :, :, 0] * 1
    context[:, :, :, 1] = context[:, :, :, 1] * 1

    fig = plt.figure(figsize=(9, 3 * nrows), constrained_layout=True)
    subfigs = fig.subfigures(nrows=nrows, ncols=1)

    # jsds = []

    for row, subfig in enumerate(subfigs):
        subfig.suptitle(f"context size: {row + 1}")
        ax = subfig.subplots(nrows=1, ncols=3, width_ratios=[1, 1, 1])

        context_size = row + 1
        sub_context = context[:, :, :context_size, :]

        target_dist = model.contextual_target(sub_context, None)

        r = model.encoder(sub_context.to(device), None)
        _, z_samples = model.cdvi.run_forward_process(target_dist, r, None, None)

        assert z_samples is not None

        z_0_samples = z_samples[0].detach().cpu().numpy()
        z_T_samples = z_samples[-1].detach().cpu().numpy()
        # z_trajectories = [
        #     [z[0, i, :].detach().cpu().numpy() for z in z_samples]
        #     for i in range(num_samples)
        # ]
        z_target_samples = target_dist.sample().detach().cpu().numpy()

        z_0_samples = z_0_samples.reshape(-1, z_0_samples.shape[-1])
        z_T_samples = z_T_samples.reshape(-1, z_T_samples.shape[-1])
        z_target_samples = z_target_samples.reshape(-1, z_target_samples.shape[-1])

        ax[0].hist2d(z_0_samples[:, 0], z_0_samples[:, 1], bins=bins, range=plot_range)
        ax[0].set_title("prior $q_\phi(z_0)$")

        ax[1].hist2d(z_T_samples[:, 0], z_T_samples[:, 1], bins=bins, range=plot_range)
        ax[1].set_title("marginal $q_\phi(z_T|z_{0:T-1},c)$")

        ax[2].hist2d(
            z_target_samples[:, 0], z_target_samples[:, 1], bins=bins, range=plot_range
        )
        ax[2].set_title("target $p(z_T|c)$")

        for a in ax:
            a.axis("off")

        # num_cells = int(np.sqrt(context.shape[1]))
        # grid = create_grid(plot_range, num_cells)

        # dvi_vals = eval_hist_on_grid(z_T_samples, plot_range, num_cells)
        # target_vals = eval_dist_on_grid(grid, target, device=device).squeeze(0)

        # jsd = compute_jsd(dvi_vals, target_vals)

        # jsds.append(jsd)

        # print(f"context size: {row + 1}, jsd: {jsd}, bd: {bd}")

    # import pandas as pd

    # df = pd.DataFrame({id: jsds}, index=[row + 1 for row in range(nrows)])
    # print(df.head())

    plt.show()


def visualize_dvi_2d_traj(
    z_trajectories: List[List[NDArray[np.float32]]], num_samples: int
) -> None:
    fig, ax = plt.subplots(2, 1, figsize=(9, 6))

    for i in range(num_samples):
        ax[0].plot([z[0] for z in z_trajectories[i]])
        ax[0].set_title("Dimension 0")

        ax[1].plot([z[1] for z in z_trajectories[i]])
        ax[1].set_title("Dimension 1")

    plt.suptitle("Samples from Forward Process $q(z_{0:T}|c)$")

    plt.tight_layout()
    plt.show()
