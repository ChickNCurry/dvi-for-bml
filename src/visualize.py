from typing import Tuple

import torch
from matplotlib import pyplot as plt
from omegaconf import DictConfig
from torch import Tensor
from torch.distributions import Distribution
from torch.utils.data import DataLoader

from src.control import Control
from src.decoder import Decoder, LikelihoodTimesPrior
from src.dvi_process import DiffusionVIProcess
from src.encoder import SetEncoder


def visualize(
    device: torch.device,
    dvi_process: DiffusionVIProcess,
    set_encoder: SetEncoder,
    decoder: Decoder,
    dataloader: DataLoader[Tuple[Tensor, Tensor]],
    control: Control,
    config: DictConfig,
    num_samples: int,
    max_context_size: int,
) -> None:

    x_data, y_data = next(iter(dataloader))
    x_data, y_data = x_data.to(device), y_data.to(device)
    x_data, y_data = x_data.expand(num_samples, -1, -1), y_data.expand(
        num_samples, -1, -1
    )

    x_data_sorted, indices = x_data.sort(dim=1)
    indices = indices[1].squeeze(1)
    x_data_sorted = x_data_sorted.cpu().detach().numpy()
    y_data_sorted = y_data[:, indices, :].cpu().detach().numpy()

    fig = plt.figure(figsize=(9, 3 * max_context_size), constrained_layout=True)
    subfigs = fig.subfigures(nrows=max_context_size, ncols=1)

    for row, subfig in enumerate(subfigs):
        subfig.suptitle(f"Context Size: {row + 1}")
        ax = subfig.subplots(nrows=1, ncols=2, width_ratios=[3, 1])

        context_size = row + 1
        x_context = x_data[:, :context_size, :]
        y_context = y_data[:, :context_size, :]
        context = torch.cat([x_context, y_context], dim=-1)

        aggregated, non_aggregated = set_encoder(context, None)

        p_z_T = LikelihoodTimesPrior(
            decoder=decoder,
            x_target=x_context,
            y_target=y_context,
            mask=None,
            context_embedding=(
                non_aggregated if config.decoder.is_cross_attentive else aggregated
            ),
        )

        _, z_samples = dvi_process.run_chain(
            p_z_T,
            non_aggregated if control.is_cross_attentive else aggregated,
            None,
        )
        y_dist: Distribution = decoder(
            x_data,
            z_samples[-1],
            non_aggregated if control.is_cross_attentive else aggregated,
            None,
        )

        y_mu_sorted = y_dist.mean[:, indices, :].cpu().detach().numpy()

        for k in range(num_samples):
            ax[0].plot(
                x_data_sorted[k].squeeze(1),
                y_mu_sorted[k].squeeze(1),
                alpha=0.2,
                c="tab:blue",
                zorder=0,
            )

        ax[0].scatter(x_data_sorted, y_data_sorted, marker="o", c="black", zorder=1)
        ax[0].scatter(
            x_context.cpu().detach().numpy(),
            y_context.cpu().detach().numpy(),
            marker="X",
            c="red",
            s=100,
            zorder=2,
        )

        ax[1].hist2d(
            z_samples[-1][:, 0].cpu().detach().numpy(),
            z_samples[-1][:, 1].cpu().detach().numpy(),
            bins=20,
        )

    plt.show()
