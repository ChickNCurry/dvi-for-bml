from typing import Callable, List

import numpy as np
import torch
import wandb
from decoder import Decoder
from dvi_process import DiffusionVIProcess
from encoder import SetEncoder, TestEncoder
from torch import Tensor
from torch.distributions import Distribution
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm


def train(
    dvi_process: DiffusionVIProcess,
    encoder: SetEncoder | TestEncoder,
    device: torch.device,
    num_epochs: int,
    dataloader: DataLoader[Tensor],
    target_constructor: Callable[[Tensor], Distribution],
    optimizer: Optimizer,
    scheduler: LRScheduler | None,
    wandb_logging: bool,
    decoder: Decoder | None = None,
) -> List[float]:

    # torch.autograd.set_detect_anomaly(True)

    dvi_process = dvi_process.to(device)

    losses = []

    for epoch in range(num_epochs):

        dvi_process.train()
        with torch.inference_mode(False):

            loop = tqdm(dataloader, total=len(dataloader))

            for batch in loop:

                if decoder is None:
                    loss = step(dvi_process, encoder, batch, device, target_constructor)
                elif decoder is not None and isinstance(encoder, SetEncoder):
                    loss = step_bml_alternative(
                        dvi_process,
                        encoder,
                        decoder,
                        batch,
                        device,
                        target_constructor,
                    )

                optimizer.zero_grad()
                loss.backward()  # type: ignore
                optimizer.step()

                if scheduler is not None:
                    scheduler.step()

                loop.set_postfix(
                    epoch=epoch,
                    loss=loss.item(),
                )

                losses.append(loss.item())

                if wandb_logging:
                    wandb.log({"train/loss": loss.item()})

                    # for i, sigma in enumerate(dvi_process.sigmas):
                    #     wandb.log({"hyperparams/sigma_" + str(i): sigma.data.item()})

                    # for i, beta in enumerate(dvi_process.betas):
                    #     wandb.log({"hyperparams/beta_" + str(i): beta.data.item()})

    return losses


def step(
    dvi_process: DiffusionVIProcess,
    encoder: TestEncoder | SetEncoder,
    batch: Tensor,
    device: torch.device,
    target_constructor: Callable[[Tensor], Distribution],
) -> Tensor:

    batch = batch.to(device)

    random_context_size: int = np.random.randint(1, batch.shape[1] + 1)
    context = batch[:, 0:random_context_size, :]

    p_z_T = target_constructor(context)
    log_w, _ = dvi_process.run_chain(p_z_T, encoder(context))

    loss = -log_w

    return loss


def step_bml(
    dvi_process: DiffusionVIProcess,
    set_encoder: SetEncoder,
    decoder: Decoder,
    batch: Tensor,
    device: torch.device,
    target_constructor: Callable[[Tensor], Distribution],
) -> Tensor:

    x_data, y_data = batch
    x_data, y_data = x_data.to(device), y_data.to(device)
    # (batch_size, context_size, x_dim), (batch_size, context_size, y_dim)

    rand_sub_context_size: int = 3  # x_data.shape[1]
    # np.random.randint(1, x_data.shape[1] + 1)

    x_context = x_data[:, 0:rand_sub_context_size, :]
    y_context = y_data[:, 0:rand_sub_context_size, :]

    context = torch.cat([x_context, y_context], dim=-1)
    # (batch_size, context_size, x_dim + y_dim)

    context = set_encoder(context)
    # (batch_size, h_dim)

    p_z_T = target_constructor(context)
    log_w, z_samples = dvi_process.run_chain(p_z_T, context)

    log_like: Tensor = (
        decoder(x_context, z_samples[-1], context)
        .log_prob(y_context)
        .mean(dim=0)
        # .mean(dim=1)
        .sum()
    )

    loss = -log_like - log_w

    return loss


def step_bml_alternative(
    dvi_process: DiffusionVIProcess,
    set_encoder: SetEncoder,
    decoder: Decoder,
    batch: Tensor,
    device: torch.device,
    target_constructor: Callable[[Tensor], Distribution],
) -> Tensor:

    x_data, y_data = batch
    x_data, y_data = x_data.to(device), y_data.to(device)
    # (batch_size, context_size, x_dim), (batch_size, context_size, y_dim)

    data = torch.cat([x_data, y_data], dim=-1)
    # (batch_size, context_size, x_dim + y_dim)

    data = set_encoder(data)
    # (batch_size, h_dim)

    rand_sub_context_size: int = np.random.randint(1, x_data.shape[1] + 1)

    x_context = x_data[:, 0:rand_sub_context_size, :]
    y_context = y_data[:, 0:rand_sub_context_size, :]

    context = torch.cat([x_context, y_context], dim=-1)
    # (batch_size, context_size, x_dim + y_dim)

    context = set_encoder(context)
    # (batch_size, h_dim)

    p_z_T = target_constructor(context)
    log_w, z_samples = dvi_process.run_chain(p_z_T, data)

    log_like_data: Tensor = (
        decoder(x_data, z_samples[-1], data).log_prob(y_data).mean(dim=0).sum()
    )

    log_like_context: Tensor = (
        decoder(x_context, z_samples[-1], context).log_prob(y_context).mean(dim=0).sum()
    )

    loss = -log_like_data - log_like_context - log_w

    return loss
