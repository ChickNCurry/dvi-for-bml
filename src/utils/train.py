from typing import Any, Callable, List

import numpy as np
import torch
import wandb
from torch import Tensor
from torch.distributions import Distribution
from torch.optim import Optimizer  # type: ignore
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.components.cdvi import ContextualDVI
from src.components.decoder import LikelihoodTimesPrior


def train(
    device: torch.device,
    contextual_dvi: ContextualDVI,
    target_constructor: Any,
    num_epochs: int,
    dataloader: DataLoader[Any],
    optimizer: Optimizer,
    scheduler: ReduceLROnPlateau | None,
    max_clip_norm: float | None,
    wandb_logging: bool,
    alpha: float | None,
) -> List[float]:

    # torch.autograd.set_detect_anomaly(True)

    losses = []

    for epoch in range(num_epochs):

        contextual_dvi.train()
        with torch.inference_mode(False):

            loop = tqdm(dataloader, total=len(dataloader))

            for batch in loop:

                loss = (
                    step_better(device, contextual_dvi, target_constructor, batch)
                    if contextual_dvi.decoder is None
                    else step_bml_better(device, contextual_dvi, batch, alpha)
                )

                optimizer.zero_grad()

                loss.backward()  # type: ignore

                if max_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        contextual_dvi.parameters(), max_clip_norm
                    )

                optimizer.step()

                if scheduler is not None:
                    scheduler.step(loss)

                loop.set_postfix(
                    epoch=epoch,
                    loss=loss.item(),
                    lr=scheduler.get_last_lr() if scheduler is not None else None,
                )

                losses.append(loss.item())

                if wandb_logging:
                    wandb.log({"train/loss": loss.item()})

    return losses


def step(
    device: torch.device,
    contextual_dvi: ContextualDVI,
    target_constructor: Callable[[Tensor], Distribution],
    batch: Tensor,
) -> Tensor:

    batch = batch.to(device)

    random_context_size: int = np.random.randint(1, batch.shape[1] + 1)
    context = batch[:, 0:random_context_size, :]

    p_z_T = target_constructor(context)
    context_embedding = contextual_dvi.encoder(context)
    log_w, _ = contextual_dvi.dvi_process.run_chain(p_z_T, context_embedding, None)

    loss = -log_w

    return loss


def step_better(
    device: torch.device,
    contextual_dvi: ContextualDVI,
    target_constructor: Callable[[Tensor, Tensor], Distribution],
    batch: Tensor,
) -> Tensor:

    batch = batch.to(device)

    rand_context_sizes = torch.randint(
        low=1, high=batch.shape[1] + 1, size=(batch.shape[0],), device=device
    )

    position_indices = torch.arange(batch.shape[1], device=device).expand(
        batch.shape[0], -1
    )
    mask = (position_indices < rand_context_sizes.unsqueeze(-1)).float()
    context = batch * mask.unsqueeze(-1).expand(-1, -1, batch.shape[2])

    p_z_T = target_constructor(context, mask)

    aggregated, _ = contextual_dvi.encoder(context, mask)
    log_w, _ = contextual_dvi.dvi_process.run_chain(p_z_T, aggregated, mask)

    loss = -log_w

    return loss


def step_bml(
    device: torch.device,
    contextual_dvi: ContextualDVI,
    batch: Tensor,
) -> Tensor:

    assert contextual_dvi.decoder is not None

    x_data, y_data = batch
    x_data, y_data = x_data.to(device), y_data.to(device)
    # (batch_size, context_size, x_dim), (batch_size, context_size, y_dim)

    rand_sub_context_size: int = np.random.randint(1, x_data.shape[1] + 1)

    x_context = x_data[:, 0:rand_sub_context_size, :]
    y_context = y_data[:, 0:rand_sub_context_size, :]

    context = torch.cat([x_context, y_context], dim=-1)
    # (batch_size, context_size, x_dim + y_dim)

    context_embedding = contextual_dvi.encoder(context)
    # (batch_size, h_dim)

    p_z_T = LikelihoodTimesPrior(
        decoder=contextual_dvi.decoder,
        x_target=x_context,
        y_target=y_context,
        mask=None,
        context_embedding=context_embedding,
    )

    log_w, _ = contextual_dvi.dvi_process.run_chain(p_z_T, context_embedding, None)

    loss = -log_w

    return loss


def step_bml_better(
    device: torch.device,
    contextual_dvi: ContextualDVI,
    batch: Tensor,
    alpha: float | None,
) -> Tensor:

    assert contextual_dvi.decoder is not None

    x_data, y_data = batch
    x_data, y_data = x_data.to(device), y_data.to(device)
    # (batch_size, context_size, x_dim), (batch_size, context_size, y_dim)

    data = torch.cat([x_data, y_data], dim=-1)
    # (batch_size, context_size, x_dim + y_dim)

    if alpha is None:
        rand_context_sizes = torch.randint(
            1, data.shape[1] + 1, (data.shape[0],), device=device
        ).unsqueeze(-1)
    else:
        rand_context_sizes = torch.tensor(
            np.ceil(
                np.random.beta(a=alpha, b=2, size=(data.shape[0], 1)) * data.shape[1]
            ),
            device=device,
        )
    # (batch_size, 1)

    position_indices = torch.arange(data.shape[1], device=device).expand(
        data.shape[0], -1
    )  # (batch_size, context_size)

    mask = (position_indices < rand_context_sizes).float()
    # (batch_size, context_size)

    context = data * mask.unsqueeze(-1).expand(-1, -1, data.shape[2])
    # (batch_size, context_size, x_dim + y_dim)

    x_context = context[:, :, 0 : x_data.shape[2]]
    # (batch_size, context_size, x_dim)

    y_context = context[:, :, x_data.shape[2] : data.shape[2]]
    # (batch_size, context_size, y_dim)

    aggregated, non_aggregated = contextual_dvi.encoder(context, mask)
    # (batch_size, h_dim)

    p_z_T = LikelihoodTimesPrior(
        decoder=contextual_dvi.decoder,
        x_target=x_context,
        y_target=y_context,
        mask=mask,
        context_embedding=(
            non_aggregated if contextual_dvi.decoder.is_cross_attentive else aggregated
        ),
    )

    log_w, _ = contextual_dvi.dvi_process.run_chain(
        p_z_T,
        (
            non_aggregated
            if contextual_dvi.dvi_process.control.is_cross_attentive
            else aggregated
        ),
        mask,
    )

    loss = -log_w

    return loss


# def step_bml_alternative(
#     dvi_process: DiffusionVIProcess,
#     set_encoder: SetEncoder,
#     decoder: Decoder,
#     batch: Tensor,
#     device: torch.device,
#     target_constructor: Callable[[Tensor], Distribution],
# ) -> Tensor:

#     x_data, y_data = batch
#     x_data, y_data = x_data.to(device), y_data.to(device)
#     # (batch_size, context_size, x_dim), (batch_size, context_size, y_dim)

#     data = torch.cat([x_data, y_data], dim=-1)
#     # (batch_size, context_size, x_dim + y_dim)

#     data = set_encoder(data)
#     # (batch_size, h_dim)

#     rand_sub_context_size: int = np.random.randint(1, x_data.shape[1] + 1)

#     x_context = x_data[:, 0:rand_sub_context_size, :]
#     y_context = y_data[:, 0:rand_sub_context_size, :]

#     context = torch.cat([x_context, y_context], dim=-1)
#     # (batch_size, context_size, x_dim + y_dim)

#     context = set_encoder(context)
#     # (batch_size, h_dim)

#     p_z_T = target_constructor(context)
#     log_w, z_samples = dvi_process.run_chain(p_z_T, data)

#     log_like_data: Tensor = (
#         decoder(x_data, z_samples[-1], data).log_prob(y_data).mean(dim=0).sum()
#     )

#     log_like_context: Tensor = (
#         decoder(x_context, z_samples[-1], context).log_prob(y_context).mean(dim=0).sum()
#     )

#     loss = -log_like_data - log_like_context - log_w

#     return loss
