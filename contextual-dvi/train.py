from typing import List, Tuple

import torch
import wandb
from contextual_gaussian import contextual_gaussian_tuple
from dvi_process import DiffusionVIProcess, zTuple
from torch import Tensor
from torch.distributions import Normal
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm


def train(
    dvi_process: DiffusionVIProcess,
    device: torch.device,
    dataloader: DataLoader[Tensor],
    optimizer: Optimizer,
    num_epochs: int,
    wandb_logging: bool = True,
) -> List[float]:

    # torch.autograd.set_detect_anomaly(True)

    dvi_process = dvi_process.to(device)

    losses = []

    for epoch in range(num_epochs):

        dvi_process.train()
        with torch.inference_mode(False):

            loop = tqdm(dataloader, total=len(dataloader))

            for batch in loop:

                loss = step(dvi_process, device, batch.to(device))

                optimizer.zero_grad()
                loss.backward()  # type: ignore
                optimizer.step()

                loop.set_postfix(
                    epoch=epoch,
                    loss=loss.item(),
                )

                losses.append(loss.item())

                if wandb_logging:
                    wandb.log({"train/loss": loss.item()})

    return losses


def step(
    dvi_process: DiffusionVIProcess,
    device: torch.device,
    batch: Tensor,
) -> Tensor:

    batch_size = batch.shape[0]

    z_0_mu = torch.zeros((batch_size, dvi_process.z_dim), device=device)
    z_0_sigma = torch.ones((batch_size, dvi_process.z_dim), device=device)
    z_0 = torch.normal(z_0_mu, z_0_sigma).to(device)
    z_0_tuple = zTuple(z_0, z_0_mu, z_0_sigma)

    z_T_tuple = contextual_gaussian_tuple(batch)

    z_tuples_forward = dvi_process.forward_process(z_0_tuple, batch, None)
    z_samples_forward = [z_tuple.z for z_tuple in z_tuples_forward]

    z_tuples_backward = dvi_process.backward_process(z_T_tuple, z_samples_forward)

    forward_log_like = torch.stack(
        [
            Normal(z_tuples_forward[i].z_mu, z_tuples_forward[i].z_sigma).log_prob(z_tuples_forward[i].z).mean(dim=0).sum()  # type: ignore
            for i in range(len(z_tuples_forward))
        ]
    ).sum()

    backward_log_like = torch.stack(
        [
            Normal(z_tuples_backward[i].z_mu, z_tuples_backward[i].z_sigma).log_prob(z_tuples_backward[i].z).mean(dim=0).sum()  # type: ignore
            for i in range(len(z_tuples_backward))
        ]
    ).sum()

    loss = -backward_log_like + forward_log_like

    return loss
