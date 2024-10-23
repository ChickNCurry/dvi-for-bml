from typing import Callable, List

import torch
import wandb
from dvi_process import DiffusionVIProcess
from torch import Tensor
from torch.distributions import Distribution, Normal
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np


def train(
    dvi_process: DiffusionVIProcess,
    device: torch.device,
    dataloader: DataLoader[Tensor],
    optimizer: Optimizer,
    num_epochs: int,
    target: Callable[[Tensor], Distribution],
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

                loss = step(dvi_process, device, batch.to(device), target)

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

                    # for i, sigma in enumerate(dvi_process.sigmas):
                    #     wandb.log({"hyperparams/sigma_" + str(i): sigma.data.item()})

                    # for i, beta in enumerate(dvi_process.betas):
                    #     wandb.log({"hyperparams/beta_" + str(i): beta.data.item()})

    return losses


def step(
    dvi_process: DiffusionVIProcess,
    device: torch.device,
    batch: Tensor,
    target: Callable[[Tensor], Distribution],
) -> Tensor:

    p_z_0 = Normal(  # type: ignore
        torch.zeros((batch.shape[0], dvi_process.z_dim), device=device),
        torch.ones((batch.shape[0], dvi_process.z_dim), device=device),
        # * dvi_process.sigmas[0],
    )

    context_size: int = np.random.randint(1, batch.shape[1])
    batch = batch[:, 0:context_size, :]

    p_z_T = target(batch)

    log_w, _ = dvi_process.run_chain(p_z_0, p_z_T, batch)

    loss = -log_w

    return loss
