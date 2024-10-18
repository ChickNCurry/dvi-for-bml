from typing import List

import torch
import wandb
from contextual_gaussian import ContextualGMM, ContextualGaussian
from dvi_process import DiffusionVIProcess
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

                    # for i, sigma in enumerate(dvi_process.sigmas):
                    #     wandb.log({"hyperparams/sigma_" + str(i): sigma.data.item()})

                    # for i, beta in enumerate(dvi_process.betas):
                    #     wandb.log({"hyperparams/beta_" + str(i): beta.data.item()})

    return losses


def step(
    dvi_process: DiffusionVIProcess,
    device: torch.device,
    batch: Tensor,
) -> Tensor:

    p_z_0 = Normal(  # type: ignore
        torch.zeros((batch.shape[0], dvi_process.z_dim), device=device),
        torch.ones((batch.shape[0], dvi_process.z_dim), device=device),
        # * dvi_process.sigmas[0],
    )

    p_z_T = ContextualGaussian(batch)

    log_w, _ = dvi_process.run_chain(p_z_0, p_z_T, batch)

    loss = -log_w

    return loss
