from typing import Callable, List

import numpy as np
import torch
import wandb
from dvi_process import DiffusionVIProcess
from torch import Tensor
from torch.distributions import Distribution, Normal
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm


def train(
    dvi_process: DiffusionVIProcess,
    device: torch.device,
    num_epochs: int,
    dataloader: DataLoader[Tensor],
    target_constructor: Callable[[Tensor], Distribution],
    optimizer: Optimizer,
    scheduler: LRScheduler | None = None,
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

                loss = step(dvi_process, batch.to(device), target_constructor)

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
    batch: Tensor,
    target_constructor: Callable[[Tensor], Distribution],
) -> Tensor:

    random_context_size: int = np.random.randint(1, batch.shape[1] + 1)
    context = batch[:, 0:random_context_size, :]

    p_z_T = target_constructor(context)
    log_w, _ = dvi_process.run_chain(p_z_T, context)

    loss = -log_w

    return loss
