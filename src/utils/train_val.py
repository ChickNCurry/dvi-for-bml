from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple

import torch
import wandb
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.components.cdvi import ContextualDVI


class TrainerAndValidater(ABC):
    def __init__(
        self,
        device: torch.device,
        cdvi: ContextualDVI,
        train_loader: DataLoader[Any],
        val_loader: DataLoader[Any],
        optimizer: Optimizer,
        scheduler: ReduceLROnPlateau | None,
        wandb_logging: bool,
    ) -> None:
        self.device = device
        self.cdvi = cdvi
        self.train_dataloader = train_loader
        self.val_dataloader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.wandb_logging = wandb_logging

    def train_and_validate(
        self,
        num_epochs: int,
        max_clip_norm: float | None,
        alpha: float | None,
        validate: bool = False,
    ) -> List[float]:

        # torch.autograd.set_detect_anomaly(True)

        losses = []

        for epoch in range(num_epochs):

            self.cdvi.train()
            with torch.inference_mode(False):

                loop = tqdm(self.train_dataloader, total=len(self.train_dataloader))

                for batch in loop:

                    self.optimizer.zero_grad()

                    loss = self.train_step(batch, alpha)
                    loss.backward()  # type: ignore

                    if max_clip_norm is not None:
                        torch.nn.utils.clip_grad_norm_(
                            self.cdvi.parameters(), max_clip_norm
                        )

                    self.optimizer.step()

                    if self.scheduler is not None:
                        self.scheduler.step(loss)

                    loop.set_postfix(
                        epoch=epoch,
                        loss=loss.item(),
                        # lr=(
                        #     self.scheduler.get_last_lr()
                        #     if self.scheduler is not None
                        #     else self.optimizer.param_groups[0]["lr"]
                        # ),
                    )

                    losses.append(loss.item())

                    if self.wandb_logging:
                        wandb.log({"train/loss": loss.item()})

            if not validate:
                continue

            self.cdvi.eval()
            with torch.inference_mode(False):

                loop = tqdm(self.val_dataloader, total=len(self.val_dataloader))

                for batch in loop:

                    metrics = self.val_step(batch)

                    # loop.set_postfix(**metrics)

                    losses.append(loss.item())

                    if self.wandb_logging:
                        wandb.log(
                            {
                                **{"val/" + k: v for k, v in metrics.items()},
                            }
                        )

        return losses

    @abstractmethod
    def train_step(self, batch: Tensor, alpha: float | None) -> Tensor:
        pass

    @abstractmethod
    def val_step(self, batch: Tensor) -> Dict[str, Tensor]:
        pass
