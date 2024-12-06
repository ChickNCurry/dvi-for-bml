from typing import Any, Dict, Tuple

import numpy as np
import torch
from torch import Tensor
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader

from src.components.cdvi import CDVI
from src.train.train import Trainer


class StaticTargetTrainer(Trainer):
    def __init__(
        self,
        device: torch.device,
        cdvi: CDVI,
        train_loader: DataLoader[Any],
        val_loader: DataLoader[Any],
        optimizer: Optimizer,
        scheduler: ReduceLROnPlateau | None,
        wandb_logging: bool,
    ) -> None:
        super().__init__(
            device,
            cdvi,
            train_loader,
            val_loader,
            optimizer,
            scheduler,
            wandb_logging,
        )

    def train_step(
        self, batch: Tensor, alpha: float | None
    ) -> Tuple[Tensor, Dict[str, float]]:
        assert self.cdvi.contextual_target is not None

        batch = batch.to(self.device)

        random_context_size: int = np.random.randint(1, batch.shape[1] + 1)
        context = batch[:, 0:random_context_size, :]

        p_z_T = self.cdvi.contextual_target(context, None)

        aggregated, _ = self.cdvi.encoder(context, None)
        elbo, _, _ = self.cdvi.dvi_process.run_chain(p_z_T, aggregated, None)

        loss = -elbo

        return loss, {}

    def val_step(self, batch: Tensor) -> Dict[str, float]:
        raise NotImplementedError


class BetterStaticTargetTrainer(Trainer):
    def __init__(
        self,
        device: torch.device,
        cdvi: CDVI,
        train_loader: DataLoader[Any],
        val_loader: DataLoader[Any],
        optimizer: Optimizer,
        scheduler: ReduceLROnPlateau | None,
        wandb_logging: bool,
    ) -> None:
        super().__init__(
            device,
            cdvi,
            train_loader,
            val_loader,
            optimizer,
            scheduler,
            wandb_logging,
        )

    def train_step(
        self, batch: Tensor, alpha: float | None
    ) -> Tuple[Tensor, Dict[str, float]]:
        assert self.cdvi.contextual_target is not None

        batch = batch.to(self.device)

        rand_context_sizes = torch.randint(
            low=1, high=batch.shape[1] + 1, size=(batch.shape[0],), device=self.device
        )

        position_indices = torch.arange(batch.shape[1], device=self.device).expand(
            batch.shape[0], -1
        )

        mask = (position_indices < rand_context_sizes.unsqueeze(-1)).float()
        context = batch * mask.unsqueeze(-1).expand(-1, -1, batch.shape[2])

        p_z_T = self.cdvi.contextual_target(context, mask)

        aggregated, _ = self.cdvi.encoder(context, mask)
        elbo, _, _ = self.cdvi.dvi_process.run_chain(p_z_T, aggregated, mask)

        loss = -elbo

        return loss, {}

    def val_step(self, batch: Tensor) -> Dict[str, float]:
        raise NotImplementedError
