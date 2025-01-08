from typing import Any, Dict, Tuple

import numpy as np
import torch
from torch import Tensor
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader, Dataset

from src.components.dvi_np import DVINP
from src.train.train import Trainer


class StaticTargetTrainer(Trainer):
    def __init__(
        self,
        device: torch.device,
        dvinp: DVINP,
        dataset: Dataset[Any],
        train_loader: DataLoader[Any],
        val_loader: DataLoader[Any],
        optimizer: Optimizer,
        scheduler: ReduceLROnPlateau | None,
        wandb_logging: bool,
        num_subtasks: int,
        sample_size: int,
    ) -> None:
        super().__init__(
            device,
            dvinp,
            dataset,
            train_loader,
            val_loader,
            optimizer,
            scheduler,
            wandb_logging,
            num_subtasks,
            sample_size,
        )

    def train_step(
        self, batch: Tensor, alpha: float | None
    ) -> Tuple[Tensor, Dict[str, float]]:
        assert self.dvinp.contextual_target is not None

        batch = batch.to(self.device).unsqueeze(1)
        # (batch_size, 1, context_size, c_dim)

        random_context_size: int = np.random.randint(1, batch.shape[1] + 1)
        context = batch[:, :, 0:random_context_size, :]
        # (batch_size, 1, context_size, c_dim)

        target = self.dvinp.contextual_target(context, None)

        r_aggr, r_non_aggr = self.dvinp.encoder(context, None)
        # (batch_size, 1, h_dim)
        # (batch_size, 1, context_size, h_dim)

        elbo, _, _ = self.dvinp.cdvi.run_chain(target, r_aggr, r_non_aggr, None)

        loss = -elbo

        return loss, {}

    def val_step(self, batch: Tensor) -> Dict[str, float]:
        raise NotImplementedError


class BetterStaticTargetTrainer(Trainer):
    def __init__(
        self,
        device: torch.device,
        dvinp: DVINP,
        dataset: Dataset[Any],
        train_loader: DataLoader[Any],
        val_loader: DataLoader[Any],
        optimizer: Optimizer,
        scheduler: ReduceLROnPlateau | None,
        wandb_logging: bool,
        num_subtasks: int,
        sample_size: int,
    ) -> None:
        super().__init__(
            device,
            dvinp,
            dataset,
            train_loader,
            val_loader,
            optimizer,
            scheduler,
            wandb_logging,
            num_subtasks,
            sample_size,
        )

    def train_step(
        self, batch: Tensor, alpha: float | None
    ) -> Tuple[Tensor, Dict[str, float]]:
        assert self.dvinp.contextual_target is not None

        batch = batch.to(self.device).unsqueeze(1).expand(-1, self.num_subtasks, -1, -1)
        # (batch_size, num_subtasks, context_size, c_dim)

        rand_context_sizes = torch.randint(
            low=1,
            high=batch.shape[2] + 1,
            size=(batch.shape[0], batch.shape[1], 1),
            device=self.device,
        )  # (batch_size, num_subtasks, 1)

        pos_indices = (
            torch.arange(batch.shape[2], device=self.device)
            .unsqueeze(0)
            .unsqueeze(0)
            .expand(batch.shape[0], batch.shape[1], -1)
        )  # (batch_size, num_subtasks, context_size)

        mask = (pos_indices < rand_context_sizes).float()
        # (batch_size, num_subtasks, context_size)

        context = batch * mask.unsqueeze(-1).expand(-1, -1, -1, batch.shape[-1])
        # (batch_size, num_subtasks, context_size, c_dim)

        target = self.dvinp.contextual_target(context, mask)
        # (batch_size, num_subtasks, z_dim)

        r_aggr, r_non_aggr = self.dvinp.encoder(context, mask)
        # (batch_size, num_subtasks, c_dim)
        # (batch_size, num_subtasks, context_size, c_dim)

        elbo, _, _ = self.dvinp.cdvi.run_chain(target, r_aggr, r_non_aggr, mask)

        loss = -elbo

        return loss, {}

    def val_step(self, batch: Tensor) -> Dict[str, float]:
        raise NotImplementedError
