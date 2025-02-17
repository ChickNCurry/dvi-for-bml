from typing import Any, Dict, Tuple

import numpy as np
import torch
from torch import Tensor
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader, Dataset

from src.components.dvi import DVI
from src.train.base_trainer import BaseTrainer


class DVITrainer(BaseTrainer):
    def __init__(
        self,
        model: DVI,
        device: torch.device,
        dataset: Dataset[Any],
        train_loader: DataLoader[Any],
        val_loader: DataLoader[Any],
        optimizer: Optimizer,
        scheduler: LRScheduler | None,
        wandb_logging: bool,
        num_subtasks: int,
        sample_size: int,
    ) -> None:
        super().__init__(
            model,
            device,
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
        assert isinstance(self.model, DVI)

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

        target = self.model.contextual_target(context, mask)
        # (batch_size, num_subtasks, z_dim)

        r = self.model.encoder(context, mask)
        # (batch_size, num_subtasks, c_dim)

        elbo, _ = self.model.cdvi.run_both_processes(target, r, mask)

        loss = -elbo

        return loss, {}

    def val_step(self, batch: Tensor) -> Dict[str, float]:
        raise NotImplementedError


# class NoisyDVITrainer(BaseTrainer):
#     def __init__(
#         self,
#         model: DVI,
#         device: torch.device,
#         dataset: Dataset[Any],
#         train_loader: DataLoader[Any],
#         val_loader: DataLoader[Any],
#         optimizer: Optimizer,
#         scheduler: LRScheduler | None,
#         wandb_logging: bool,
#         num_subtasks: int,
#         sample_size: int,
#     ) -> None:
#         super().__init__(
#             model,
#             device,
#             dataset,
#             train_loader,
#             val_loader,
#             optimizer,
#             scheduler,
#             wandb_logging,
#             num_subtasks,
#             sample_size,
#         )

#     def train_step(
#         self, batch: Tensor, alpha: float | None
#     ) -> Tuple[Tensor, Dict[str, float]]:
#         assert isinstance(self.model, DVI)

#         batch = batch.to(self.device).unsqueeze(1)
#         # (batch_size, 1, context_size, c_dim)

#         rand_context_size: int = np.random.randint(1, batch.shape[1] + 1)
#         context = batch[:, :, 0:rand_context_size, :]
#         # (batch_size, 1, context_size, c_dim)

#         target = self.model.contextual_target(context, None)

#         r = self.model.encoder(context, None)
#         # (batch_size, 1, h_dim)

#         elbo, _, _ = self.model.cdvi.run_both_processes(target, r, None)

#         loss = -elbo

#         return loss, {}

#     def val_step(self, batch: Tensor) -> Dict[str, float]:
#         raise NotImplementedError
