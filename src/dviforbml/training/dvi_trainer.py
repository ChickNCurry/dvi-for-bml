from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple

import torch
from torch import Generator, Tensor
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader, Dataset

from dviforbml.architectures.dvi import DVI
from dviforbml.training.abstract_trainer import AbstractTrainer


class DVITrainer(AbstractTrainer, ABC):
    def __init__(
        self,
        model: DVI,
        device: torch.device,
        dataset: Dataset[Any],
        train_loader: DataLoader[Any],
        val_loader: DataLoader[Any],
        optimizer: Optimizer,
        scheduler: LRScheduler | None,
        generator: Generator,
        wandb_logging: bool,
        num_subtasks: int,
        num_samples: int,
        val_grad_off: bool,
    ) -> None:
        super().__init__(
            model,
            device,
            dataset,
            train_loader,
            val_loader,
            optimizer,
            scheduler,
            generator,
            wandb_logging,
            num_subtasks,
            num_samples,
            val_grad_off,
        )

    @abstractmethod
    def train_step(
        self, batch: Tensor, alpha: float | None
    ) -> Tuple[Tensor, Dict[str, float]]:
        pass

    def val_step(self, batch: Tensor) -> Dict[str, float]:
        raise NotImplementedError


class DVITrainerContext(DVITrainer):
    def __init__(
        self,
        model: DVI,
        device: torch.device,
        dataset: Dataset[Any],
        train_loader: DataLoader[Any],
        val_loader: DataLoader[Any],
        optimizer: Optimizer,
        scheduler: LRScheduler | None,
        generator: Generator,
        wandb_logging: bool,
        num_subtasks: int,
        num_samples: int,
        val_grad_off: bool,
    ) -> None:
        super().__init__(
            model,
            device,
            dataset,
            train_loader,
            val_loader,
            optimizer,
            scheduler,
            generator,
            wandb_logging,
            num_subtasks,
            num_samples,
            val_grad_off,
        )

    def train_step(
        self, batch: Tensor, alpha: float | None
    ) -> Tuple[Tensor, Dict[str, float]]:
        assert isinstance(self.model, DVI)

        data = batch.to(self.device).unsqueeze(1).expand(-1, self.num_subtasks, -1, -1)
        # (batch_size, num_subtasks, data_size, c_dim)

        mask = self.get_mask(alpha, data)
        # (batch_size, num_subtasks, data_size)

        context = batch * mask.unsqueeze(-1).expand(-1, -1, -1, data.shape[-1])
        # (batch_size, num_subtasks, data_size, c_dim)

        target_dist = self.model.contextual_target(context, mask)
        # (batch_size, num_subtasks, z_dim)

        r = self.model.encoder(context, mask)
        # (batch_size, num_subtasks, c_dim)

        elbo, _ = self.model.cdvi.run_both_processes(target_dist, r, mask)

        loss = -elbo

        return loss, {}


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
#         num_samples: int,
#         val_grad_off: bool,
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
#             num_samples,
#             val_grad_off,
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
