from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple

import torch
from torch import Generator, Tensor
from torch.distributions.normal import Normal
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader, Dataset

from src.architectures.cnp import BCACNP, AggrCNP
from src.evaluation.predictive.pred_metrics import compute_lmpl, compute_mse
from src.training.abstract_trainer import AbstractTrainer


class CNPTrainer(AbstractTrainer, ABC):
    def __init__(
        self,
        model: AggrCNP | BCACNP,
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
        val_grad_off: bool = True,
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
        assert isinstance(self.model, AggrCNP | BCACNP)

        data, x_data, y_data = self.get_data_subtasks(batch)
        # (batch_size, num_subtasks, data_size, x_dim + y_dim)
        # (batch_size, num_subtasks, data_size, x_dim)
        # (batch_size, num_subtasks, data_size, y_dim)

        mask = self.get_mask(None, data)
        # (batch_size, num_subtasks, data_size)

        context, _, _ = self.get_context(data, x_data, mask)
        # (batch_size, num_subtasks, data_size, x_dim + y_dim)
        # (batch_size, num_subtasks, data_size, x_dim)
        # (batch_size, num_subtasks, data_size, y_dim)

        y_dist_data = self.model(context, mask, x_data)
        # (batch_size, num_subtasks, data_size, y_dim)

        lmpl = compute_lmpl(y_dist_data, y_data)
        mse = compute_mse(y_dist_data, y_data)

        return {"lmpl": lmpl.item(), "mse": mse.item()}


class CNPTrainerData(CNPTrainer):
    def __init__(
        self,
        model: AggrCNP | BCACNP,
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
        val_grad_off: bool = True,
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

    def cnp_loss_data(self, y_dist_data: Normal, y_data: Tensor) -> Tensor:
        # (batch_size, num_subtasks, data_size, y_dim)
        # (batch_size, num_subtasks, data_size, y_dim)

        log_like: Tensor = y_dist_data.log_prob(y_data)
        # (batch_size, num_subtasks, data_size, y_dim)

        return -log_like.sum(dim=-1).sum(dim=-1).mean()

    def train_step(
        self, batch: Tensor, alpha: float | None
    ) -> Tuple[Tensor, Dict[str, float]]:
        data, x_data, y_data = self.get_data_subtasks(batch)
        # (batch_size, num_subtasks, data_size, x_dim + y_dim)
        # (batch_size, num_subtasks, data_size, x_dim)
        # (batch_size, num_subtasks, data_size, y_dim)

        mask = self.get_mask(alpha, data)
        # (batch_size, num_subtasks, data_size)

        context, _, _ = self.get_context(data, x_data, mask)
        # (batch_size, num_subtasks, data_size, x_dim + y_dim)
        # (batch_size, num_subtasks, data_size, x_dim)
        # (batch_size, num_subtasks, data_size, y_dim)

        y_dist_data = self.model(context, mask, x_data)
        # (batch_size, num_subtasks, data_size, y_dim)

        loss = self.cnp_loss_data(y_dist_data, y_data)

        lmpl = compute_lmpl(y_dist_data, y_data)
        mse = compute_mse(y_dist_data, y_data)

        return loss, {"lmpl": lmpl.item(), "mse": mse.item()}


class CNPTrainerTarget(CNPTrainer):
    def __init__(
        self,
        model: AggrCNP | BCACNP,
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
        val_grad_off: bool = True,
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

    def cnp_loss_target(
        self, y_dist_target: Normal, y_target: Tensor, mask: Tensor | None
    ) -> Tensor:
        # (batch_size, num_subtasks, data_size, y_dim)
        # (batch_size, num_subtasks, data_size, y_dim)
        # (batch_size, num_subtasks, data_size)

        log_like: Tensor = y_dist_target.log_prob(y_target)
        # (batch_size, num_subtasks, data_size, y_dim)

        if mask is not None:
            mask = mask.unsqueeze(-1).expand(-1, -1, -1, log_like.shape[-1])
            # (batch_size, num_subtasks, data_size, y_dim)

            log_like = log_like * mask

        return -log_like.sum(dim=-1).sum(dim=-1).mean()

    def train_step(
        self, batch: Tensor, alpha: float | None
    ) -> Tuple[Tensor, Dict[str, float]]:
        data, x_data, y_data = self.get_data_subtasks(batch)
        # (batch_size, num_subtasks, data_size, x_dim + y_dim)
        # (batch_size, num_subtasks, data_size, x_dim)
        # (batch_size, num_subtasks, data_size, y_dim)

        mask = self.get_mask(alpha, data)
        # (batch_size, num_subtasks, data_size)

        context, _, _ = self.get_context(data, x_data, mask)
        _, x_target, y_target = self.get_target(data, x_data, mask)
        # (batch_size, num_subtasks, data_size, x_dim + y_dim)
        # (batch_size, num_subtasks, data_size, x_dim)
        # (batch_size, num_subtasks, data_size, y_dim)

        y_dist_target = self.model(context, mask, x_target)
        # (batch_size, num_subtasks, data_size, y_dim)

        loss = self.cnp_loss_target(y_dist_target, y_target, 1 - mask)

        with torch.no_grad():
            y_dist_data = self.model(
                context.clone().detach(), mask.clone().detach(), x_data.clone().detach()
            )  # (batch_size, num_subtasks, data_size, y_dim)

        lmpl = compute_lmpl(y_dist_data, y_data)
        mse = compute_mse(y_dist_data, y_data)

        return loss, {"lmpl": lmpl.item(), "mse": mse.item()}


class CNPTrainerContext(CNPTrainer):
    def __init__(
        self,
        model: AggrCNP | BCACNP,
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
        val_grad_off: bool = True,
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

    def cnp_loss_context(
        self, y_dist_context: Normal, y_context: Tensor, mask: Tensor | None
    ) -> Tensor:
        # (batch_size, num_subtasks, data_size, y_dim)
        # (batch_size, num_subtasks, data_size, y_dim)
        # (batch_size, num_subtasks, data_size)

        log_like: Tensor = y_dist_context.log_prob(y_context)
        # (batch_size, num_subtasks, data_size, y_dim)

        if mask is not None:
            mask = mask.unsqueeze(-1).expand(-1, -1, -1, log_like.shape[-1])
            # (batch_size, num_subtasks, data_size, y_dim)

            log_like = log_like * mask

        return -log_like.sum(dim=-1).sum(dim=-1).mean()

    def train_step(
        self, batch: Tensor, alpha: float | None
    ) -> Tuple[Tensor, Dict[str, float]]:
        data, x_data, y_data = self.get_data_subtasks(batch)
        # (batch_size, num_subtasks, data_size, x_dim + y_dim)
        # (batch_size, num_subtasks, data_size, x_dim)
        # (batch_size, num_subtasks, data_size, y_dim)

        mask = self.get_mask(alpha, data)
        # (batch_size, num_subtasks, data_size)

        context, x_context, y_context = self.get_context(data, x_data, mask)
        # (batch_size, num_subtasks, data_size, x_dim + y_dim)
        # (batch_size, num_subtasks, data_size, x_dim)
        # (batch_size, num_subtasks, data_size, y_dim)

        y_dist_context = self.model(context, mask, x_context)
        # (batch_size, num_subtasks, data_size, y_dim)

        loss = self.cnp_loss_context(y_dist_context, y_context, mask)

        with torch.no_grad():
            y_dist_data = self.model(
                context.clone().detach(), mask.clone().detach(), x_data.clone().detach()
            )  # (batch_size, num_subtasks, data_size, y_dim)

        lmpl = compute_lmpl(y_dist_data, y_data)
        mse = compute_mse(y_dist_data, y_data)

        return loss, {"lmpl": lmpl.item(), "mse": mse.item()}


# class NoisyCNPTrainer(BaseTrainer):
#     def __init__(
#         self,
#         model: AggrCNP | BcaCNP,
#         device: torch.device,
#         dataset: Dataset[Any],
#         train_loader: DataLoader[Any],
#         val_loader: DataLoader[Any],
#         optimizer: Optimizer,
#         scheduler: LRScheduler | None,
#         wandb_logging: bool,
#         num_subtasks: int,
#         num_samples: int,
#         val_grad_off: bool = True,
#     ) -> None:
#         super().__init__(
#             device,
#             model,
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
#         x_data, y_data = batch
#         x_data = x_data.to(self.device)
#         y_data = y_data.to(self.device)
#         # (batch_size, data_size, x_dim)
#         # (batch_size, data_size, y_dim)

#         x_data = x_data.unsqueeze(1)
#         y_data = y_data.unsqueeze(1)
#         # (batch_size, 1, data_size, x_dim)
#         # (batch_size, 1, data_size, y_dim)

#         context_size: int = np.random.randint(1, x_data.shape[1] + 1)
#         x_context = x_data[:, :, 0:context_size, :]
#         y_context = y_data[:, :, 0:context_size, :]
#         # (batch_size, 1, context_size, x_dim)
#         # (batch_size, 1, context_size, y_dim)

#         context = torch.cat([x_context, y_context], dim=-1)
#         # (batch_size, 1, context_size, x_dim + y_dim)

#         output = self.model(context, None, x_context)
#         loss = self.model.loss(*output, y_target=y_context, mask=None)

#         return loss, {}

#     def val_step(
#         self, batch: Tensor, ranges: List[Tuple[float, float]]
#     ) -> Dict[str, float]:
#         raise NotImplementedError
