from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple

import torch
from torch import Tensor
from torch.distributions.normal import Normal
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader, Dataset

from src.components.lnp import AggrLNP, BcaLNP
from src.eval.metrics import compute_lmpl, compute_mse
from src.train.base_trainer import BaseTrainer


class LNPTrainer(BaseTrainer, ABC):
    def __init__(
        self,
        model: AggrLNP | BcaLNP,
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
            device,
            model,
            dataset,
            train_loader,
            val_loader,
            optimizer,
            scheduler,
            wandb_logging,
            num_subtasks,
            sample_size,
        )

    @abstractmethod
    def train_step(
        self, batch: Tensor, alpha: float | None
    ) -> Tuple[Tensor, Dict[str, float]]:
        pass

    def val_step(
        self, batch: Tensor, ranges: List[Tuple[float, float]]
    ) -> Dict[str, float]:
        raise NotImplementedError


class LNPTrainerData(LNPTrainer):
    def __init__(
        self,
        model: AggrLNP | BcaLNP,
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

    def lnp_loss_data(
        self,
        y_dist_data: Normal,
        y_data: Normal,
        z_dist_data: Normal,
        z_data: Normal,
        z_dist_context: Normal,
    ) -> Tensor:
        # (batch_size, num_subtasks, data_size, y_dim)
        # (batch_size, num_subtasks, data_size, y_dim)
        # (batch_size, num_subtasks, z_dim)
        # (batch_size, num_subtasks, z_dim)
        # (batch_size, num_subtasks, z_dim)

        log_like: Tensor = y_dist_data.log_prob(y_data)
        log_like = log_like.sum(dim=-1).sum(dim=-1, keepdim=True)
        # (batch_size, num_subtasks, 1)

        log_tp_context: Tensor = z_dist_context.log_prob(z_data)
        log_tp_context.sum(dim=-1, keepdim=True)
        # (batch_size, num_subtasks, 1)

        log_tp_data: Tensor = z_dist_data.log_prob(z_data)
        log_tp_data.sum(dim=-1, keepdim=True)
        # (batch_size, num_subtasks, 1)

        return -(log_like + log_tp_context - log_tp_data).sum(dim=-1).mean()

    def train_step(
        self, batch: Tensor, alpha: float | None
    ) -> Tuple[Tensor, Dict[str, float]]:
        assert isinstance(self.model, AggrLNP | BcaLNP)

        data, x_data, y_data = self.get_data(batch)
        # (batch_size, num_subtasks, data_size, x_dim + y_dim)
        # (batch_size, num_subtasks, data_size, x_dim)
        # (batch_size, num_subtasks, data_size, y_dim)

        mask = self.get_mask(alpha, data)
        # (batch_size, num_subtasks, data_size)

        context, _, _ = self.get_context(data, x_data, mask)
        # (batch_size, num_subtasks, data_size, x_dim + y_dim)
        # (batch_size, num_subtasks, data_size, x_dim)
        # (batch_size, num_subtasks, data_size, y_dim)

        y_dist_data, z_dist_data, z_data = self.model(data, mask, x_data)
        # (batch_size, num_subtasks, data_size, y_dim)

        z_dist_context, _ = self.model.encode(context, mask)
        # (batch_size, num_subtasks, z_dim)

        loss = self.lnp_loss_data(
            y_dist_data, y_data, z_dist_data, z_data, z_dist_context
        )

        lmpl = compute_lmpl(y_dist_data, y_data)
        mse = compute_mse(y_dist_data, y_data)

        return loss, {"lmpl": lmpl.item(), "mse": mse.item()}


class LNPTrainerTarget(LNPTrainer):
    def __init__(
        self,
        model: AggrLNP | BcaLNP,
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

    def lnp_loss_target(
        self,
        y_dist_target: Normal,
        y_target: Normal,
        mask: Tensor,
        z_dist_data: Normal,
        z_data: Normal,
        z_dist_context: Normal,
    ) -> Tensor:
        # (batch_size, num_subtasks, data_size, y_dim)
        # (batch_size, num_subtasks, data_size, y_dim)
        # (batch_size, num_subtasks, z_dim)
        # (batch_size, num_subtasks, z_dim)
        # (batch_size, num_subtasks, z_dim)

        log_like: Tensor = y_dist_target.log_prob(y_target)

        if mask is not None:
            mask = mask.unsqueeze(-1).expand(-1, -1, -1, log_like.shape[-1])
            mask = 1 - mask
            # (batch_size, num_subtasks, data_size, y_dim)

            log_like = log_like * mask

        log_like = log_like.sum(dim=-1).sum(dim=-1, keepdim=True)
        # (batch_size, num_subtasks, 1)

        log_tp_context: Tensor = z_dist_context.log_prob(z_data)
        log_tp_context.sum(dim=-1, keepdim=True)
        # (batch_size, num_subtasks, 1)

        log_tp_data: Tensor = z_dist_data.log_prob(z_data)
        log_tp_data.sum(dim=-1, keepdim=True)
        # (batch_size, num_subtasks, 1)

        return -(log_like + log_tp_context - log_tp_data).sum(dim=-1).mean()

    def train_step(
        self, batch: Tensor, alpha: float | None
    ) -> Tuple[Tensor, Dict[str, float]]:
        assert isinstance(self.model, AggrLNP | BcaLNP)

        data, x_data, y_data = self.get_data(batch)
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

        y_dist_target, z_dist_data, z_data = self.model(data, mask, x_target)
        # (batch_size, num_subtasks, data_size, y_dim)

        z_dist_context, _ = self.model.encode(context, mask)
        # (batch_size, num_subtasks, z_dim)

        loss = self.lnp_loss_target(
            y_dist_target, y_target, mask, z_dist_data, z_data, z_dist_context
        )

        with torch.no_grad():
            y_dist_data, _, _ = self.model(data, mask, x_data)
            # (batch_size, num_subtasks, data_size, y_dim)

        lmpl = compute_lmpl(y_dist_data, y_data)
        mse = compute_mse(y_dist_data, y_data)

        return loss, {"lmpl": lmpl.item(), "mse": mse.item()}


class LNPTrainerContext(LNPTrainer):
    def __init__(
        self,
        model: AggrLNP | BcaLNP,
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

    def lnp_loss_context(
        self,
        y_dist_context: Normal,
        y_context: Tensor,
        mask: Tensor | None,
        z_dist_context: Normal,
        z_context: Tensor,
    ) -> Tensor:
        # (batch_size, num_subtasks, data_size, y_dim)
        # (batch_size, num_subtasks, data_size, y_dim)
        # (batch_size, num_subtasks, data_size)
        # (batch_size, num_subtasks, z_dim)
        # (batch_size, num_subtasks, z_dim)

        batch_size = y_context.shape[0]
        num_subtasks = y_context.shape[1]
        device = y_context.device
        z_dim = z_context.shape[-1]

        log_like: Tensor = y_dist_context.log_prob(y_context)
        # (batch_size, num_subtasks, data_size, y_dim)

        if mask is not None:
            mask = mask.unsqueeze(-1).expand(-1, -1, -1, log_like.shape[-1])
            # (batch_size, num_subtasks, data_size, y_dim)

            log_like = log_like * mask

        log_like = log_like.sum(dim=-1).sum(dim=-1, keepdim=True)
        # (batch_size, num_subtasks, 1)

        prior_dist = Normal(  # type: ignore
            torch.zeros((batch_size, num_subtasks, z_dim), device=device),
            torch.ones((batch_size, num_subtasks, z_dim), device=device),
        )

        log_prior: Tensor = prior_dist.log_prob(z_context)  # type: ignore
        log_prior = log_prior.sum(dim=-1, keepdim=True)
        # (batch_size, num_subtasks, 1)

        log_tp: Tensor = z_dist_context.log_prob(z_context)
        log_tp = log_tp.sum(dim=-1, keepdim=True)
        # (batch_size, num_subtasks, 1)

        return -(log_like + log_prior - log_tp).sum(dim=-1).mean()

    def train_step(
        self, batch: Tensor, alpha: float | None
    ) -> Tuple[Tensor, Dict[str, float]]:
        assert isinstance(self.model, AggrLNP | BcaLNP)

        data, x_data, y_data = self.get_data(batch)
        # (batch_size, num_subtasks, data_size, x_dim + y_dim)
        # (batch_size, num_subtasks, data_size, x_dim)
        # (batch_size, num_subtasks, data_size, y_dim)

        mask = self.get_mask(alpha, data)
        # (batch_size, num_subtasks, data_size)

        context, x_context, y_context = self.get_context(data, x_data, mask)
        # (batch_size, num_subtasks, data_size, x_dim + y_dim)
        # (batch_size, num_subtasks, data_size, x_dim)
        # (batch_size, num_subtasks, data_size, y_dim)

        y_dist_context, z_dist_context, z_context = self.model(context, mask, x_context)
        # (batch_size, num_subtasks, data_size, y_dim)

        loss = self.lnp_loss_context(
            y_dist_context, y_context, mask, z_dist_context, z_context
        )

        with torch.no_grad():
            y_dist_data, _, _ = self.model(data, mask, x_data)
            # (batch_size, num_subtasks, data_size, y_dim)

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
#         sample_size: int,
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
#             sample_size,
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
