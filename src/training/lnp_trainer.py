from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple

import numpy as np
import torch
from torch import Generator, Tensor
from torch.distributions.kl import kl_divergence
from torch.distributions.normal import Normal
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader, Dataset

from src.architectures.lnp import BCALNP, AggrLNP
from src.evaluation.predictive.pred_metrics import compute_lmpl, compute_mse
from src.training.abstract_trainer import AbstractTrainer


class LNPTrainer(AbstractTrainer, ABC):
    def __init__(
        self,
        model: AggrLNP | BCALNP,
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
        val_grad_off: bool = False,
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
        assert isinstance(self.model, AggrLNP | BCALNP)

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

        y_dist_data, _, _ = self.model(context, mask, x_data)
        # (batch_size, num_subtasks, data_size, y_dim)

        lmpl = compute_lmpl(y_dist_data, y_data)
        mse = compute_mse(y_dist_data, y_data)

        return {"lmpl": lmpl.item(), "mse": mse.item()}


class LNPTrainerData(LNPTrainer):
    def __init__(
        self,
        model: AggrLNP | BCALNP,
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
        val_grad_off: bool = False,
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

    def lnp_loss_data_kl(
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
        log_like = log_like.sum(dim=-1).sum(dim=-1)
        # (batch_size, num_subtasks)

        kl_div = kl_divergence(z_dist_data, z_dist_context).sum(dim=-1)
        # (batch_size, num_subtasks)

        return -(log_like - kl_div).mean()

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
        log_like = log_like.sum(dim=-1).sum(dim=-1)
        # (batch_size, num_subtasks)

        log_tp_context: Tensor = z_dist_context.log_prob(z_data)
        log_tp_context.sum(dim=-1)
        # (batch_size, num_subtasks)

        log_tp_data: Tensor = z_dist_data.log_prob(z_data)
        log_tp_data.sum(dim=-1)
        # (batch_size, num_subtasks)

        return -(log_like + log_tp_context - log_tp_data).mean()

    def train_step(
        self, batch: Tensor, alpha: float | None
    ) -> Tuple[Tensor, Dict[str, float]]:
        assert isinstance(self.model, AggrLNP | BCALNP)

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

        y_dist_data, z_dist_data, z_data = self.model(data, None, x_data)
        # (batch_size, num_subtasks, data_size, y_dim)

        z_dist_context, _ = self.model.encode(context, mask)
        # (batch_size, num_subtasks, z_dim)

        loss = self.lnp_loss_data_kl(
            y_dist_data, y_data, z_dist_data, z_data, z_dist_context
        )

        lmpl = compute_lmpl(y_dist_data, y_data)
        mse = compute_mse(y_dist_data, y_data)

        return loss, {"lmpl": lmpl.item(), "mse": mse.item()}

    # def train_step_noisy(
    #     self, batch: Tensor, alpha: float | None
    # ) -> Tuple[Tensor, Dict[str, float]]:
    #     assert isinstance(self.model, AggrLNP | BcaLNP)

    #     data, x_data, y_data = self.get_data_subtasks(batch)
    #     # (batch_size, num_subtasks, data_size, x_dim + y_dim)
    #     # (batch_size, num_subtasks, data_size, x_dim)
    #     # (batch_size, num_subtasks, data_size, y_dim)

    #     context_size: int = np.random.randint(1, data.shape[1] + 1)
    #     context = data[:, :, 0:context_size, :]
    #     # (batch_size, num_subtasks, data_size, x_dim + y_dim)

    #     y_dist_data, z_dist_data, z_data = self.model(data, None, x_data)
    #     # (batch_size, num_subtasks, data_size, y_dim)

    #     z_dist_context, _ = self.model.encode(context, None)
    #     # (batch_size, num_subtasks, z_dim)

    #     loss = self.lnp_loss_data_kl(
    #         y_dist_data, y_data, z_dist_data, z_data, z_dist_context
    #     )

    #     lmpl = compute_lmpl(y_dist_data, y_data)
    #     mse = compute_mse(y_dist_data, y_data)

    #     return loss, {"lmpl": lmpl.item(), "mse": mse.item()}


class LNPTrainerTarget(LNPTrainer):
    def __init__(
        self,
        model: AggrLNP | BCALNP,
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
        val_grad_off: bool = False,
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

    def lnp_loss_target_kl(
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

        log_like = log_like.sum(dim=-1).sum(dim=-1)
        # (batch_size, num_subtasks)

        kl_div = kl_divergence(z_dist_data, z_dist_context).sum(dim=-1)
        # (batch_size, num_subtasks)

        return -(log_like - kl_div).mean()

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

        log_like = log_like.sum(dim=-1).sum(dim=-1)
        # (batch_size, num_subtasks)

        log_tp_context: Tensor = z_dist_context.log_prob(z_data)
        log_tp_context.sum(dim=-1)
        # (batch_size, num_subtasks)

        log_tp_data: Tensor = z_dist_data.log_prob(z_data)
        log_tp_data.sum(dim=-1)
        # (batch_size, num_subtasks)

        return -(log_like + log_tp_context - log_tp_data).mean()

    def train_step(
        self, batch: Tensor, alpha: float | None
    ) -> Tuple[Tensor, Dict[str, float]]:
        assert isinstance(self.model, AggrLNP | BCALNP)

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

        y_dist_target, z_dist_data, z_data = self.model(data, None, x_target)
        # (batch_size, num_subtasks, data_size, y_dim)

        z_dist_context, _ = self.model.encode(context, mask)
        # (batch_size, num_subtasks, z_dim)

        loss = self.lnp_loss_target_kl(
            y_dist_target, y_target, mask, z_dist_data, z_data, z_dist_context
        )

        with torch.no_grad():
            y_dist_data, _, _ = self.model(
                context.clone().detach(), mask.clone().detach(), x_data.clone().detach()
            )
            # (batch_size, num_subtasks, data_size, y_dim)

        lmpl = compute_lmpl(y_dist_data, y_data)
        mse = compute_mse(y_dist_data, y_data)

        return loss, {"lmpl": lmpl.item(), "mse": mse.item()}


class LNPTrainerContext(LNPTrainer):
    def __init__(
        self,
        model: AggrLNP | BCALNP,
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

        log_like = log_like.sum(dim=-1).sum(dim=-1)
        # (batch_size, num_subtasks)

        prior_dist = Normal(  # type: ignore
            torch.zeros((batch_size, num_subtasks, z_dim), device=device),
            torch.ones((batch_size, num_subtasks, z_dim), device=device),
        )  # (batch_size, num_subtasks, z_dim)

        log_prior: Tensor = prior_dist.log_prob(z_context)  # type: ignore
        log_prior = log_prior.sum(dim=-1)
        # (batch_size, num_subtasks)

        log_tp: Tensor = z_dist_context.log_prob(z_context)
        log_tp = log_tp.sum(dim=-1)
        # (batch_size, num_subtasks)

        return -(log_like + log_prior - log_tp).mean()

    def train_step(
        self, batch: Tensor, alpha: float | None
    ) -> Tuple[Tensor, Dict[str, float]]:
        assert isinstance(self.model, AggrLNP | BCALNP)

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

        y_dist_context, z_dist_context, z_context = self.model(context, mask, x_context)
        # (batch_size, num_subtasks, data_size, y_dim)

        loss = self.lnp_loss_context(
            y_dist_context, y_context, mask, z_dist_context, z_context
        )

        with torch.no_grad():
            y_dist_data, _, _ = self.model(
                context.clone().detach(), mask.clone().detach(), x_data.clone().detach()
            )
            # (batch_size, num_subtasks, data_size, y_dim)

        lmpl = compute_lmpl(y_dist_data, y_data)
        mse = compute_mse(y_dist_data, y_data)

        return loss, {"lmpl": lmpl.item(), "mse": mse.item()}
