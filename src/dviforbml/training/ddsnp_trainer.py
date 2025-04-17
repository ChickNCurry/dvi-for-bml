from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from torch import Generator, Tensor
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader, Dataset

from dviforbml.architectures.dvinp import DVINP
from dviforbml.components.cdvi.dds import DDS
from dviforbml.components.decoder.decoder_times_prior import DecoderTimesPrior
from dviforbml.evaluation.predictive.pred_metrics import compute_lmpl, compute_mse
from dviforbml.evaluation.taskposterior.grid import (
    create_grid,
    eval_dist_on_grid,
    eval_hist_on_grid,
)
from dviforbml.evaluation.taskposterior.tp_metrics import compute_bd, compute_jsd
from dviforbml.training.abstract_trainer import AbstractTrainer


class DDSNPTrainer(AbstractTrainer, ABC):
    def __init__(
        self,
        model: DVINP,
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

    def val_step(
        self,
        batch: Tensor,
        ranges: List[Tuple[float, float]] = [(-6, 6), (-6, 6)],
    ) -> Dict[str, float]:
        pred_metrics = self.val_pred_step(batch)
        tp_metrics = self.val_tp_step(batch, ranges)
        return {**pred_metrics, **tp_metrics}

    def val_pred_step(
        self,
        batch: Tensor,
    ) -> Dict[str, float]:
        assert isinstance(self.model, DVINP)

        data, x_data, y_data = self.get_data_subtasks(batch)
        # (batch_size, num_subtasks, data_size, x_dim + y_dim)
        # (batch_size, num_subtasks, data_size, x_dim)
        # (batch_size, num_subtasks, data_size, y_dim)

        mask = self.get_mask(None, data)
        # (batch_size, num_subtasks, data_size)

        context, x_context, y_context = self.get_context(data, x_data, mask)
        # (batch_size, num_subtasks, data_size, x_dim + y_dim)
        # (batch_size, num_subtasks, data_size, x_dim)
        # (batch_size, num_subtasks, data_size, y_dim)

        target_dist = DecoderTimesPrior(
            decoder=self.model.decoder,
            x=x_context,
            y=y_context,
            mask=mask,
        )

        r_context, s_context = self.model.encoder(context, mask)
        # (batch_size, num_samples, h_dim) or (batch_size, num_samples, data_size, h_dim)
        # (batch_size, num_samples, data_size)

        _, zs = self.model.cdvi.run_forward_process(
            target_dist, r_context, mask, None, s_context
        )  # (num_steps, batch_size, num_samples, z_dim)

        y_dist_data = self.model.decoder(
            zs[-1].clone().detach(), x_data.clone().detach()
        )  # (batch_size, num_subtasks, z_dim)

        lmpl = compute_lmpl(y_dist_data, y_data)
        mse = compute_mse(y_dist_data, y_data)

        return {"lmpl": lmpl.item(), "mse": mse.item()}

    def val_tp_step(
        self,
        batch: Tensor,
        ranges: List[Tuple[float, float]] = [(-6, 6), (-6, 6)],
    ) -> Dict[str, float]:
        assert isinstance(self.model, DVINP)

        data, x_data, _ = self.get_data_samples(batch)
        # (batch_size, num_samples, data_size, x_dim + y_dim)
        # (batch_size, num_samples, data_size, x_dim)
        # (batch_size, num_samples, data_size, y_dim)

        mask = self.get_mask_1(data)
        # (batch_size, num_samples, data_size)

        context, x_context, y_context = self.get_context(data, x_data, mask)
        # (batch_size, num_samples, data_size, x_dim + y_dim)
        # (batch_size, num_samples, data_size, x_dim)
        # (batch_size, num_samples, data_size, y_dim)

        target_dist = DecoderTimesPrior(
            decoder=self.model.decoder,
            x=x_context,
            y=y_context,
            mask=mask,
        )

        r_context, s_context = self.model.encoder(context, mask)
        # (batch_size, num_samples, h_dim) or (batch_size, num_samples, data_size, h_dim)
        # (batch_size, num_samples, data_size)

        _, zs = self.model.cdvi.run_forward_process(
            target_dist, r_context, mask, None, s_context
        )
        # (num_steps, batch_size, num_samples, z_dim)

        tp_samples = zs[-1].detach().cpu().numpy()
        # (batch_size, num_samples, z_dim)

        sqrt = np.sqrt(self.num_samples)
        assert sqrt.is_integer()
        num_cells = int(sqrt)
        grid = create_grid(ranges, num_cells)

        target_log_probs = eval_dist_on_grid(
            grid, target_dist, self.device, batch_size=data.shape[0]
        )

        jsds = []
        bds = []

        for i in range(tp_samples.shape[0]):
            tp_log_probs = eval_hist_on_grid(tp_samples[i], ranges, num_cells)

            jsd = compute_jsd(target_log_probs[i], tp_log_probs)
            bd = compute_bd(target_log_probs[i], tp_log_probs)

            jsds.append(jsd)
            bds.append(bd)

        jsd = np.median(jsds)
        bd = np.median(bds)

        return {"jsd": jsd, "bd": bd}


class DDSNPTrainerContext(DDSNPTrainer):
    def __init__(
        self,
        model: DVINP,
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
        assert isinstance(self.model, DVINP)
        assert isinstance(self.model.cdvi, DDS)

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

        r_context, s_context = self.model.encoder(context, mask)
        # (batch_size, num_subtasks, h_dim)
        # (batch_size, num_subtasks)

        target = DecoderTimesPrior(
            decoder=self.model.decoder,
            x=x_context,
            y=y_context,
            mask=mask,
        )

        loss, z = self.model.cdvi.compute_loss(target, r_context, mask, s_context)
        # (batch_size, num_subtasks, z_dim)

        # y_dist_context = self.model.decoder(z[:, :, -1, :], x_context)
        y_dist_data = self.model.decoder(z[:, :, -1, :], x_data)

        loss = loss  # - compute_lmpl(y_dist_context, y_context)

        lmpl = compute_lmpl(y_dist_data, y_data)
        mse = compute_mse(y_dist_data, y_data)

        return loss, {"lmpl": lmpl.item(), "mse": mse.item()}
