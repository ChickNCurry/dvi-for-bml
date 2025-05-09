from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from torch import Generator, Tensor
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader, Dataset

from dviforbml.architectures.dvinp import DVINP
from dviforbml.components.decoder.decoder_times_prior import DecoderTimesPrior
from dviforbml.evaluation.predictive.pred_metrics import compute_lmpl, compute_mse
from dviforbml.evaluation.taskposterior.grid import (
    create_grid,
    eval_dist_on_grid,
    eval_hist_on_grid,
)
from dviforbml.evaluation.taskposterior.tp_metrics import compute_bd, compute_jsd
from dviforbml.training.abstract_trainer import AbstractTrainer


class DVINPTrainer(AbstractTrainer, ABC):
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
            target_dist, r_context, mask, s_context, None
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
            target_dist, r_context, mask, s_context, None
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


class DVINPTrainerContext(DVINPTrainer):
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

        elbo, zs = self.model.cdvi.run_both_processes(
            target, r_context, mask, s_context
        )  # (1), (num_steps, batch_size, num_subtasks, z_dim)

        with torch.no_grad():
            y_dist_data = self.model.decoder(
                zs[-1].clone().detach(), x_data.clone().detach()
            )  # (batch_size, num_subtasks, z_dim)

        loss = -elbo

        lmpl = compute_lmpl(y_dist_data, y_data)
        mse = compute_mse(y_dist_data, y_data)

        return loss, {"lmpl": lmpl.item(), "mse": mse.item()}


class DVINPTrainerContextNoise(DVINPTrainer):
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

        data, x_data, y_data = self.get_data_subtasks(batch)
        # (batch_size, num_subtasks, data_size, x_dim + y_dim)
        # (batch_size, num_subtasks, data_size, x_dim)
        # (batch_size, num_subtasks, data_size, y_dim)

        rand_context_size: int = np.random.randint(1, x_data.shape[1] + 1)
        x_context = x_data[:, :, 0:rand_context_size, :]
        y_context = y_data[:, :, 0:rand_context_size, :]
        context = torch.cat((x_context, y_context), dim=-1)
        # (batch_size, num_subtasks, data_size, x_dim + y_dim)
        # (batch_size, num_subtasks, data_size, x_dim)
        # (batch_size, num_subtasks, data_size, y_dim)

        r_context, s_context = self.model.encoder(context, None)
        # (batch_size, num_subtasks, h_dim)
        # (batch_size, num_subtasks)

        target = DecoderTimesPrior(
            decoder=self.model.decoder,
            x=x_context,
            y=y_context,
            mask=None,
        )

        elbo, zs = self.model.cdvi.run_both_processes(
            target, r_context, None, s_context
        )  # (1), (num_steps, batch_size, num_subtasks, z_dim)

        with torch.no_grad():
            y_dist_data = self.model.decoder(
                zs[-1].clone().detach(), x_data.clone().detach()
            )  # (batch_size, num_subtasks, z_dim)

        loss = -elbo

        lmpl = compute_lmpl(y_dist_data, y_data)
        mse = compute_mse(y_dist_data, y_data)

        return loss, {"lmpl": lmpl.item(), "mse": mse.item()}


class DVINPTrainerData(DVINPTrainer):
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

        r_context, s_context = self.model.encoder(context, mask)
        r_data, s_data = self.model.encoder(data, None)
        # (batch_size, num_subtasks, h_dim)
        # (batch_size, num_subtasks)

        target = DecoderTimesPrior(
            decoder=self.model.decoder,
            x=x_data,
            y=y_data,
            mask=None,
        )

        log_prob_fw, zs = self.model.cdvi.run_forward_process(
            target, r_context, mask, None, s_context
        )  # (1), (num_entries, batch_size, num_subtasks, z_dim)

        assert zs is not None

        log_prob_bw = self.model.cdvi.run_backward_process(
            target, r_data, None, s_data, zs
        )  # (num_entries, batch_size, num_subtasks, z_dim)

        with torch.no_grad():
            y_dist_data = self.model.decoder(
                zs[-1].clone().detach(), x_data.clone().detach()
            )  # (batch_size, num_subtasks, z_dim)

        loss = log_prob_fw - log_prob_bw

        lmpl = compute_lmpl(y_dist_data, y_data)
        mse = compute_mse(y_dist_data, y_data)

        return loss, {"lmpl": lmpl.item(), "mse": mse.item()}


class DVINPTrainerForward(DVINPTrainer):
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

    def train_step_old(
        self, batch: Tensor, alpha: float | None
    ) -> Tuple[Tensor, Dict[str, float]]:
        assert isinstance(self.model, DVINP)

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

        r_context, s_context = self.model.encoder(context, mask)
        r_data, s_data = self.model.encoder(data, None)
        # (batch_size, num_subtasks, h_dim)
        # (batch_size, num_subtasks)

        target_dist = DecoderTimesPrior(
            decoder=self.model.decoder,
            x=x_target,
            y=y_target,
            mask=1 - mask,
        )

        log_prob_fw_data, zs = self.model.cdvi.run_forward_process(
            target_dist, r_data, None, s_data, None
        )

        assert zs is not None

        log_prob_fw_context, _ = self.model.cdvi.run_forward_process(
            target_dist, r_context, mask, s_context, zs
        )  # (1), (num_steps, batch_size, num_subtasks, z_dim)

        y_dist_target = self.model.decoder(zs[-1], x_target)
        # (batch_size, num_subtasks, z_dim)

        with torch.no_grad():
            y_dist_data = self.model.decoder(
                zs[-1].clone().detach(), x_data.clone().detach()
            )  # (batch_size, num_subtasks, z_dim)

        log_like: Tensor = y_dist_target.log_prob(y_target)
        mask = (1 - mask).unsqueeze(-1).expand(-1, -1, -1, log_like.shape[-1])
        log_like = (log_like * mask).sum(-1).mean()

        loss = -(log_like + log_prob_fw_context - log_prob_fw_data)

        lmpl = compute_lmpl(y_dist_data, y_data)
        mse = compute_mse(y_dist_data, y_data)

        return loss, {"lmpl": lmpl.item(), "mse": mse.item()}

    def train_step(
        self, batch: Tensor, alpha: float | None
    ) -> Tuple[Tensor, Dict[str, float]]:
        assert isinstance(self.model, DVINP)

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

        r_context, s_context = self.model.encoder(context, mask)
        r_data, s_data = self.model.encoder(data, None)
        # (batch_size, num_subtasks, h_dim)
        # (batch_size, num_subtasks)

        target_dist = DecoderTimesPrior(
            decoder=self.model.decoder,
            x=x_target,
            y=y_target,
            mask=1 - mask,
        )

        log_prob, zs = self.model.cdvi.run_2_forward_processes(
            target_dist, r_data, r_context, mask, s_data, s_context
        )  # (1), (num_steps, batch_size, num_subtasks, z_dim)

        y_dist_target = self.model.decoder(zs[-1], x_target)
        # (batch_size, num_subtasks, z_dim)

        with torch.no_grad():
            y_dist_data = self.model.decoder(
                zs[-1].clone().detach(), x_data.clone().detach()
            )  # (batch_size, num_subtasks, z_dim)

        log_like: Tensor = y_dist_target.log_prob(y_target)
        mask = (1 - mask).unsqueeze(-1).expand(-1, -1, -1, log_like.shape[-1])
        log_like = (log_like * mask).sum(-1).mean()

        loss = -(log_like - log_prob)

        lmpl = compute_lmpl(y_dist_data, y_data)
        mse = compute_mse(y_dist_data, y_data)

        return loss, {"lmpl": lmpl.item(), "mse": mse.item()}


class DVINPTrainerForwardAndContext(DVINPTrainer):
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

        data, x_data, y_data = self.get_data_subtasks(batch)
        # (batch_size, num_subtasks, data_size, x_dim + y_dim)
        # (batch_size, num_subtasks, data_size, x_dim)
        # (batch_size, num_subtasks, data_size, y_dim)

        mask = self.get_mask(alpha, data)
        # (batch_size, num_subtasks, data_size)

        context, x_context, y_context = self.get_context(data, x_data, mask)
        _, x_target, y_target = self.get_target(data, x_data, mask)
        # (batch_size, num_subtasks, data_size, x_dim + y_dim)
        # (batch_size, num_subtasks, data_size, x_dim)
        # (batch_size, num_subtasks, data_size, y_dim)

        r_context, s_context = self.model.encoder(context, mask)
        r_data, s_data = self.model.encoder(data, None)
        # (batch_size, num_subtasks, h_dim)
        # (batch_size, num_subtasks)

        target_dist = DecoderTimesPrior(
            decoder=self.model.decoder,
            x=x_context,
            y=y_context,
            mask=mask,
        )

        elbo, _ = self.model.cdvi.run_both_processes(
            target_dist, r_context, mask, s_context
        )  # (1), (num_steps, batch_size, num_subtasks, z_dim)

        loss_context = -elbo

        target_dist = DecoderTimesPrior(
            decoder=self.model.decoder,
            x=x_target,
            y=y_target,
            mask=1 - mask,
        )

        log_prob_fw_data, zs = self.model.cdvi.run_forward_process(
            target_dist, r_data, None, s_data, None
        )

        assert zs is not None

        log_prob_fw_context, _ = self.model.cdvi.run_forward_process(
            target_dist, r_context, mask, s_context, zs
        )
        # (1), (num_steps, batch_size, num_subtasks, z_dim)

        y_dist_target = self.model.decoder(zs[-1], x_target)
        # (batch_size, num_subtasks, z_dim)

        with torch.no_grad():
            y_dist_data = self.model.decoder(
                zs[-1].clone().detach(), x_data.clone().detach()
            )
            # (batch_size, num_subtasks, z_dim)

        log_like: Tensor = y_dist_target.log_prob(y_target)
        mask = (1 - mask).unsqueeze(-1).expand(-1, -1, -1, log_like.shape[-1])
        log_like = (log_like * mask).sum(-1).mean()

        loss_forward = -(log_like + log_prob_fw_context - log_prob_fw_data)

        loss = loss_context + loss_forward

        lmpl = compute_lmpl(y_dist_data, y_data)
        mse = compute_mse(y_dist_data, y_data)

        return loss, {"lmpl": lmpl.item(), "mse": mse.item()}


# class NoisyDVINPTrainer(BaseTrainer):
#     def __init__(
#         self,
#         device: torch.device,
#         dvinp: DVINP,
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
#             device,
#             dvinp,
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
#         assert self.model.decoder is not None

#         x_data, y_data = batch
#         x_data = x_data.to(self.device)
#         y_data = y_data.to(self.device)
#         # (batch_size, context_size, x_dim)
#         # (batch_size, context_size, y_dim)

#         x_data = x_data.unsqueeze(1)
#         y_data = y_data.unsqueeze(1)
#         # (batch_size, 1, context_size, x_dim)
#         # (batch_size, 1, context_size, y_dim)

#         rand_sub_context_size: int = np.random.randint(1, x_data.shape[1] + 1)
#         x_context = x_data[:, :, 0:rand_sub_context_size, :]
#         y_context = y_data[:, :, 0:rand_sub_context_size, :]
#         # (batch_size, 1, context_size, x_dim)
#         # (batch_size, 1, context_size, y_dim)

#         context = torch.cat([x_context, y_context], dim=-1)
#         # (batch_size, 1, context_size, x_dim + y_dim)

#         r = self.model.encoder(context)
#         # (batch_size, 1, h_dim)

#         target = DecoderTimesPrior(
#             decoder=self.model.decoder,
#             x=x_context,
#             y=y_context,
#             mask=None,
#         )

#         elbo, _, _ = self.model.cdvi.run_chain(target, r, None)

#         loss = -elbo

#         return loss, {}
