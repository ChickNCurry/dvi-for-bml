from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import wandb
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.components.cdvi import ContextualDVI
from src.components.decoder import LikelihoodTimesPrior
from src.utils.eval import (
    compute_bd,
    compute_jsd,
    create_grid,
    eval_dist_on_grid,
    eval_kde_on_grid,
    normalize_vals_on_grid,
)


class Trainer(ABC):
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

            if not validate or epoch % 10 != 0:
                continue

            self.cdvi.eval()
            with torch.inference_mode(False):

                loop = tqdm(self.val_dataloader, total=len(self.val_dataloader))

                for batch in loop:

                    metrics = self.val_step(batch)

                    # loop.set_postfix(**metrics)

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


class AlternatingTrainer(ABC):
    def __init__(
        self,
        device: torch.device,
        cdvi: ContextualDVI,
        train_loader: DataLoader[Any],
        val_loader: DataLoader[Any],
        optimizer: Optimizer,
        wandb_logging: bool,
    ) -> None:
        self.device = device
        self.cdvi = cdvi
        self.train_dataloader = train_loader
        self.val_dataloader = val_loader
        self.optimizer = optimizer
        self.wandb_logging = wandb_logging

    def train_and_validate(
        self,
        num_epochs: int,
        max_clip_norm: float | None,
        alpha: float | None,
        validate: bool = False,
    ) -> None:

        # torch.autograd.set_detect_anomaly(True)

        for epoch in range(num_epochs):

            self.cdvi.train()
            with torch.inference_mode(False):

                loop = tqdm(self.train_dataloader, total=len(self.train_dataloader))

                for batch in loop:

                    self.optimizer.zero_grad()

                    loss_decoder = self.train_step_decoder(batch, alpha)
                    loss_decoder.backward()  # type: ignore

                    if max_clip_norm is not None:
                        torch.nn.utils.clip_grad_norm_(
                            self.cdvi.parameters(), max_clip_norm
                        )

                    self.optimizer.step()

                    self.optimizer.zero_grad()

                    loss_cdvi = self.train_step_cdvi(batch, alpha)
                    loss_cdvi.backward()  # type: ignore

                    if max_clip_norm is not None:
                        torch.nn.utils.clip_grad_norm_(
                            self.cdvi.parameters(), max_clip_norm
                        )

                    self.optimizer.step()

                    loop.set_postfix(
                        epoch=epoch,
                        loss_decoder=loss_decoder.item(),
                        loss_cdvi=loss_cdvi.item(),
                    )

                    if self.wandb_logging:
                        wandb.log(
                            {
                                "train/loss_decoder": loss_decoder.item(),
                                "train/loss_cdvi": loss_cdvi.item(),
                            }
                        )

            if not validate or epoch % 10 != 0:
                continue

            self.cdvi.eval()
            with torch.inference_mode(False):

                loop = tqdm(self.val_dataloader, total=len(self.val_dataloader))

                for batch in loop:

                    metrics = self.val_step(batch)

                    # loop.set_postfix(**metrics)

                    if self.wandb_logging:
                        wandb.log(
                            {
                                **{"val/" + k: v for k, v in metrics.items()},
                            }
                        )

    @abstractmethod
    def train_step_decoder(self, batch: Tensor, alpha: float | None) -> Tensor:
        pass

    @abstractmethod
    def train_step_cdvi(self, batch: Tensor, alpha: float | None) -> Tensor:
        pass

    def val_step(
        self,
        batch: Tensor,
        sample_size: int = 256,
        context_sizes: np.ndarray = np.arange(1, 4),
        intervals=[(-6, 6), (-6, 6)],
        num=16,
    ) -> Dict[str, Tensor]:

        x_data, y_data = batch
        x_data, y_data = x_data.to(self.device), y_data.to(self.device)
        # (batch_size, context_size, x_dim), (batch_size, context_size, y_dim)

        data = torch.cat([x_data, y_data], dim=-1)
        # (batch_size, context_size, x_dim + y_dim)

        grid = create_grid(intervals, num)

        lmpls = []
        mses = []
        jsds = []
        bds = []

        for context_size in context_sizes:

            lmpls_per_task = []
            mses_per_task = []
            jsds_per_task = []
            bds_per_task = []

            for i in range(0, data.shape[0]):

                data_per_task = data[i].unsqueeze(0).expand(sample_size, -1, -1)
                # (sample_size, context_size, x_dim + y_dim)

                context = data_per_task[:, 0:context_size, :]
                # (sample_size, context_size, x_dim + y_dim)

                x_data = data_per_task[:, :, 0 : x_data.shape[2]]
                # (sample_size, context_size, x_dim)

                y_data = data_per_task[:, :, x_data.shape[2] : data.shape[2]]
                # (sample_size, context_size, y_dim)

                aggregated, non_aggregated = self.cdvi.encoder(context, None)
                # (sample_size, h_dim)

                p_z_T = LikelihoodTimesPrior(
                    decoder=self.cdvi.decoder,
                    x_target=x_data,
                    y_target=y_data,
                    context_embedding=(
                        non_aggregated
                        if self.cdvi.decoder.is_cross_attentive
                        else aggregated
                    ),
                    mask=None,
                )

                _, _, z_samples = self.cdvi.dvi_process.run_chain(
                    p_z_T=p_z_T,
                    context_embedding=(
                        non_aggregated
                        if self.cdvi.dvi_process.control.is_cross_attentive
                        else aggregated
                    ),
                    mask=None,
                )
                # (1), (num_steps, sample_size, z_dim)

                lmpl = -np.log(sample_size) + torch.logsumexp(
                    p_z_T.log_likelihood(z_samples[-1]), dim=0
                )

                mse = p_z_T.val_mse(z_samples[-1])

                target_vals = eval_dist_on_grid(grid, p_z_T, device=self.device)
                dvi_vals = eval_kde_on_grid(grid, z_samples[-1].detach().cpu().numpy())

                target_vals = normalize_vals_on_grid(target_vals, intervals, num)
                dvi_vals = normalize_vals_on_grid(dvi_vals, intervals, num)

                jsd = compute_jsd(target_vals, dvi_vals, intervals, num)
                bd = compute_bd(target_vals, dvi_vals, intervals, num)

                lmpls_per_task.append(lmpl.item())
                mses_per_task.append(mse.item())
                jsds_per_task.append(jsd.item())
                bds_per_task.append(bd.item())

            lmpls.append(np.median(lmpls_per_task).item())
            mses.append(np.median(mses_per_task).item())
            jsds.append(np.median(jsds_per_task).item())
            bds.append(np.median(bds_per_task).item())

        lmpls_dict = {f"lmpl_{i + 1}": lmpls[i] for i in range(len(lmpls))}
        mses_dict = {f"mse_{i + 1}": mses[i] for i in range(len(mses))}
        jsds_dict = {f"jsd_{i + 1}": jsds[i] for i in range(len(jsds))}
        bds_dict = {f"bd_{i + 1}": bds[i] for i in range(len(bds))}

        return {**lmpls_dict, **mses_dict, **jsds_dict, **bds_dict}
