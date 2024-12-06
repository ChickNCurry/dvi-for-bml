from typing import Any, Dict, Tuple

import numpy as np
import torch
from torch import Tensor
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader

from src.components.cdvi import CDVI
from src.components.decoder import LikelihoodTimesPrior
from src.train.train import Trainer
from src.utils.grid import (
    compute_bd,
    compute_jsd,
    create_grid,
    eval_dist_on_grid,
    eval_kde_on_grid,
)


class BMLTrainer(Trainer):
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

        assert self.cdvi.decoder is not None

    def val_step(
        self,
        batch: Tensor,
    ) -> Dict[str, float]:
        raise NotImplementedError


class NoisyBMLTrainer(BMLTrainer):
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
        assert self.cdvi.decoder is not None

        x_data, y_data = batch
        x_data, y_data = x_data.to(self.device), y_data.to(self.device)
        # (batch_size, context_size, x_dim), (batch_size, context_size, y_dim)

        rand_sub_context_size: int = np.random.randint(1, x_data.shape[1] + 1)

        x_context = x_data[:, 0:rand_sub_context_size, :]
        y_context = y_data[:, 0:rand_sub_context_size, :]

        context = torch.cat([x_context, y_context], dim=-1)
        # (batch_size, context_size, x_dim + y_dim)

        context_embedding = self.cdvi.encoder(context)
        # (batch_size, h_dim)

        p_z_T = LikelihoodTimesPrior(
            decoder=self.cdvi.decoder,
            x_target=x_context,
            y_target=y_context,
            mask=None,
            context_embedding=context_embedding,
        )

        elbo, _, _ = self.cdvi.dvi_process.run_chain(p_z_T, context_embedding, None)

        loss = -elbo

        return loss, {}


class BetterBMLTrainer(BMLTrainer):
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
        assert self.cdvi.decoder is not None

        x_data, y_data = batch
        x_data, y_data = x_data.to(self.device), y_data.to(self.device)
        # (batch_size, context_size, x_dim), (batch_size, context_size, y_dim)

        data = torch.cat([x_data, y_data], dim=-1)
        # (batch_size, context_size, x_dim + y_dim)

        if alpha is None:
            rand_context_sizes = torch.randint(
                1, data.shape[1] + 1, (data.shape[0],), device=self.device
            ).unsqueeze(-1)
        else:
            rand_context_sizes = torch.tensor(
                np.ceil(
                    np.random.beta(a=alpha, b=2, size=(data.shape[0], 1))
                    * data.shape[1]
                ),
                device=self.device,
            )
        # (batch_size, 1)

        position_indices = torch.arange(data.shape[1], device=self.device).expand(
            data.shape[0], -1
        )  # (batch_size, context_size)

        mask = (position_indices < rand_context_sizes).float()
        # (batch_size, context_size)

        context = data * mask.unsqueeze(-1).expand(-1, -1, data.shape[2])
        # (batch_size, context_size, x_dim + y_dim)

        x_context = context[:, :, 0 : x_data.shape[2]]
        # (batch_size, context_size, x_dim)

        y_context = context[:, :, x_data.shape[2] : data.shape[2]]
        # (batch_size, context_size, y_dim)

        aggregated, non_aggregated = self.cdvi.encoder(context, mask)
        # (batch_size, h_dim)

        p_z_T = LikelihoodTimesPrior(
            decoder=self.cdvi.decoder,
            x_target=x_context,
            y_target=y_context,
            mask=mask,
            context_embedding=(
                non_aggregated if self.cdvi.decoder.is_cross_attentive else aggregated
            ),
        )

        elbo, _, z_samples = self.cdvi.dvi_process.run_chain(
            p_z_T,
            (
                non_aggregated
                if self.cdvi.dvi_process.control.is_cross_attentive
                else aggregated
            ),
            mask,
        )  # (1), (num_steps, batch_size, z_dim)

        loss = -elbo

        lmpl = -np.log(data.shape[0]) + (
            torch.logsumexp(
                p_z_T.log_like(
                    z=z_samples[-1],
                    x_target=x_data,
                    y_target=y_data,
                    mask=None,
                ),
                dim=0,
            )
        )

        mse = p_z_T.mse(
            z=z_samples[-1],
            x_target=x_data,
            y_target=y_data,
            mask=None,
        )

        return loss, {"lmpl": lmpl.item(), "mse": mse.item()}
