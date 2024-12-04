from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from src.components.cdvi import ContextualDVI
from src.components.decoder import LikelihoodTimesPrior
from src.utils.grid import (
    compute_bd,
    compute_jsd,
    create_grid,
    eval_dist_on_grid,
    eval_kde_on_grid,
    normalize_vals_on_grid,
)
from src.utils.train_val import AlternatingTrainer, Trainer


class BMLTrainer(Trainer):
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


class NoisyBMLTrainer(BMLTrainer):
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

    def train_step(self, batch: Tensor, alpha: float | None) -> Tensor:

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

        elbo, _ = self.cdvi.dvi_process.run_chain(p_z_T, context_embedding, None)

        loss = -elbo

        return loss


class BetterBMLTrainer(BMLTrainer):
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

    def train_step(self, batch: Tensor, alpha: float | None) -> Tensor:

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

        elbo, _ = self.cdvi.dvi_process.run_chain(
            p_z_T,
            (
                non_aggregated
                if self.cdvi.dvi_process.control.is_cross_attentive
                else aggregated
            ),
            mask,
        )  # (1), (num_steps, batch_size, z_dim)

        loss = -elbo

        return loss


class AlternatingBMLTrainer(AlternatingTrainer):
    def __init__(
        self,
        device: torch.device,
        cdvi: ContextualDVI,
        train_loader: DataLoader[Any],
        val_loader: DataLoader[Any],
        optimizer: Optimizer,
        wandb_logging: bool,
    ) -> None:
        super().__init__(
            device,
            cdvi,
            train_loader,
            val_loader,
            optimizer,
            wandb_logging,
        )

        assert self.cdvi.decoder is not None

    def setup_step(self, batch: Tensor, alpha: float | None) -> Tuple[Tensor, Tensor]:

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

        elbo, log_like, _ = self.cdvi.dvi_process.run_chain(
            p_z_T,
            (
                non_aggregated
                if self.cdvi.dvi_process.control.is_cross_attentive
                else aggregated
            ),
            mask,
        )  # (1), (num_steps, batch_size, z_dim)

        return elbo, log_like

    def train_step_decoder(self, batch: Tensor, alpha: float | None) -> Tensor:
        self.cdvi.freeze(only_decoder=False)

        _, log_like = self.setup_step(batch, alpha)

        loss = -log_like

        return loss

    def train_step_cdvi(self, batch: Tensor, alpha: float | None) -> Tensor:
        self.cdvi.freeze(only_decoder=True)

        elbo, _ = self.setup_step(batch, alpha)

        loss = -elbo

        return loss


class TestAlternatingBMLTrainer(AlternatingTrainer):
    def __init__(
        self,
        device: torch.device,
        cdvi: ContextualDVI,
        train_loader: DataLoader[Any],
        val_loader: DataLoader[Any],
        optimizer: Optimizer,
        wandb_logging: bool,
    ) -> None:
        super().__init__(
            device,
            cdvi,
            train_loader,
            val_loader,
            optimizer,
            wandb_logging,
        )

        assert self.cdvi.decoder is not None

    def train_step_decoder(self, batch: Tensor, alpha: float | None) -> Tensor:
        self.cdvi.freeze(only_decoder=False)

        x_data, y_data = batch
        x_data, y_data = x_data.to(self.device), y_data.to(self.device)
        # (batch_size, context_size, x_dim), (batch_size, context_size, y_dim)

        data = torch.cat([x_data, y_data], dim=-1)
        # (batch_size, context_size, x_dim + y_dim)

        aggregated, non_aggregated = self.cdvi.encoder(data, None)
        # (batch_size, h_dim)

        p_z_T = LikelihoodTimesPrior(
            decoder=self.cdvi.decoder,
            x_target=x_data,
            y_target=y_data,
            mask=None,
            context_embedding=(
                non_aggregated if self.cdvi.decoder.is_cross_attentive else aggregated
            ),
        )

        _, log_like, _ = self.cdvi.dvi_process.run_chain(
            p_z_T,
            (
                non_aggregated
                if self.cdvi.dvi_process.control.is_cross_attentive
                else aggregated
            ),
            None,
        )  # (1), (num_steps, batch_size, z_dim)

        loss = -log_like

        return loss

    def train_step_cdvi(self, batch: Tensor, alpha: float | None) -> Tensor:
        self.cdvi.freeze(only_decoder=True)

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

        elbo, _, _ = self.cdvi.dvi_process.run_chain(
            p_z_T,
            (
                non_aggregated
                if self.cdvi.dvi_process.control.is_cross_attentive
                else aggregated
            ),
            mask,
        )  # (1), (num_steps, batch_size, z_dim)

        loss = -elbo

        return loss
