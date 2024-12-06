from collections import OrderedDict
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import wandb
from torch import Tensor
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.components.cdvi import CDVI
from src.components.decoder import LikelihoodTimesPrior
from src.utils.grid import (
    compute_bd,
    compute_jsd,
    create_grid,
    eval_dist_on_grid,
    eval_hist_on_grid,
    eval_kde_on_grid,
)


class AlternatingBMLTrainer:
    def __init__(
        self,
        device: torch.device,
        cdvi: CDVI,
        train_loader: DataLoader[Any],
        val_loader: DataLoader[Any],
        optimizer: Optimizer,
        wandb_logging: bool,
    ) -> None:
        self.device = device
        self.cdvi = cdvi
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.wandb_logging = wandb_logging

    def train(
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

                loop = tqdm(self.train_loader, total=len(self.train_loader))

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
                        ordered_dict=OrderedDict(
                            [
                                ("epoch", epoch),
                                ("loss_cdvi", loss_cdvi.item()),
                                ("loss_decoder", loss_decoder.item()),
                            ]
                        )
                    )

                    if self.wandb_logging:
                        wandb.log(
                            {
                                "train/loss_cdvi": loss_cdvi.item(),
                                "train/loss_decoder": loss_decoder.item(),
                            }
                        )

            if not validate or epoch % 10 != 0:
                continue

            self.cdvi.eval()
            with torch.inference_mode(False):

                loop = tqdm(self.val_loader, total=len(self.val_loader))

                for batch in loop:

                    metrics = self.val_step(batch)

                    # loop.set_postfix(**metrics)

                    if self.wandb_logging:
                        wandb.log(
                            {
                                **{f"val/{k}": v for k, v in metrics.items()},
                            }
                        )

    def setup_step(self, batch: Tensor, alpha: float | None) -> Tuple[Tensor, Tensor]:
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

    def val_step(
        self,
        batch: Tensor,
        sample_size: int = 256,
        context_sizes: List[int] = list(range(1, 4)),
        intervals: List[Tuple[float, float]] = [(-6, 6), (-6, 6)],
        num: int = 16,
    ) -> Dict[str, float]:
        assert self.cdvi.decoder is not None

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
                    p_z_T.log_like(z_samples[-1], x_data, y_data, None), dim=0
                )

                mse = p_z_T.val_mse(z_samples[-1])

                target_vals = eval_dist_on_grid(grid, p_z_T, device=self.device)
                dvi_vals = eval_kde_on_grid(grid, z_samples[-1].detach().cpu().numpy())

                jsd = compute_jsd(target_vals, dvi_vals)
                bd = compute_bd(target_vals, dvi_vals)

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


class TestAlternatingBMLTrainer(AlternatingBMLTrainer):
    def __init__(
        self,
        device: torch.device,
        cdvi: CDVI,
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
        assert self.cdvi.decoder is not None

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
        assert self.cdvi.decoder is not None

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
