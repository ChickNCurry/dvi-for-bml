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
    eval_kde_on_grid,
)


class AlternatingBMLTrainer:
    def __init__(
        self,
        device: torch.device,
        cdvi: CDVI,
        train_decoder_loader: DataLoader[Any],
        train_cdvi_loader: DataLoader[Any],
        val_loader: DataLoader[Any],
        optimizer: Optimizer,
        wandb_logging: bool,
        num_subtasks: int = 32,
    ) -> None:
        self.device = device
        self.cdvi = cdvi
        self.train_decoder_loader = train_decoder_loader
        self.train_cdvi_loader = train_cdvi_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.wandb_logging = wandb_logging
        self.num_subtasks = num_subtasks

    def train(
        self,
        num_epochs: int,
        max_clip_norm: float | None,
        alpha: float | None,
        validate: bool = False,
    ) -> None:

        # torch.autograd.set_detect_anomaly(True)

        for epoch in range(num_epochs):

            flip = epoch % 2 == 0

            if validate:

                self.cdvi.eval()
                with torch.inference_mode(False):

                    loop = tqdm(self.val_loader, total=len(self.val_loader))

                    for batch in loop:

                        metrics = self.val_step(batch)

                        loop.set_postfix(
                            ordered_dict=OrderedDict(
                                [
                                    ("epoch", epoch),
                                    *[(k, v) for k, v in metrics.items()],
                                ]
                            )
                        )

                        if self.wandb_logging:
                            wandb.log(
                                {
                                    **{f"val/{k}": v for k, v in metrics.items()},
                                }
                            )

            self.cdvi.train()
            with torch.inference_mode(False):

                loop = tqdm(
                    self.train_decoder_loader if flip else self.train_cdvi_loader,
                    total=len(
                        self.train_decoder_loader if flip else self.train_cdvi_loader
                    ),
                )

                for batch in loop:

                    self.optimizer.zero_grad()

                    loss, metrics = (
                        self.train_step_decoder(batch, alpha)
                        if flip
                        else self.train_step_cdvi(batch, alpha)
                    )

                    loss.backward()  # type: ignore

                    if max_clip_norm is not None:
                        torch.nn.utils.clip_grad_norm_(
                            self.cdvi.parameters(), max_clip_norm
                        )

                    self.optimizer.step()

                    loop.set_postfix(
                        ordered_dict=OrderedDict(
                            [
                                ("epoch", epoch),
                                (
                                    "loss_decoder" if flip else "loss_cdvi",
                                    loss.item(),
                                ),
                                *[(k, v) for k, v in metrics.items()],
                            ]
                        )
                    )

                    if self.wandb_logging:
                        wandb.log(
                            {
                                (
                                    "train/loss_decoder" if flip else "train/loss_cdvi"
                                ): loss.item(),
                                **(
                                    {f"train/{k}": v for k, v in metrics.items()}
                                    if metrics is not None
                                    else {}
                                ),
                            }
                        )

    def setup_step(
        self, batch: Tensor, alpha: float | None
    ) -> Tuple[Tensor, Tensor, Dict[str, float]]:
        assert self.cdvi.decoder is not None

        x_data, y_data = batch
        x_data = x_data.to(self.device)
        y_data = y_data.to(self.device)
        # (batch_size, context_size, x_dim)
        # (batch_size, context_size, y_dim)

        x_data = x_data.unsqueeze(1).expand(-1, self.num_subtasks, -1, -1)
        y_data = y_data.unsqueeze(1).expand(-1, self.num_subtasks, -1, -1)
        # (batch_size, num_subtasks, context_size, x_dim)
        # (batch_size, num_subtasks, context_size, y_dim)

        data = torch.cat([x_data, y_data], dim=-1)
        # (batch_size, num_subtasks, context_size, x_dim + y_dim)

        if alpha is None:
            rand_context_sizes = torch.randint(
                low=1,
                high=data.shape[2] + 1,
                size=(data.shape[0], data.shape[1], 1),
                device=self.device,
            )  # (batch_size, num_subtasks, 1)
        else:
            rand_context_sizes = torch.tensor(
                np.ceil(
                    np.random.beta(a=alpha, b=2, size=(data.shape[0], data.shape[1], 1))
                    * data.shape[2]
                ),
                device=self.device,
            )  # (batch_size, num_subtasks, 1)

        pos_indices = (
            torch.arange(data.shape[2], device=self.device)
            .unsqueeze(0)
            .unsqueeze(0)
            .expand(data.shape[0], data.shape[1], -1)
        )  # (batch_size, num_subtasks, context_size)

        mask = (pos_indices < rand_context_sizes).float()
        # (batch_size, num_subtasks, context_size)

        context = data * mask.unsqueeze(-1).expand(-1, -1, -1, data.shape[-1])
        # (batch_size, num_subtasks, context_size, x_dim + y_dim)

        x_context = context[:, :, :, 0 : x_data.shape[-1]]
        y_context = context[:, :, :, x_data.shape[-1] : data.shape[-1]]
        # (batch_size, num_subtasks, context_size, x_dim)
        # (batch_size, num_subtasks, context_size, y_dim)

        aggregated, non_aggregated = self.cdvi.encoder(context, mask)
        # (batch_size, num_subtasks, h_dim)

        p_z_T = LikelihoodTimesPrior(
            decoder=self.cdvi.decoder,
            x_target=x_context,
            y_target=y_context,
            mask=mask,
            context_embedding=(
                non_aggregated if self.cdvi.decoder.is_cross_attentive else aggregated
            ),
        )

        elbo, log_like, z_samples = self.cdvi.dvi_process.run_chain(
            p_z_T=p_z_T,
            context_embedding=(
                non_aggregated
                if self.cdvi.dvi_process.control.is_cross_attentive
                else aggregated
            ),
            mask=mask,
        )  # (num_steps, batch_size, num_subtasks, z_dim)

        lmpl: Tensor = torch.median(
            torch.logsumexp(p_z_T.log_like(z_samples[-1], x_data, y_data, None), dim=1)
            - np.log(data.shape[1])
        )

        mse: Tensor = torch.median(p_z_T.mse(z_samples[-1], x_data, y_data).mean(1))

        return elbo, log_like, {"lmpl": lmpl.item(), "mse": mse.item()}

    def train_step_decoder(
        self, batch: Tensor, alpha: float | None
    ) -> Tuple[Tensor, Dict[str, float]]:
        self.cdvi.freeze(only_decoder=False)

        _, log_like, metrics = self.setup_step(batch, alpha)

        loss = -log_like

        return loss, metrics

    def train_step_cdvi(
        self, batch: Tensor, alpha: float | None
    ) -> Tuple[Tensor, Dict[str, float]]:
        self.cdvi.freeze(only_decoder=True)

        elbo, _, metrics = self.setup_step(batch, alpha)

        loss = -elbo

        return loss, metrics

    def val_step(
        self,
        batch: Tensor,
        ranges: List[Tuple[float, float]] = [(-6, 6), (-6, 6)],
    ) -> Dict[str, float]:
        assert self.cdvi.decoder is not None

        x_data, y_data = batch
        x_data = x_data.to(self.device)
        y_data = y_data.to(self.device)
        # (batch_size, context_size, x_dim)
        # (batch_size, context_size, y_dim)

        x_data = x_data.unsqueeze(1).expand(-1, x_data.shape[0], -1, -1)
        y_data = y_data.unsqueeze(1).expand(-1, x_data.shape[0], -1, -1)
        # (batch_size, sample_size, context_size, x_dim)
        # (batch_size, sample_size, context_size, y_dim)

        data = torch.cat([x_data, y_data], dim=-1)
        # (batch_size, sample_size, context_size, x_dim + y_dim)

        context_sizes = torch.ones(
            size=(data.shape[0], data.shape[1], 1),
            device=self.device,
        )  # (batch_size, num_subtasks, 1)

        pos_indices = (
            torch.arange(data.shape[2], device=self.device)
            .unsqueeze(0)
            .unsqueeze(0)
            .expand(data.shape[0], data.shape[1], -1)
        )  # (batch_size, sample_size, context_size)

        mask = (pos_indices < context_sizes).float()
        # (batch_size, sample_size, context_size)

        context = data * mask.unsqueeze(-1).expand(-1, -1, -1, data.shape[-1])
        # (batch_size, sample_size, context_size, x_dim + y_dim)

        x_context = context[:, :, :, 0 : x_data.shape[-1]]
        y_context = context[:, :, :, x_data.shape[-1] : data.shape[-1]]
        # (batch_size, sample_size, context_size, x_dim)
        # (batch_size, sample_size, context_size, y_dim)

        aggr, non_aggr = self.cdvi.encoder(context, mask)
        # (batch_size, sample_size, h_dim)

        p_z_T = LikelihoodTimesPrior(
            decoder=self.cdvi.decoder,
            x_target=x_context,
            y_target=y_context,
            context_embedding=(
                non_aggr if self.cdvi.decoder.is_cross_attentive else aggr
            ),
            mask=mask,
        )

        _, _, z_samples = self.cdvi.dvi_process.run_chain(
            p_z_T=p_z_T,
            context_embedding=(
                non_aggr if self.cdvi.dvi_process.control.is_cross_attentive else aggr
            ),
            mask=mask,
        )  # (num_steps, batch_size, sample_size, z_dim)

        tp_samples = z_samples[-1].detach().cpu().numpy()
        # (batch_size, sample_size, z_dim)

        jsds = []
        bds = []

        grid = create_grid(ranges, int(np.sqrt(x_data.shape[0] * x_data.shape[1])))

        for i in range(tp_samples.shape[0]):
            target_vals = eval_dist_on_grid(
                grid, p_z_T, x_data.shape[0], x_data.shape[1], device=self.device
            )
            tp_vals = eval_kde_on_grid(grid, tp_samples[i])

            jsd = compute_jsd(target_vals, tp_vals)
            bd = compute_bd(target_vals, tp_vals)

            jsds.append(jsd)
            bds.append(bd)

        jsd = np.median(jsds)
        bd = np.median(bds)

        return {"jsd": jsd, "bd": bd}
