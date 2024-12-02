from abc import ABC, abstractmethod
from typing import Any, List

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


class Trainer(ABC):
    def __init__(
        self,
        device: torch.device,
        cdvi: ContextualDVI,
        dataloader: DataLoader[Any],
        optimizer: Optimizer,
        scheduler: ReduceLROnPlateau | None,
        wandb_logging: bool,
    ) -> None:
        self.device = device
        self.cdvi = cdvi
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.wandb_logging = wandb_logging

    def train(
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

                loop = tqdm(self.dataloader, total=len(self.dataloader))

                for batch in loop:

                    self.optimizer.zero_grad()

                    loss = self.step(batch, alpha)
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
                        scheduler=(
                            self.scheduler.get_last_lr()
                            if self.scheduler is not None
                            else None
                        ),
                    )

                    losses.append(loss.item())

                    if self.wandb_logging:
                        wandb.log({"train/loss": loss.item()})

            if not validate:
                continue

            self.cdvi.eval()
            with torch.inference_mode(True):

                loop = tqdm(self.dataloader, total=len(self.dataloader))

                for batch in loop:

                    metric = self.validate(batch, alpha)

                    loop.set_postfix(
                        epoch=epoch,
                        metric=metric.item(),
                    )

        return losses

    @abstractmethod
    def step(self, batch: Tensor, alpha: float | None) -> Tensor:
        pass

    @abstractmethod
    def validate(self, batch: Tensor, alpha: float | None) -> None:
        pass


class StaticTargetTrainer(Trainer):
    def __init__(
        self,
        device: torch.device,
        cdvi: ContextualDVI,
        dataloader: DataLoader[Any],
        optimizer: Optimizer,
        scheduler: ReduceLROnPlateau | None,
        wandb_logging: bool,
    ) -> None:
        super().__init__(
            device,
            cdvi,
            dataloader,
            optimizer,
            scheduler,
            wandb_logging,
        )

        assert self.cdvi.contextual_target is not None

    def step(self, batch: Tensor, alpha: float | None) -> Tensor:
        batch = batch.to(self.device)

        random_context_size: int = np.random.randint(1, batch.shape[1] + 1)
        context = batch[:, 0:random_context_size, :]

        p_z_T = self.cdvi.contextual_target(context, None)

        aggregated, _ = self.cdvi.encoder(context, None)
        log_w, _ = self.cdvi.dvi_process.run_chain(p_z_T, aggregated, None)

        loss = -log_w

        return loss

    def validate(self, batch: Tensor, alpha: float | None) -> Tensor:
        raise NotImplementedError


class BetterStaticTargetTrainer(Trainer):
    def __init__(
        self,
        device: torch.device,
        cdvi: ContextualDVI,
        dataloader: DataLoader[Any],
        optimizer: Optimizer,
        scheduler: ReduceLROnPlateau | None,
        wandb_logging: bool,
    ) -> None:
        super().__init__(
            device,
            cdvi,
            dataloader,
            optimizer,
            scheduler,
            wandb_logging,
        )

        assert self.cdvi.contextual_target is not None

    def step(self, batch: Tensor, alpha: float | None) -> Tensor:

        batch = batch.to(self.device)

        rand_context_sizes = torch.randint(
            low=1, high=batch.shape[1] + 1, size=(batch.shape[0],), device=self.device
        )

        position_indices = torch.arange(batch.shape[1], device=self.device).expand(
            batch.shape[0], -1
        )

        mask = (position_indices < rand_context_sizes.unsqueeze(-1)).float()
        context = batch * mask.unsqueeze(-1).expand(-1, -1, batch.shape[2])

        p_z_T = self.cdvi.contextual_target(context, mask)

        aggregated, _ = self.cdvi.encoder(context, mask)
        log_w, _ = self.cdvi.dvi_process.run_chain(p_z_T, aggregated, mask)

        loss = -log_w

        return loss

    def validate(self, batch: Tensor, alpha: float | None) -> Tensor:
        raise NotImplementedError


class BMLTrainer(Trainer):
    def __init__(
        self,
        device: torch.device,
        cdvi: ContextualDVI,
        dataloader: DataLoader[Any],
        optimizer: Optimizer,
        scheduler: ReduceLROnPlateau | None,
        wandb_logging: bool,
    ) -> None:
        super().__init__(
            device,
            cdvi,
            dataloader,
            optimizer,
            scheduler,
            wandb_logging,
        )

        assert self.cdvi.decoder is not None

    def step(self, batch: Tensor, alpha: float | None) -> Tensor:

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

        log_w, _ = self.cdvi.dvi_process.run_chain(p_z_T, context_embedding, None)

        loss = -log_w

        return loss

    def validate(self, batch: Tensor, alpha: float | None) -> Tensor:
        raise NotImplementedError


class BetterBMLTrainer(Trainer):
    def __init__(
        self,
        device: torch.device,
        cdvi: ContextualDVI,
        dataloader: DataLoader[Any],
        optimizer: Optimizer,
        scheduler: ReduceLROnPlateau | None,
        wandb_logging: bool,
    ) -> None:
        super().__init__(
            device,
            cdvi,
            dataloader,
            optimizer,
            scheduler,
            wandb_logging,
        )

        assert self.cdvi.decoder is not None

    def step(self, batch: Tensor, alpha: float | None) -> Tensor:

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

        log_w, _ = self.cdvi.dvi_process.run_chain(
            p_z_T,
            (
                non_aggregated
                if self.cdvi.dvi_process.control.is_cross_attentive
                else aggregated
            ),
            mask,
        )

        loss = -log_w

        return loss

    def validate(self, batch: Tensor, alpha: float | None) -> None:
        raise NotImplementedError


# def step_bml_alternative(
#     dvi_process: DiffusionVIProcess,
#     set_encoder: SetEncoder,
#     decoder: Decoder,
#     batch: Tensor,
#     device: torch.device,
#     target_constructor: Callable[[Tensor], Distribution],
# ) -> Tensor:

#     x_data, y_data = batch
#     x_data, y_data = x_data.to(device), y_data.to(device)
#     # (batch_size, context_size, x_dim), (batch_size, context_size, y_dim)

#     data = torch.cat([x_data, y_data], dim=-1)
#     # (batch_size, context_size, x_dim + y_dim)

#     data = set_encoder(data)
#     # (batch_size, h_dim)

#     rand_sub_context_size: int = np.random.randint(1, x_data.shape[1] + 1)

#     x_context = x_data[:, 0:rand_sub_context_size, :]
#     y_context = y_data[:, 0:rand_sub_context_size, :]

#     context = torch.cat([x_context, y_context], dim=-1)
#     # (batch_size, context_size, x_dim + y_dim)

#     context = set_encoder(context)
#     # (batch_size, h_dim)

#     p_z_T = target_constructor(context)
#     log_w, z_samples = dvi_process.run_chain(p_z_T, data)

#     log_like_data: Tensor = (
#         decoder(x_data, z_samples[-1], data).log_prob(y_data).mean(dim=0).sum()
#     )

#     log_like_context: Tensor = (
#         decoder(x_context, z_samples[-1], context).log_prob(y_context).mean(dim=0).sum()
#     )

#     loss = -log_like_data - log_like_context - log_w

#     return loss
