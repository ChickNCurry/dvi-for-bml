from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Any, Dict, Tuple

import numpy as np
import torch
from torch import Generator, Tensor, nn
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import wandb


class BaseTrainer(ABC):
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        dataset: Dataset[Any],
        train_loader: DataLoader[Any],
        val_loader: DataLoader[Any],
        optimizer: Optimizer,
        scheduler: LRScheduler | None,
        generator: Generator,
        wandb_logging: bool,
        num_subtasks: int,
        sample_size: int,
    ) -> None:
        self.model = model
        self.device = device
        self.dataset = dataset
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.generator = generator
        self.wandb_logging = wandb_logging
        self.num_subtasks = num_subtasks
        self.sample_size = sample_size

    @abstractmethod
    def train_step(
        self, batch: Tensor, alpha: float | None
    ) -> Tuple[Tensor, Dict[str, float]]:
        pass

    @abstractmethod
    def val_step(self, batch: Tensor) -> Dict[str, float]:
        pass

    def train(
        self,
        num_epochs: int,
        max_clip_norm: float | None = None,
        alpha: float | None = None,
        validate: bool = False,
    ) -> None:

        # torch.autograd.set_detect_anomaly(True)

        for epoch in range(num_epochs):

            if validate:

                self.model.eval()
                # with torch.inference_mode(False):

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

            self.model.train()
            # with torch.inference_mode(False):

            loop = tqdm(self.train_loader, total=len(self.train_loader))

            for batch in loop:

                self.optimizer.zero_grad()

                loss, metrics = self.train_step(batch, alpha)
                loss.backward()  # type: ignore

                if max_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), max_clip_norm
                    )

                self.optimizer.step()

                loop.set_postfix(
                    ordered_dict=OrderedDict(
                        [
                            ("epoch", epoch),
                            ("loss", loss.item()),
                            *[(k, v) for k, v in metrics.items()],
                        ]
                    )
                )

                if self.wandb_logging:
                    wandb.log(
                        {
                            "train/loss": loss.item(),
                            **(
                                {f"train/{k}": v for k, v in metrics.items()}
                                if metrics is not None
                                else {}
                            ),
                        }
                    )

            if self.scheduler is not None:
                self.scheduler.step(epoch)
                print(self.scheduler.get_last_lr())

    def get_data(self, batch: Tensor) -> Tuple[Tensor, Tensor, Tensor]:

        x_data, y_data = batch
        x_data = x_data.to(self.device)
        y_data = y_data.to(self.device)
        # (batch_size, data_size, x_dim)
        # (batch_size, data_size, y_dim)

        x_data = x_data.unsqueeze(1).expand(-1, self.num_subtasks, -1, -1)
        y_data = y_data.unsqueeze(1).expand(-1, self.num_subtasks, -1, -1)
        # (batch_size, num_subtasks, data_size, x_dim)
        # (batch_size, num_subtasks, data_size, y_dim)

        data = torch.cat([x_data, y_data], dim=-1)
        # (batch_size, num_subtasks, data_size, x_dim + y_dim)

        return data, x_data, y_data

    def get_mask(self, alpha: float | None, data: Tensor) -> Tensor:
        # (batch_size, num_subtasks, data_size, x_dim + y_dim)

        if alpha is None:
            rand_data_sizes = torch.randint(
                low=1,
                high=data.shape[2] + 1,
                size=(data.shape[0], data.shape[1], 1),
                device=self.device,
                # generator=self.generator,
            )  # (batch_size, num_subtasks, 1)

        else:
            rand_data_sizes = torch.tensor(
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
        )  # (batch_size, num_subtasks, data_size)

        mask = (pos_indices < rand_data_sizes).float()
        # (batch_size, num_subtasks, data_size)

        return mask

    def get_mask_1(self, data: Tensor) -> Tensor:
        # (batch_size, num_subtasks, data_size, x_dim + y_dim)

        context_sizes = torch.ones(
            size=(data.shape[0], data.shape[1], 1),
            device=self.device,
        )  # (batch_size, num_subtasks, 1)

        pos_indices = (
            torch.arange(data.shape[2], device=self.device)
            .unsqueeze(0)
            .unsqueeze(0)
            .expand(data.shape[0], data.shape[1], -1)
        )  # (batch_size, sample_size, data_size)

        mask = (pos_indices < context_sizes).float()
        # (batch_size, sample_size, data_size)

        return mask

    def get_context(
        self, data: Tensor, x_data: Tensor, mask: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        context = data * mask.unsqueeze(-1).expand(-1, -1, -1, data.shape[-1])
        # (batch_size, num_subtasks, data_size, x_dim + y_dim)

        x_context = context[:, :, :, 0 : x_data.shape[-1]]
        y_context = context[:, :, :, x_data.shape[-1] : data.shape[-1]]
        # (batch_size, num_subtasks, data_size, x_dim)
        # (batch_size, num_subtasks, data_size, y_dim)

        return context, x_context, y_context

    def get_target(
        self, data: Tensor, x_data: Tensor, mask: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        target = data * (1 - mask).unsqueeze(-1).expand(-1, -1, -1, data.shape[-1])
        # (batch_size, num_subtasks, data_size, x_dim + y_dim)

        x_target = target[:, :, :, 0 : x_data.shape[-1]]
        y_target = target[:, :, :, x_data.shape[-1] : data.shape[-1]]
        # (batch_size, num_subtasks, data_size, x_dim)
        # (batch_size, num_subtasks, data_size, y_dim)

        return target, x_target, y_target
