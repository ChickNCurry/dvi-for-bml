from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Any, Dict, Tuple

import numpy as np
import torch
import wandb
from torch import Generator, Tensor, nn
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class AbstractTrainer(ABC):
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
        num_samples: int,
        val_grad_off: bool,
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
        self.num_samples = num_samples
        self.val_grad_off = val_grad_off

    @abstractmethod
    def train_step(
        self, batch: Tensor, alpha: float | None
    ) -> Tuple[Tensor, Dict[str, float]]:
        pass

    @abstractmethod
    def val_step(self, batch: Tensor) -> Dict[str, float]:
        pass

    def val_loop(self, epoch: int) -> None:
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

    def train(
        self,
        num_epochs: int,
        max_clip_norm: float | None = None,
        alpha: float | None = None,
        validate: bool = False,
    ) -> None:

        debug = False

        if debug:
            torch.autograd.set_detect_anomaly(True)

        for epoch in range(num_epochs):

            if validate:

                self.model.eval()

                if self.val_grad_off:
                    with torch.no_grad():
                        self.val_loop(epoch)
                else:
                    self.val_loop(epoch)

            self.model.train()

            loop = tqdm(self.train_loader, total=len(self.train_loader))

            for batch in loop:

                self.optimizer.zero_grad()

                loss, metrics = self.train_step(batch, alpha)

                if debug and torch.isnan(loss) or torch.isinf(loss):
                    print("loss is nan or inf")

                loss.backward()  # type: ignore

                if debug and (
                    torch.stack(
                        [
                            torch.isnan(p.grad).any()
                            for p in self.model.parameters()
                            if p.grad is not None
                        ]
                    ).any()
                    or torch.stack(
                        [
                            torch.isinf(p.grad).any()
                            for p in self.model.parameters()
                            if p.grad is not None
                        ]
                    ).any()
                ):
                    print("grads are nan or inf")

                if max_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), max_clip_norm
                    )

                self.optimizer.step()

                if debug and (
                    torch.stack(
                        [torch.isnan(p).any() for p in self.model.parameters()]
                    ).any()
                    or torch.stack(
                        [torch.isinf(p).any() for p in self.model.parameters()]
                    ).any()
                ):
                    print("params are nan or inf")

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

    def get_data_subtasks(self, batch: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
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

    def get_data_samples(self, batch: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        x_data, y_data = batch
        x_data = x_data.to(self.device)
        y_data = y_data.to(self.device)
        # (batch_size, data_size, x_dim)
        # (batch_size, data_size, y_dim)

        x_data = x_data.unsqueeze(1).expand(-1, self.num_samples, -1, -1)
        y_data = y_data.unsqueeze(1).expand(-1, self.num_samples, -1, -1)
        # (batch_size, num_subtasks, data_size, x_dim)
        # (batch_size, num_subtasks, data_size, y_dim)

        data = torch.cat([x_data, y_data], dim=-1)
        # (batch_size, num_subtasks, data_size, x_dim + y_dim)

        return data, x_data, y_data

    def get_mask(self, alpha: float | None, data: Tensor) -> Tensor:
        # (batch_size, num_subtasks, data_size, x_dim + y_dim)

        if alpha is None:
            rand_context_sizes = torch.randint(
                low=1,
                high=data.shape[2] + 1,
                size=(data.shape[0], data.shape[1], 1),
                device=self.device,
                # generator=self.generator,
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
        )  # (batch_size, num_subtasks, data_size)

        mask = (pos_indices < rand_context_sizes).float()
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
        )  # (batch_size, num_samples, data_size)

        mask = (pos_indices < context_sizes).float()
        # (batch_size, num_samples, data_size)

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
