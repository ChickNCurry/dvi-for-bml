import math
from typing import List

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn.functional import softplus

from src.components.schedule.abstract_schedule import AbstractSchedule


class CosineNoiseSchedule(AbstractSchedule):
    def __init__(
        self,
        z_dim: int,
        num_steps: int,
        device: torch.device,
        min: float = 0.01,
        max: float = 1,
        requires_grad: bool = True,
    ) -> None:
        super(CosineNoiseSchedule, self).__init__()

        self.num_entries = num_steps + 1

        self.amplitude = nn.Parameter(
            torch.ones((z_dim), device=device) * (max - min),
            requires_grad=requires_grad,
        )

        self.cosine_square_schedule: List[float] = [
            min + np.square(np.cos((math.pi / 2) * (n / self.num_entries)))
            for n in range(0, self.num_entries)
        ]  # (num_entries)

    def get(self, n: int) -> Tensor:
        var_n = (softplus(self.amplitude) * self.cosine_square_schedule[n]).pow(2)
        return var_n


class AggrCosineNoiseSchedule(AbstractSchedule):
    def __init__(
        self,
        z_dim: int,
        h_dim: int,
        non_linearity: str,
        num_steps: int,
        min: float = 0.01,
        max: float = 1,
    ) -> None:
        super(AggrCosineNoiseSchedule, self).__init__()

        self.num_entries = num_steps + 1

        self.amplitude_mlp = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            getattr(nn, non_linearity)(),
            nn.Linear(h_dim, z_dim),
            nn.Softplus(),
        )

        nn.init.constant_(self.amplitude_mlp[2].weight, 0)
        nn.init.constant_(self.amplitude_mlp[2].bias, max - min)

        self.cosine_square_schedule: List[float] = [
            min + np.square(np.cos((math.pi / 2) * (n / self.num_entries)))
            for n in range(0, self.num_entries)
        ]  # (num_entries)

    def update(self, r: Tensor, mask: Tensor | None) -> None:
        # (batch_size, num_subtasks, h_dim)

        self.amplitude: Tensor = self.amplitude_mlp(r)

    def get(self, n: int) -> Tensor:
        var_n = (self.amplitude * self.cosine_square_schedule[n]).pow(2)
        return var_n


class BCACosineNoiseSchedule(AbstractSchedule):
    def __init__(
        self,
        z_dim: int,
        h_dim: int,
        non_linearity: str,
        num_steps: int,
        min: float = 0.01,
        max: float = 1,
    ) -> None:
        super(BCACosineNoiseSchedule, self).__init__()

        self.num_entries = num_steps + 1

        self.amplitude_mlp = nn.Sequential(
            nn.Linear(2 * h_dim, h_dim),
            getattr(nn, non_linearity)(),
            nn.Linear(h_dim, z_dim),
            nn.Softplus(),
        )

        nn.init.constant_(self.amplitude_mlp[2].weight, 0)
        nn.init.constant_(self.amplitude_mlp[2].bias, max - min)

        self.cosine_square_schedule: List[float] = [
            min + np.square(np.cos((math.pi / 2) * (n / self.num_entries)))
            for n in range(0, self.num_entries)
        ]

    def update(self, r: Tensor, mask: Tensor | None) -> None:
        # (batch_size, num_subtasks, h_dim)

        z_mu, z_var = r
        input = torch.cat([z_mu, z_var], dim=-1)
        # (batch_size, num_subtasks, 2 * h_dim)

        self.amplitude: Tensor = self.amplitude_mlp(input)

    def get(self, n: int) -> Tensor:
        var_n: Tensor = (self.amplitude * self.cosine_square_schedule[n]).pow(2)
        return var_n
