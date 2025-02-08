import math
from typing import List

import numpy as np
import torch
from torch import Tensor, nn

from src.components.schedule.base_schedule import BaseSchedule


class CosineNoiseSchedule(BaseSchedule, nn.Module):
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
        ]

    def get(self, n: int) -> Tensor:
        var_n = (
            nn.functional.softplus(self.amplitude) * self.cosine_square_schedule[n]
        ).pow(2)

        return var_n


class AggrCosineNoiseSchedule(BaseSchedule, nn.Module):
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
        ]

    def update(self, r: Tensor, mask: Tensor | None) -> None:
        # (batch_size, num_subtasks, h_dim)

        self.amplitude: Tensor = self.amplitude_mlp(r)

    def get(self, n: int) -> Tensor:
        var_n = (self.amplitude * self.cosine_square_schedule[n]).pow(2)
        return var_n


class BCACosineNoiseSchedule(BaseSchedule, nn.Module):
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

        self.proj_z_mu = nn.Linear(h_dim, h_dim)
        self.proj_z_var = nn.Linear(h_dim, h_dim)

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
        ]

    def update(self, r: Tensor, mask: Tensor | None) -> None:
        # (batch_size, num_subtasks, h_dim)

        z_mu, z_var = r
        z_mu, z_var = self.proj_z_mu(z_mu), self.proj_z_var(z_var)
        # (batch_size, num_subtasks, h_dim)

        self.amplitude: Tensor = self.amplitude_mlp(z_mu + z_var)

    def get(self, n: int) -> Tensor:
        var_n: Tensor = (self.amplitude * self.cosine_square_schedule[n]).pow(2)
        return var_n
