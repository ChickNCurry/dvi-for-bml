import math
from typing import List

import numpy as np
import torch
from torch import Tensor, nn

from src.components.schedule.base_schedule import BaseSchedule


class NoiseSchedule(BaseSchedule):
    def __init__(
        self,
        z_dim: int,
        num_steps: int,
        device: torch.device,
        min: float = 0.01,
        max: float = 1,
        requires_grad: bool = True,
    ) -> None:

        self.noise_schedule = nn.Parameter(
            torch.linspace(max, min, num_steps, device=device)
            .unsqueeze(1)
            .expand(num_steps, z_dim),
            requires_grad=requires_grad,
        )

    def get(self, n: int) -> Tensor:
        var_n = torch.nn.functional.softplus(self.noise_schedule[n - 1, :])
        return var_n


class ContextualNoiseSchedule(BaseSchedule, nn.Module):
    def __init__(
        self,
        z_dim: int,
        h_dim: int,
        non_linearity: str,
        num_steps: int,
        device: torch.device,
        min: float = 0.01,
        max: float = 1,
    ) -> None:
        super(ContextualNoiseSchedule, self).__init__()

        self.z_dim = z_dim
        self.num_steps = num_steps

        self.noise_mlp = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            getattr(nn, non_linearity)(),
            nn.Linear(h_dim, num_steps * z_dim),
        )

        nn.init.constant_(self.noise_mlp[-1].weight, 0)
        nn.init.constant_(self.noise_mlp[-1].bias, 0)

        self.noise_init = torch.linspace(max, min, num_steps, device=device)

    def update(self, r: Tensor) -> None:
        noise: Tensor = self.noise_mlp(r)
        noise = noise.view(r.shape[0], r.shape[1], self.z_dim, self.num_steps)
        noise_init = self.noise_init[None, None, None, :]
        self.noise_schedule = nn.functional.softplus(noise + noise_init)

    def get(self, n: int) -> Tensor:
        var_n: Tensor = self.noise_schedule[:, :, :, n - 1]
        return var_n


class CosineNoiseSchedule(BaseSchedule, nn.Module):
    def __init__(
        self,
        z_dim: int,
        num_steps: int,
        device: torch.device,
        init_val: float = 1.0,
        requires_grad: bool = True,
    ) -> None:
        super(CosineNoiseSchedule, self).__init__()

        self.amplitude = nn.Parameter(
            torch.ones((z_dim), device=device) * init_val,
            requires_grad=requires_grad,
        )

        self.cosine_square_schedule: List[float] = [
            np.square(np.cos((math.pi / 2) * (n / num_steps)))
            for n in range(0, num_steps)
        ]

    def get(self, n: int) -> Tensor:
        var_n = (
            nn.functional.softplus(self.amplitude) * self.cosine_square_schedule[n - 1]
        )
        return var_n


class ContextualCosineNoiseSchedule(BaseSchedule, nn.Module):
    def __init__(
        self,
        z_dim: int,
        h_dim: int,
        non_linearity: str,
        num_steps: int,
        init_val: float = 1.0,
    ) -> None:
        super(ContextualCosineNoiseSchedule, self).__init__()

        self.amplitude_mlp = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            getattr(nn, non_linearity)(),
            nn.Linear(h_dim, z_dim),
            nn.Softplus(),
        )

        nn.init.constant_(self.amplitude_mlp[2].weight, 0)
        nn.init.constant_(self.amplitude_mlp[2].bias, init_val)

        self.cosine_square_schedule: List[float] = [
            np.square(np.cos((math.pi / 2) * (n / num_steps)))
            for n in range(0, num_steps)
        ]

    def update(self, r: Tensor) -> None:
        self.amplitude = self.amplitude_mlp(r)

    def get(self, n: int) -> Tensor:
        var_n: Tensor = self.amplitude * self.cosine_square_schedule[n - 1]
        return var_n
