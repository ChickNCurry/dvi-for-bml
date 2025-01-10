import math
from typing import List

import numpy as np
import torch
from torch import Tensor, nn

from src.components.schedule.base_schedule import BaseSchedule


class StepSizeSchedule(BaseSchedule):
    def __init__(self, num_steps: int, device: torch.device) -> None:
        self.step_size = torch.tensor(1 / num_steps, device=device)

    def get(self, n: int) -> Tensor:
        return self.step_size


class CosineStepSizeSchedule(BaseSchedule, nn.Module):
    def __init__(
        self,
        num_steps: int,
        device: torch.device,
        init_val: float = 0.1,
        require_grad: bool = True,
    ) -> None:
        super(CosineStepSizeSchedule, self).__init__()

        self.amplitude = nn.Parameter(
            torch.tensor([1], device=device) * init_val, requires_grad=require_grad
        )

        self.cosine_square_schedule: List[float] = [
            np.square(np.cos((math.pi / 2) * (n / num_steps)))
            for n in range(0, num_steps)
        ]

    def get(self, n: int) -> Tensor:
        delta_t_n = (
            torch.nn.functional.softplus(self.amplitude)
            * self.cosine_square_schedule[n - 1]
        )
        return delta_t_n


class ContextualCosineStepSizeSchedule(BaseSchedule, nn.Module):
    def __init__(
        self,
        h_dim: int,
        non_linearity: str,
        num_steps: int,
        init_val: float = 0.1,
    ) -> None:
        super(ContextualCosineStepSizeSchedule, self).__init__()

        self.amplitude_mlp = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            getattr(nn, non_linearity)(),
            nn.Linear(h_dim, 1),
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
        delta_t_n: Tensor = self.amplitude * self.cosine_square_schedule[n - 1]
        return delta_t_n
