import math
from typing import List

import numpy as np
import torch
from torch import Tensor, nn

from src.components.schedule.base_schedule import BaseSchedule


class StepSizeSchedule(BaseSchedule):
    def __init__(self, num_steps: int, device: torch.device) -> None:
        super(StepSizeSchedule, self).__init__()

        self.num_entries = num_steps + 1
        self.step_size = torch.tensor(1 / self.num_entries, device=device)

    def get(self, n: int) -> Tensor:
        return self.step_size


class CosineStepSizeSchedule(BaseSchedule):
    def __init__(
        self,
        num_steps: int,
        device: torch.device,
        init_val: float = 0.1,
        require_grad: bool = True,
    ) -> None:
        super(CosineStepSizeSchedule, self).__init__()

        self.num_entries = num_steps + 1

        self.amplitude = nn.Parameter(
            torch.tensor([1], device=device) * init_val, requires_grad=require_grad
        )

        self.cosine_square_schedule: List[float] = [
            np.square(np.cos((math.pi / 2) * (n / self.num_entries)))
            for n in range(0, self.num_entries)
        ]

    def get(self, n: int) -> Tensor:
        delta_t_n = (
            torch.nn.functional.softplus(self.amplitude)
            * self.cosine_square_schedule[n]
        )
        return delta_t_n


class ContextualCosineStepSizeSchedule(BaseSchedule):
    def __init__(
        self,
        h_dim: int,
        non_linearity: str,
        num_steps: int,
        init_val: float = 0.1,
    ) -> None:
        super(ContextualCosineStepSizeSchedule, self).__init__()

        self.num_entries = num_steps + 1

        self.amplitude_mlp = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            getattr(nn, non_linearity)(),
            nn.Linear(h_dim, 1),
            nn.Softplus(),
        )

        nn.init.constant_(self.amplitude_mlp[2].weight, 0)
        nn.init.constant_(self.amplitude_mlp[2].bias, init_val)

        self.cosine_square_schedule: List[float] = [
            np.square(np.cos((math.pi / 2) * (n / self.num_entries)))
            for n in range(0, self.num_entries)
        ]

    def update(self, r: Tensor, mask: Tensor | None) -> None:
        self.amplitude = self.amplitude_mlp(r)

    def get(self, n: int) -> Tensor:
        delta_t_n: Tensor = self.amplitude * self.cosine_square_schedule[n]
        return delta_t_n
