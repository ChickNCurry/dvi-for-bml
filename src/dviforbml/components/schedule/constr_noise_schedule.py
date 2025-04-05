import math
from typing import List

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn.functional import softplus

from dviforbml.components.schedule.abstract_schedule import AbstractSchedule


class ConstrNoiseSchedule(AbstractSchedule):
    def __init__(
        self,
        z_dim: int,
        num_steps: int,
        device: torch.device,
        min: float = 0.01,
        max: float = 1,
    ) -> None:
        super(ConstrNoiseSchedule, self).__init__()

        self.num_entries = num_steps + 1

        self.amplitude = nn.Parameter(torch.ones((z_dim), device=device) * (max - min))

        self.cos_sqr_schedule: List[float] = [
            min + np.square(np.cos((math.pi / 2) * (n / self.num_entries)))
            for n in range(0, self.num_entries)
        ]  # (num_entries)

    def get(self, n: int) -> Tensor:
        var_n = ((softplus(self.amplitude) + 1e-6) * self.cos_sqr_schedule[n]).pow(2)
        return var_n


class AggrConstrNoiseSchedule(AbstractSchedule):
    def __init__(
        self,
        z_dim: int,
        h_dim: int,
        num_steps: int,
        num_layers: int,
        non_linearity: str,
        max_context_size: int | None,
        min: float = 0.01,
        max: float = 1,
    ) -> None:
        super(AggrConstrNoiseSchedule, self).__init__()

        self.num_entries = num_steps + 1

        input_size = h_dim + (z_dim if max_context_size is not None else 0)

        self.amp_mlp = nn.Sequential(
            nn.Linear(input_size, h_dim),
            *[
                layer
                for _ in range(num_layers)
                for layer in (getattr(nn, non_linearity)(), nn.Linear(h_dim, h_dim))
            ],
            getattr(nn, non_linearity)(),
            nn.Linear(h_dim, z_dim),
            nn.Softplus(),
        )

        nn.init.constant_(self.amp_mlp[-2].weight, 0)
        nn.init.constant_(self.amp_mlp[-2].bias, max - min)

        self.cos_sqr_schedule: List[float] = [
            min + np.square(np.cos((math.pi / 2) * (n / self.num_entries)))
            for n in range(0, self.num_entries)
        ]  # (num_entries)

    def update(self, r: Tensor, mask: Tensor | None, s: Tensor | None) -> None:
        # (batch_size, num_subtasks, h_dim)
        # (batch_size, num_subtasks, z_dim)

        input = torch.cat([r, s], dim=-1) if s is not None else r
        self.amplitude: Tensor = self.amp_mlp(input) + 1e-6

    def get(self, n: int) -> Tensor:
        var_n = (self.amplitude * self.cos_sqr_schedule[n]).pow(2)
        return var_n


class BCAConstrNoiseSchedule(AbstractSchedule):
    def __init__(
        self,
        z_dim: int,
        h_dim: int,
        num_steps: int,
        num_layers: int,
        non_linearity: str,
        max_context_size: int | None,
        min: float = 0.01,
        max: float = 1,
    ) -> None:
        super(BCAConstrNoiseSchedule, self).__init__()

        self.num_entries = num_steps + 1

        input_size = 2 * h_dim + (z_dim if max_context_size is not None else 0)

        self.amp_mlp = nn.Sequential(
            nn.Linear(input_size, h_dim),
            *[
                layer
                for _ in range(num_layers)
                for layer in (getattr(nn, non_linearity)(), nn.Linear(h_dim, h_dim))
            ],
            getattr(nn, non_linearity)(),
            nn.Linear(h_dim, z_dim),
            nn.Softplus(),
        )

        nn.init.constant_(self.amp_mlp[-2].weight, 0)
        nn.init.constant_(self.amp_mlp[-2].bias, max - min)

        self.cos_sqr_schedule: List[float] = [
            min + np.square(np.cos((math.pi / 2) * (n / self.num_entries)))
            for n in range(0, self.num_entries)
        ]

    def update(self, r: Tensor, mask: Tensor | None, s: Tensor | None) -> None:
        # (batch_size, num_subtasks, h_dim)
        # (batch_size, num_subtasks, z_dim)

        z_mu, z_var = r
        input = torch.cat([z_mu, z_var], dim=-1)
        input = torch.cat([input, s], dim=-1) if s is not None else input
        # (batch_size, num_subtasks, 2 * h_dim)

        self.amplitude: Tensor = self.amp_mlp(input) + 1e-6

    def get(self, n: int) -> Tensor:
        var_n: Tensor = (self.amplitude * self.cos_sqr_schedule[n]).pow(2)
        return var_n


class MHCAConstrNoiseSchedule(AbstractSchedule):
    def __init__(
        self,
        z_dim: int,
        h_dim: int,
        num_steps: int,
        num_layers: int,
        non_linearity: str,
        num_heads: int,
        max_context_size: int | None,
        device: torch.device,
        min: float = 0.01,
        max: float = 1,
    ) -> None:
        super(MHCAConstrNoiseSchedule, self).__init__()

        self.z_dim = z_dim
        self.h_dim = h_dim
        self.num_entries = num_steps + 1

        self.proj_in = nn.Linear(z_dim, h_dim)
        self.cross_attn = nn.MultiheadAttention(h_dim, num_heads, batch_first=True)

        input_size = h_dim + (z_dim if max_context_size is not None else 0)

        self.amp_mlp = nn.Sequential(
            nn.Linear(input_size, h_dim),
            *[
                layer
                for _ in range(num_layers)
                for layer in (getattr(nn, non_linearity)(), nn.Linear(h_dim, h_dim))
            ],
            getattr(nn, non_linearity)(),
            nn.Linear(h_dim, z_dim),
            nn.Softplus(),
        )

        nn.init.constant_(self.amp_mlp[-2].weight, 0)
        nn.init.constant_(self.amp_mlp[-2].bias, max - min)

        self.cos_sqr_schedule: List[float] = [
            min + np.square(np.cos((math.pi / 2) * (n / self.num_entries)))
            for n in range(0, self.num_entries)
        ]

        self.amp_init = torch.ones((z_dim), device=device)

    def update(self, r: Tensor, mask: Tensor | None, s: Tensor | None) -> None:
        # (batch_size, num_subtasks, h_dim)
        # (batch_size, num_subtasks, data_size)
        # (batch_size, num_subtasks, z_dim)

        batch_size = r.shape[0]
        num_subtasks = r.shape[1]
        data_size = r.shape[2]

        r = r.view(batch_size * num_subtasks, data_size, -1)
        # (batch_size * num_subtasks, data_size, h_dim)

        input = self.amp_init[None, None, :].expand(
            batch_size * num_subtasks, 1, self.z_dim
        )  # (batch_size * num_subtasks, 1, z_dim)

        input = self.proj_in(input)
        # (batch_size * num_subtasks, 1, h_dim)

        key_padding_mask = (
            mask.view(batch_size * num_subtasks, data_size).bool().logical_not()
            if mask is not None
            else None
        )  # (batch_size * num_subtasks, data_size)

        input, _ = self.cross_attn(
            query=input, key=r, value=r, key_padding_mask=key_padding_mask
        )  # (batch_size * num_subtasks, 1, h_dim)

        input = input.squeeze(1).view(batch_size, num_subtasks, self.h_dim)
        # (batch_size, num_subtasks, h_dim)

        input = torch.cat([input, s], dim=-1) if s is not None else input
        # (batch_size, num_subtasks, h_dim + z_dim)

        self.amp: Tensor = self.amp_mlp(input) + 1e-6
        # (batch_size, num_subtasks, z_dim)

    def get(self, n: int) -> Tensor:
        var_n: Tensor = (self.amp * self.cos_sqr_schedule[n]).pow(2)
        return var_n
