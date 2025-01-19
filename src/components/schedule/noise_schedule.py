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

        self.num_entries = num_steps + 1

        self.noise_schedule = nn.Parameter(
            torch.linspace(max, min, self.num_entries, device=device)
            .unsqueeze(1)
            .expand(self.num_entries, z_dim),
            requires_grad=requires_grad,
        )

    def get(self, n: int) -> Tensor:
        var_n = torch.nn.functional.softplus(self.noise_schedule[n, :])
        return var_n


class AggrNoiseSchedule(BaseSchedule, nn.Module):
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
        super(AggrNoiseSchedule, self).__init__()

        self.z_dim = z_dim
        self.num_entries = num_steps + 1

        self.noise_mlp = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            getattr(nn, non_linearity)(),
            nn.Linear(h_dim, z_dim * self.num_entries),
        )

        nn.init.constant_(self.noise_mlp[-1].weight, 0)
        nn.init.constant_(self.noise_mlp[-1].bias, 0)

        self.noise_init = torch.linspace(max, min, self.num_entries, device=device)

    def update(self, r: Tensor, mask: Tensor | None) -> None:
        # (batch_size, num_subtasks, h_dim)

        batch_size = r.shape[0]
        num_subtasks = r.shape[1]

        noise: Tensor = self.noise_mlp(r)
        # (batch_size, num_subtasks, z_dim * num_entries)

        noise = noise.view(batch_size, num_subtasks, self.z_dim, self.num_entries)
        # (batch_size, num_subtasks, z_dim, num_entries)

        self.noise_schedule = nn.functional.softplus(
            noise + self.noise_init[None, None, None, :]
        )  # (batch_size, num_subtasks, z_dim, num_entries)

    def get(self, n: int) -> Tensor:
        var_n: Tensor = self.noise_schedule[:, :, :, n]
        # (batch_size, num_subtasks, z_dim)

        return var_n


class BCANoiseSchedule(BaseSchedule, nn.Module):
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
        super(BCANoiseSchedule, self).__init__()

        self.z_dim = z_dim
        self.num_entries = num_steps + 1

        self.proj_z_mu = nn.Linear(h_dim, h_dim)
        self.proj_z_var = nn.Linear(h_dim, h_dim)

        self.noise_mlp = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            getattr(nn, non_linearity)(),
            nn.Linear(h_dim, z_dim * self.num_entries),
        )

        nn.init.constant_(self.noise_mlp[-1].weight, 0)
        nn.init.constant_(self.noise_mlp[-1].bias, 0)

        self.noise_init = torch.linspace(max, min, self.num_entries, device=device)

    def update(self, r: Tensor, mask: Tensor | None) -> None:
        # (batch_size, num_subtasks, h_dim)

        batch_size = r[0].shape[0]
        num_subtasks = r[0].shape[1]

        z_mu, z_var = r
        z_mu, z_var = self.proj_z_mu(z_mu), self.proj_z_var(z_var)
        # (batch_size, num_subtasks, h_dim)

        noise: Tensor = self.noise_mlp(z_mu + z_var)
        # (batch_size, num_subtasks, z_dim * num_entries)

        noise = noise.view(batch_size, num_subtasks, self.z_dim, self.num_entries)
        # (batch_size, num_subtasks, z_dim, num_entries)

        self.noise_schedule = nn.functional.softplus(
            noise + self.noise_init[None, None, None, :]
        )  # (batch_size, num_subtasks, z_dim, num_entries)

    def get(self, n: int) -> Tensor:
        var_n: Tensor = self.noise_schedule[:, :, :, n]
        # (batch_size, num_subtasks, z_dim)

        return var_n


class MHANoiseSchedule(BaseSchedule, nn.Module):
    def __init__(
        self,
        z_dim: int,
        h_dim: int,
        non_linearity: str,
        num_steps: int,
        device: torch.device,
        num_heads: int,
        min: float = 0.01,
        max: float = 1,
    ) -> None:
        super(MHANoiseSchedule, self).__init__()

        self.z_dim = z_dim
        self.num_entries = num_steps + 1

        self.proj_in = nn.Linear(1, h_dim)
        self.cross_attn = nn.MultiheadAttention(h_dim, num_heads, batch_first=True)
        self.noise_mlp = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            getattr(nn, non_linearity)(),
            nn.Linear(h_dim, z_dim),
        )

        nn.init.constant_(self.noise_mlp[-1].weight, 0)
        nn.init.constant_(self.noise_mlp[-1].bias, 0)

        self.noise_init = torch.linspace(max, min, self.num_entries, device=device)

    def update(self, r: Tensor, mask: Tensor | None) -> None:
        # (batch_size, num_subtasks, context_size, h_dim)
        # (batch_size, num_subtasks, context_size)

        batch_size = r.shape[0]
        num_subtasks = r.shape[1]
        context_size = r.shape[2]

        r = r.view(batch_size * num_subtasks, context_size, -1)
        # (batch_size * num_subtasks, context_size, h_dim)

        h: Tensor = self.proj_in(self.noise_init.unsqueeze(1))
        # (num_entries, h_dim)

        h = h.unsqueeze(0).expand(batch_size * num_subtasks, -1, -1)
        # (batch_size * num_subtasks, num_entries, h_dim)

        key_padding_mask = (
            mask.view(batch_size * num_subtasks, -1).bool().logical_not()
            if mask is not None
            else None
        )  # (batch_size * num_subtasks, context_size)

        h, _ = self.cross_attn(
            query=h,
            key=r,
            value=r,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )  # (batch_size * num_subtasks, num_entries, h_dim)

        h = h.view(batch_size, num_subtasks, self.num_entries, -1)
        # (batch_size, num_subtasks, num_entries, h_dim)

        noise = self.noise_mlp(h).transpose(2, 3)
        # (batch_size, num_subtasks, z_dim, num_entries)

        self.noise_schedule = nn.functional.softplus(
            noise + self.noise_init[None, None, None, :]
        )  # (batch_size, num_subtasks, z_dim, num_entries)

    def get(self, n: int) -> Tensor:
        var_n: Tensor = self.noise_schedule[:, :, :, n]
        # (batch_size, num_subtasks, z_dim)

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

        self.num_entries = num_steps + 1

        self.amplitude = nn.Parameter(
            torch.ones((z_dim), device=device) * init_val,
            requires_grad=requires_grad,
        )

        self.cosine_square_schedule: List[float] = [
            np.square(np.cos((math.pi / 2) * (n / self.num_entries)))
            for n in range(0, self.num_entries)
        ]

    def get(self, n: int) -> Tensor:
        var_n = nn.functional.softplus(self.amplitude) * self.cosine_square_schedule[n]
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

        self.num_entries = num_steps + 1

        self.amplitude_mlp = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            getattr(nn, non_linearity)(),
            nn.Linear(h_dim, z_dim),
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
        var_n: Tensor = self.amplitude * self.cosine_square_schedule[n]
        return var_n
