import math
from abc import ABC, abstractmethod
from typing import List

import numpy as np
import torch
from torch import Tensor, nn


class NoiseSchedule(nn.Module):
    def __init__(self, z_dim: int, num_steps: int, device: torch.device) -> None:

        # init amplitude with 0.1
        self.amplitude = nn.Parameter(torch.ones((z_dim), device=device))

        self.cosine_square_schedule: List[float] = [
            np.square(np.cos((math.pi / 2) * (n / num_steps)))
            for n in range(0, num_steps)
        ]

    def get(self, n: int) -> Tensor:
        # transition from n-1 to n

        noise = (
            torch.nn.functional.softplus(self.amplitude)
            * self.cosine_square_schedule[n - 1]
        )
        # start at amplitude
        # never reaches 0

        return noise


class ContextualNoiseSchedule(nn.Module):
    def __init__(
        self,
        h_dim: int,
        z_dim: int,
        non_linearity: str,
        num_steps: int,
    ) -> None:

        self.amplitude_mlp = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            getattr(nn, non_linearity)(),
            nn.Linear(h_dim, z_dim),
            nn.Softplus(),
        )

        # init amplitude_mlp such that amplitude is 1
        nn.init.constant_(self.amplitude_mlp[-1].weight, 0)
        nn.init.constant_(self.amplitude_mlp[-1].bias, 1)

        self.cosine_square_schedule: List[float] = [
            np.square(np.cos((math.pi / 2) * (n / num_steps)))
            for n in range(0, num_steps)
        ]

    def update(self, r: Tensor) -> None:
        self.amplitude = self.amplitude_mlp(r)

    def get(self, n: int) -> Tensor:
        # transition from n-1 to n

        sigma: Tensor = self.amplitude * self.cosine_square_schedule[n - 1]
        # start at amplitude
        # never reaches 0

        return sigma


class StepSizeSchedule(nn.Module):
    def __init__(self, z_dim: int, num_steps: int, device: torch.device) -> None:

        # init amplitude with 0.1
        self.amplitude = nn.Parameter(torch.ones((z_dim), device=device) * 0.1)

        self.cosine_square_schedule: List[float] = [
            np.square(np.cos((math.pi / 2) * (n / num_steps)))
            for n in range(0, num_steps)
        ][::-1]

    def get(self, n: int) -> Tensor:
        # transition from n-1 to n

        step_size = (
            torch.nn.functional.softplus(self.amplitude)
            * self.cosine_square_schedule[n - 1]
        )
        # starts higher than 0
        # reaches amplitude

        return step_size


class ContextualStepSizeSchedule(nn.Module):
    def __init__(
        self,
        h_dim: int,
        z_dim: int,
        non_linearity: str,
        num_steps: int,
    ) -> None:

        self.amplitude_mlp = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            getattr(nn, non_linearity)(),
            nn.Linear(h_dim, z_dim),
            nn.Softplus(),
        )

        # init amplitude_mlp such that amplitude is 0.1
        nn.init.constant_(self.amplitude_mlp[-1].weight, 0)
        nn.init.constant_(self.amplitude_mlp[-1].bias, 0.1)

        self.cosine_square_schedule: List[float] = [
            np.square(np.cos((math.pi / 2) * (n / num_steps)))
            for n in range(0, num_steps)
        ][::-1]

    def update(self, r: Tensor) -> None:
        self.amplitude = self.amplitude_mlp(r)

    def get(self, n: int) -> Tensor:
        # transition from n-1 to n

        step_size: Tensor = self.amplitude * self.cosine_square_schedule[n - 1]
        # starts higher than 0
        # reaches amplitude

        return step_size


class AnnealingSchedule(nn.Module):
    def __init__(self, z_dim: int, num_steps: int, device: torch.device) -> None:

        # init params such that betas increase linearly
        self.params = nn.Parameter(
            torch.ones((num_steps, z_dim), device=device) / num_steps
        )

    def get(self, n: int) -> Tensor:
        # transition from n-1 to n

        params = torch.nn.functional.softplus(self.params)
        betas = torch.cumsum(params, dim=0) / torch.sum(params, dim=0)
        beta = betas[n - 1]
        # starts higher than 0
        # reaches 1

        return beta


class ContextualAnnealingSchedule(nn.Module):
    def __init__(
        self,
        h_dim: int,
        z_dim: int,
        non_linearity: str,
        num_steps: int,
    ) -> None:

        self.params_mlp = nn.Sequential(
            nn.Linear(num_steps * h_dim, num_steps * h_dim),
            getattr(nn, non_linearity)(),
            nn.Linear(num_steps * h_dim, num_steps * z_dim),
            nn.Softplus(),
        )

        # init params_mlp corresponding to linearly increasing betas
        nn.init.constant_(self.params_mlp[-1].weight, 0)
        nn.init.constant_(self.params_mlp[-1].bias, 1 / num_steps)

        self.num_steps = num_steps

    def update(self, r: Tensor) -> None:
        r = r.expand(self.num_steps, -1).flatten()
        self.params: Tensor = self.params_mlp(r).view(self.num_steps, -1)

    def get(self, n: int) -> Tensor:
        # transition from n-1 to n

        betas = torch.cumsum(self.params, dim=0) / torch.sum(self.params, dim=0)
        beta = betas[n - 1]
        # starts higher than 0
        # reaches 1

        return beta


# class Schedule(ABC, nn.Module):
#     def __init__(self, z_dim: int, num_steps: int, device: torch.device) -> None:
#         super(Schedule, self).__init__()

#         self.z_dim = z_dim
#         self.num_steps = num_steps
#         self.device = device

#     @abstractmethod
#     def get(self, n: int) -> Tensor:
#         pass

# class AnnealingSchedule(Schedule):
#     def __init__(
#         self, z_dim: int, num_steps: int, device: torch.device, min: float
#     ) -> None:
#         super(AnnealingSchedule, self).__init__(z_dim, num_steps, device)

#         self.schedule = nn.ParameterList(
#             [
#                 nn.Parameter(torch.ones((self.z_dim), device=device) * beta)
#                 for beta in np.linspace(min, 1 - min, num_steps)
#             ]
#         )

#     def get(self, t: int) -> Tensor:
#         entry: Tensor = self.schedule[t - 1]

#         return entry


# class CosineSchedule(Schedule):
#     def __init__(
#         self, z_dim: int, num_steps: int, device: torch.device, min: float
#     ) -> None:
#         super(CosineSchedule, self).__init__(z_dim, num_steps, device)

#         self.schedule = nn.ParameterList(
#             [
#                 nn.Parameter(torch.ones((z_dim), device=device) * beta)
#                 for beta in [
#                     (1 - min) * np.cos(math.pi * (1 - t) * 0.5) ** 2 + min
#                     for t in np.linspace(1, 0, num_steps)
#                 ]
#             ]
#         )

#     def get(self, t: int) -> Tensor:
#         entry: Tensor = self.schedule[t - 1]

#         return entry


# class ContextualSchedule(Schedule):
#     def __init__(
#         self,
#         z_dim: int,
#         num_steps: int,
#         h_dim: int,
#         num_layers: int,
#         non_linearity: str,
#         is_cross_attentive: bool,
#         num_heads: int | None,
#         device: torch.device,
#         r_aggr: Tensor | None,
#         r_non_aggr: Tensor | None,
#         mask: Tensor | None,
#         min: float = 0.1,
#     ) -> None:
#         super(ContextualSchedule, self).__init__(z_dim, num_steps, device, min)

#         self.is_cross_attentive = is_cross_attentive
#         self.num_heads = num_heads

#         if self.is_cross_attentive:
#             assert self.num_heads is not None
#             self.cross_attn = nn.MultiheadAttention(h_dim, num_heads, batch_first=True)

#         self.mlp = nn.Sequential(
#             *[
#                 layer
#                 for layer in (getattr(nn, non_linearity)(), nn.Linear(h_dim, h_dim))
#                 for _ in range(num_layers - 2)
#             ],
#             getattr(nn, non_linearity)()
#         )

#         self.proj_out = nn.Linear(h_dim, z_dim)

#         self.schedule = self.forward(r_aggr, r_non_aggr, mask)

#     def forward(
#         self,
#         r_aggr: Tensor | None,
#         r_non_aggr: Tensor | None,
#         mask: Tensor | None,
#     ) -> Tuple[Tensor, Tensor]:
#         # (batch_size, num_subtasks, h_dim)
#         # (batch_size, num_subtasks, context_size, h_dim)
#         # (batch_size, num_subtasks, context_size)

#         h: Tensor = self.proj_t(torch.tensor([t], device=z.device)) + self.proj_z(z)
#         # (batch_size, num_subtasks, h_dim)

#         if self.is_cross_attentive:
#             assert r_non_aggr is not None and self.num_heads is not None

#             h = h.unsqueeze(2)
#             # (batch_size, num_subtasks, 1, h_dim)

#             mask = (
#                 mask.unsqueeze(2)
#                 .view(batch_size * num_subtasks, -1, -1)
#                 .repeat(self.num_heads, 1, 1)
#                 if mask is not None
#                 else None
#             )  # (num_heads * num_subtasks * batch_size, 1, context_size)

#             h, _ = self.cross_attn(
#                 query=h, key=r_non_aggr, value=r_non_aggr, attn_mask=mask
#             )
#             # (batch_size, num_subtasks, 1, h_dim)

#             h = h.squeeze(2)
#             # (batch_size, num_subtasks, h_dim)

#         else:
#             assert r_aggr is not None

#             h = h + r_aggr
#             # (batch_size, num_subtasks, h_dim)

#         h = self.mlp(h)

#         control_t: Tensor = self.proj_control(h)  # + z  # TODO skip connection
#         sigma_t: Tensor = (
#             nn.Softplus()(s + self.proj_sigma(h)) if s is not None else None
#         )
#         # (batch_size, num_subtasks, z_dim)

#         return control_t, sigma_t
