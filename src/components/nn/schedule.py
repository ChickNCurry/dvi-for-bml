import math
from abc import ABC, abstractmethod

import numpy as np
import torch
from torch import Tensor, nn


class Schedule(ABC, nn.Module):
    def __init__(self, z_dim: int, num_steps: int, device: torch.device) -> None:
        super(Schedule, self).__init__()

        self.z_dim = z_dim
        self.num_steps = num_steps
        self.device = device

    @abstractmethod
    def get(self, t: int) -> Tensor:
        pass


class LinearSchedule(Schedule):
    def __init__(
        self, z_dim: int, num_steps: int, device: torch.device, min: float
    ) -> None:
        super(LinearSchedule, self).__init__(z_dim, num_steps, device)

        self.schedule = nn.ParameterList(
            [
                nn.Parameter(torch.ones((self.z_dim), device=device) * beta)
                for beta in np.linspace(min, 1 - min, num_steps)
            ]
        )

    def get(self, t: int) -> Tensor:
        entry: Tensor = self.schedule[t - 1]

        return entry


class CosineSchedule(Schedule):
    def __init__(
        self, z_dim: int, num_steps: int, device: torch.device, min: float
    ) -> None:
        super(CosineSchedule, self).__init__(z_dim, num_steps, device)

        self.schedule = nn.ParameterList(
            [
                nn.Parameter(torch.ones((z_dim), device=device) * beta)
                for beta in [
                    (1 - min) * np.cos(math.pi * (1 - t) * 0.5) ** 2 + min
                    for t in np.linspace(1, 0, num_steps)
                ]
            ]
        )

    def get(self, t: int) -> Tensor:
        entry: Tensor = self.schedule[t - 1]

        return entry


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
