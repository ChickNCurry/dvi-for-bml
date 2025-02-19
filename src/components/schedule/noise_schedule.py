import torch
from torch import Tensor, nn
from torch.nn.functional import softplus

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
        super(NoiseSchedule, self).__init__()

        self.num_entries = num_steps + 1

        self.vars = nn.Parameter(
            torch.linspace(max, min, self.num_entries, device=device)
            .unsqueeze(1)
            .expand(self.num_entries, z_dim)
            .pow(2),
            requires_grad=requires_grad,
        )  # (num_entries, z_dim)

    def get(self, n: int) -> Tensor:
        var_n = softplus(self.vars[n, :])
        return var_n


class AggrNoiseSchedule(BaseSchedule):
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

        self.vars_init = torch.linspace(max, min, self.num_entries, device=device).pow(
            2
        )  # (num_entries)

    def update(self, r: Tensor, mask: Tensor | None) -> None:
        # (batch_size, num_subtasks, h_dim)

        batch_size = r.shape[0]
        num_subtasks = r.shape[1]

        vars_pred: Tensor = self.noise_mlp(r)
        # (batch_size, num_subtasks, z_dim * num_entries)

        vars_pred = vars_pred.view(
            batch_size, num_subtasks, self.z_dim, self.num_entries
        )  # (batch_size, num_subtasks, z_dim, num_entries)

        self.vars = softplus(self.vars_init[None, None, None, :] + vars_pred)
        # (batch_size, num_subtasks, z_dim, num_entries)

    def get(self, n: int) -> Tensor:
        var_n: Tensor = self.vars[:, :, :, n]
        # (batch_size, num_subtasks, z_dim)

        return var_n


class BCANoiseSchedule(BaseSchedule):
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

        self.noise_mlp = nn.Sequential(
            nn.Linear(2 * h_dim, h_dim),
            getattr(nn, non_linearity)(),
            nn.Linear(h_dim, z_dim * self.num_entries),
        )

        nn.init.constant_(self.noise_mlp[-1].weight, 0)
        nn.init.constant_(self.noise_mlp[-1].bias, 0)

        self.vars_init = torch.linspace(max, min, self.num_entries, device=device).pow(
            2
        )  # (num_entries)

    def update(self, r: Tensor, mask: Tensor | None) -> None:
        # (batch_size, num_subtasks, h_dim)

        batch_size = r[0].shape[0]
        num_subtasks = r[0].shape[1]

        z_mu, z_var = r
        input = torch.cat([z_mu, z_var], dim=-1)
        # (batch_size, num_subtasks, 2 * h_dim)

        vars_pred: Tensor = self.noise_mlp(input)
        # (batch_size, num_subtasks, z_dim * num_entries)

        vars_pred = vars_pred.view(
            batch_size, num_subtasks, self.z_dim, self.num_entries
        )  # (batch_size, num_subtasks, z_dim, num_entries)

        self.vars = softplus(self.vars_init[None, None, None, :] + vars_pred)
        # (batch_size, num_subtasks, z_dim, num_entries)

    def get(self, n: int) -> Tensor:
        var_n: Tensor = self.vars[:, :, :, n]
        # (batch_size, num_subtasks, z_dim)

        return var_n


# class MHANoiseSchedule(BaseSchedule):
#     def __init__(
#         self,
#         z_dim: int,
#         h_dim: int,
#         non_linearity: str,
#         num_steps: int,
#         device: torch.device,
#         num_heads: int,
#         min: float = 0.01,
#         max: float = 1,
#     ) -> None:
#         super(MHANoiseSchedule, self).__init__()

#         self.z_dim = z_dim
#         self.num_entries = num_steps + 1

#         self.proj_in = nn.Linear(1, h_dim)
#         self.cross_attn = nn.MultiheadAttention(h_dim, num_heads, batch_first=True)
#         self.noise_mlp = nn.Sequential(
#             nn.Linear(h_dim, h_dim),
#             getattr(nn, non_linearity)(),
#             nn.Linear(h_dim, z_dim),
#         )

#         nn.init.constant_(self.noise_mlp[-1].weight, 0)
#         nn.init.constant_(self.noise_mlp[-1].bias, 0)

#         self.noise_init = torch.linspace(max, min, self.num_entries, device=device)

#     def update(self, r: Tensor, mask: Tensor | None) -> None:
#         # (batch_size, num_subtasks, context_size, h_dim)
#         # (batch_size, num_subtasks, context_size)

#         batch_size = r.shape[0]
#         num_subtasks = r.shape[1]
#         context_size = r.shape[2]

#         r = r.view(batch_size * num_subtasks, context_size, -1)
#         # (batch_size * num_subtasks, context_size, h_dim)

#         h: Tensor = self.proj_in(self.noise_init.unsqueeze(1))
#         # (num_entries, h_dim)

#         h = h.unsqueeze(0).expand(batch_size * num_subtasks, -1, -1)
#         # (batch_size * num_subtasks, num_entries, h_dim)

#         key_padding_mask = (
#             mask.view(batch_size * num_subtasks, -1).bool().logical_not()
#             if mask is not None
#             else None
#         )  # (batch_size * num_subtasks, context_size)

#         h, _ = self.cross_attn(
#             query=h,
#             key=r,
#             value=r,
#             key_padding_mask=key_padding_mask,
#             need_weights=False,
#         )  # (batch_size * num_subtasks, num_entries, h_dim)

#         h = h.view(batch_size, num_subtasks, self.num_entries, -1)
#         # (batch_size, num_subtasks, num_entries, h_dim)

#         noise = self.noise_mlp(h).transpose(2, 3)
#         # (batch_size, num_subtasks, z_dim, num_entries)

#         self.noise_schedule = nn.functional.softplus(
#             noise + self.noise_init[None, None, None, :]
#         )  # (batch_size, num_subtasks, z_dim, num_entries)

#     def get(self, n: int) -> Tensor:
#         var_n: Tensor = self.noise_schedule[:, :, :, n]
#         # (batch_size, num_subtasks, z_dim)

#         return var_n
