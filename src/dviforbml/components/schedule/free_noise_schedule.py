import torch
from torch import Tensor, nn
from torch.nn.functional import softplus

from dviforbml.components.schedule.abstract_schedule import AbstractSchedule


class FreeNoiseSchedule(AbstractSchedule):
    def __init__(
        self,
        z_dim: int,
        num_steps: int,
        device: torch.device,
        min: float = 0.01,
        max: float = 1,
    ) -> None:
        super(FreeNoiseSchedule, self).__init__()

        self.num_entries = num_steps + 1

        self.vars = nn.Parameter(
            torch.linspace(max, min, self.num_entries, device=device)
            .unsqueeze(1)
            .repeat(self.num_entries, z_dim)
            .pow(2),
        )  # (num_entries, z_dim)

    def get(self, n: int) -> Tensor:
        var_n = softplus(self.vars[n, :]) + 1e-6
        return var_n


class AggrFreeNoiseSchedule(AbstractSchedule):
    def __init__(
        self,
        z_dim: int,
        h_dim: int,
        num_steps: int,
        num_layers: int,
        non_linearity: str,
        max_context_size: int | None,
        device: torch.device,
        min: float = 0.01,
        max: float = 1,
    ) -> None:
        super(AggrFreeNoiseSchedule, self).__init__()

        self.z_dim = z_dim
        self.num_entries = num_steps + 1

        input_dim = h_dim + (z_dim if max_context_size is not None else 0)

        self.noise_mlp = nn.Sequential(
            nn.Linear(input_dim, h_dim),
            *[
                layer
                for _ in range(num_layers - 1)
                for layer in (getattr(nn, non_linearity)(), nn.Linear(h_dim, h_dim))
            ],
            getattr(nn, non_linearity)(),
            nn.Linear(h_dim, z_dim * self.num_entries),
        )

        nn.init.constant_(self.noise_mlp[-1].weight, 0)
        nn.init.constant_(self.noise_mlp[-1].bias, 0)

        self.vars_init = (
            torch.linspace(max, min, self.num_entries, device=device) ** 2
        )  # (num_entries)

    def update(self, r: Tensor, mask: Tensor | None, s_emb: Tensor | None) -> None:
        # (batch_size, num_subtasks, h_dim)
        # (batch_size, num_subtasks, data_size)
        # (batch_size, num_subtasks, z_dim)

        batch_size = r.shape[0]
        num_subtasks = r.shape[1]

        input = torch.cat([r, s_emb], dim=-1) if s_emb is not None else r
        vars_pred: Tensor = self.noise_mlp(input)
        # (batch_size, num_subtasks, z_dim * num_entries)

        vars_pred = vars_pred.view(
            batch_size, num_subtasks, self.z_dim, self.num_entries
        )  # (batch_size, num_subtasks, z_dim, num_entries)

        self.vars = softplus(self.vars_init[None, None, None, :] + vars_pred) + 1e-6
        # (batch_size, num_subtasks, z_dim, num_entries)

    def get(self, n: int) -> Tensor:
        var_n: Tensor = self.vars[:, :, :, n]
        # (batch_size, num_subtasks, z_dim)

        return var_n


class BCAFreeNoiseSchedule(AbstractSchedule):
    def __init__(
        self,
        z_dim: int,
        h_dim: int,
        num_steps: int,
        num_layers: int,
        non_linearity: str,
        max_context_size: int | None,
        device: torch.device,
        min: float = 0.01,
        max: float = 1,
    ) -> None:
        super(BCAFreeNoiseSchedule, self).__init__()

        self.z_dim = z_dim
        self.num_entries = num_steps + 1

        input_size = 2 * h_dim + (z_dim if max_context_size is not None else 0)

        self.noise_mlp = nn.Sequential(
            nn.Linear(input_size, h_dim),
            *[
                layer
                for _ in range(num_layers - 1)
                for layer in (getattr(nn, non_linearity)(), nn.Linear(h_dim, h_dim))
            ],
            getattr(nn, non_linearity)(),
            nn.Linear(h_dim, z_dim * self.num_entries),
        )

        nn.init.constant_(self.noise_mlp[-1].weight, 0)
        nn.init.constant_(self.noise_mlp[-1].bias, 0)

        self.vars_init = (
            torch.linspace(max, min, self.num_entries, device=device) ** 2
        )  # (num_entries)

    def update(self, r: Tensor, mask: Tensor | None, s_emb: Tensor | None) -> None:
        # (batch_size, num_subtasks, h_dim)
        # (batch_size, num_subtasks, z_dim)

        batch_size = r[0].shape[0]
        num_subtasks = r[0].shape[1]

        z_mu, z_var = r
        input = torch.cat([z_mu, z_var], dim=-1)
        input = torch.cat([input, s_emb], dim=-1) if s_emb is not None else input
        # (batch_size, num_subtasks, 2 * h_dim)

        vars_pred: Tensor = self.noise_mlp(input)
        # (batch_size, num_subtasks, z_dim * num_entries)

        vars_pred = vars_pred.view(
            batch_size, num_subtasks, self.z_dim, self.num_entries
        )  # (batch_size, num_subtasks, z_dim, num_entries)

        self.vars = softplus(self.vars_init[None, None, None, :] + vars_pred) + 1e-6
        # (batch_size, num_subtasks, z_dim, num_entries)

    def get(self, n: int) -> Tensor:
        var_n: Tensor = self.vars[:, :, :, n]
        # (batch_size, num_subtasks, z_dim)

        return var_n


# class MHCAFreeNoiseSchedule(AbstractSchedule):
#     def __init__(
#         self,
#         z_dim: int,
#         h_dim: int,
#         num_steps: int,
#         num_layers: int,
#         non_linearity: str,
#         num_heads: int,
#         max_context_size: int | None,
#         device: torch.device,
#         min: float = 0.01,
#         max: float = 1,
#     ) -> None:
#         super(MHCAFreeNoiseSchedule, self).__init__()

#         self.z_dim = z_dim
#         self.h_dim = h_dim
#         self.num_entries = num_steps + 1

#         self.proj_in = nn.Linear(z_dim, h_dim)
#         self.cross_attn = nn.MultiheadAttention(h_dim, num_heads, batch_first=True)

#         input_size = h_dim * self.num_entries + (
#             z_dim if max_context_size is not None else 0
#         )

#         self.noise_mlp1 = nn.Sequential(
#             nn.Linear(z_dim * self.num_entries, h_dim),
#             *[
#                 layer
#                 for _ in range(num_layers - 1)
#                 for layer in (getattr(nn, non_linearity)(), nn.Linear(h_dim, h_dim))
#             ],
#             getattr(nn, non_linearity)(),
#             nn.Linear(h_dim, z_dim * self.num_entries),
#         )

#         self.noise_mlp = nn.Sequential(
#             nn.Linear(input_size, h_dim),
#             *[
#                 layer
#                 for _ in range(num_layers - 1)
#                 for layer in (getattr(nn, non_linearity)(), nn.Linear(h_dim, h_dim))
#             ],
#             getattr(nn, non_linearity)(),
#             nn.Linear(h_dim, z_dim * self.num_entries),
#         )

#         nn.init.constant_(self.noise_mlp[-1].weight, 0)
#         nn.init.constant_(self.noise_mlp[-1].bias, 0)

#         self.vars_init = (
#             torch.linspace(max, min, self.num_entries, device=device) ** 2
#         )  # (num_entries)

#     def update(self, r: Tensor, mask: Tensor | None, s_emb: Tensor | None) -> None:
#         # (batch_size, num_subtasks, h_dim)
#         # (batch_size, num_subtasks, data_size)
#         # (batch_size, num_subtasks, z_dim)

#         batch_size = r.shape[0]
#         num_subtasks = r.shape[1]
#         data_size = r.shape[2]

#         r = r.view(batch_size * num_subtasks, data_size, -1)
#         # (batch_size * num_subtasks, data_size, h_dim)

#         input = self.vars_init.unsqueeze(1).expand(-1, self.z_dim)
#         input = self.proj_in(input)
#         # (num_entries, h_dim)

#         input = input.unsqueeze(0).expand(batch_size * num_subtasks, -1, -1)
#         # (batch_size * num_subtasks, num_entries, h_dim)

#         key_padding_mask = (
#             mask.view(batch_size * num_subtasks, data_size).bool().logical_not()
#             if mask is not None
#             else None
#         )  # (batch_size * num_subtasks, data_size)

#         input, _ = self.cross_attn(
#             query=input, key=r, value=r, key_padding_mask=key_padding_mask
#         )  # (batch_size * num_subtasks, num_entries, h_dim)

#         input = input.reshape(batch_size, num_subtasks, self.h_dim * self.num_entries)
#         # (batch_size, num_subtasks, h_dim * num_entries)

#         input = torch.cat([input, s_emb], dim=-1) if s_emb is not None else input
#         # (batch_size, num_subtasks, h_dim * num_entries + z_dim)

#         vars_pred: Tensor = self.noise_mlp(input)
#         # (batch_size, num_subtasks, z_dim * num_entries)

#         vars_pred = vars_pred.view(
#             batch_size, num_subtasks, self.z_dim, self.num_entries
#         )  # (batch_size, num_subtasks, z_dim, num_entries)

#         self.vars = softplus(self.vars_init[None, None, None, :] + vars_pred) + 1e-6
#         # (batch_size, num_subtasks, z_dim, num_entries)

#     def get(self, n: int) -> Tensor:
#         var_n: Tensor = self.vars[:, :, :, n]
#         # (batch_size, num_subtasks, z_dim)

#         return var_n
