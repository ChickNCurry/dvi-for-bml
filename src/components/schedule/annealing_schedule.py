import torch
from torch import Tensor, nn
from torch.nn.functional import softplus

from src.components.schedule.abstract_schedule import AbstractSchedule


class AnnealingSchedule(AbstractSchedule, nn.Module):
    def __init__(
        self,
        num_steps: int,
        device: torch.device,
        requires_grad: bool = True,
    ) -> None:
        super(AnnealingSchedule, self).__init__()

        self.num_entries = num_steps + 1

        self.increments = nn.Parameter(
            torch.ones((self.num_entries), device=device),
            requires_grad=requires_grad,
        )  # (num_entries)

    def update(self, r: Tensor, mask: Tensor | None) -> None:
        params = softplus(self.increments)
        self.betas = torch.cumsum(params, dim=0) / torch.sum(params, dim=0)
        # (num_entries)

    def get(self, n: int) -> Tensor:
        beta_n = self.betas[n]
        return beta_n


class AggrAnnealingSchedule(AbstractSchedule, nn.Module):
    def __init__(
        self,
        h_dim: int,
        non_linearity: str,
        num_steps: int,
        device: torch.device,
    ) -> None:
        super(AggrAnnealingSchedule, self).__init__()

        self.num_entries = num_steps + 1

        self.increments_mlp = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            getattr(nn, non_linearity)(),
            nn.Linear(h_dim, self.num_entries),
        )

        nn.init.constant_(self.increments_mlp[-1].weight, 0)
        nn.init.constant_(self.increments_mlp[-1].bias, 0)

        self.increments_init = (
            torch.ones((self.num_entries), device=device) / self.num_entries
        )  # (num_entries)

    def update(self, r: Tensor, mask: Tensor | None) -> None:
        # (batch_size, num_subtasks, h_dim)

        increments: Tensor = softplus(
            self.increments_mlp(r) + self.increments_init[None, None, :]
        )  # (batch_size, num_subtasks, num_entries)

        increments = increments / torch.sum(increments, dim=-1, keepdim=True)
        # (batch_size, num_subtasks, num_entries)

        self.betas = torch.cumsum(increments, dim=-1)
        # (batch_size, num_subtasks, num_entries)

    def get(self, n: int) -> Tensor:
        beta_n = self.betas[:, :, n][:, :, None]
        # (batch_size, num_subtasks, 1)

        return beta_n


class BCAAnnealingSchedule(AbstractSchedule, nn.Module):
    def __init__(
        self,
        h_dim: int,
        non_linearity: str,
        num_steps: int,
        device: torch.device,
    ) -> None:
        super(BCAAnnealingSchedule, self).__init__()

        self.num_entries = num_steps + 1

        self.increments_mlp = nn.Sequential(
            nn.Linear(2 * h_dim, h_dim),
            getattr(nn, non_linearity)(),
            nn.Linear(h_dim, self.num_entries),
        )

        nn.init.constant_(self.increments_mlp[-1].weight, 0)
        nn.init.constant_(self.increments_mlp[-1].bias, 0)

        self.increments_init = (
            torch.ones((self.num_entries), device=device) / self.num_entries
        )

    def update(self, r: Tensor, mask: Tensor | None) -> None:
        # (batch_size, num_subtasks, h_dim)

        z_mu, z_var = r
        input = torch.cat([z_mu, z_var], dim=-1)
        # (batch_size, num_subtasks, 2 * h_dim)

        increments: Tensor = softplus(
            self.increments_mlp(input) + self.increments_init[None, None, :]
        )  # (batch_size, num_subtasks, num_entries)

        increments = increments / torch.sum(increments, dim=-1, keepdim=True)
        # (batch_size, num_subtasks, num_entries)

        self.betas = torch.cumsum(increments, dim=-1)
        # (batch_size, num_subtasks, num_entries)

    def get(self, n: int) -> Tensor:
        beta_n = self.betas[:, :, n][:, :, None]
        # (batch_size, num_subtasks, 1)

        return beta_n


# class MHAAnnealingSchedule(BaseSchedule, nn.Module):
#     def __init__(
#         self,
#         h_dim: int,
#         non_linearity: str,
#         num_steps: int,
#         device: torch.device,
#         num_heads: int,
#     ) -> None:
#         super(MHAAnnealingSchedule, self).__init__()

#         self.num_entries = num_steps + 1

#         self.proj_in = nn.Linear(1, h_dim)
#         self.cross_attn = nn.MultiheadAttention(h_dim, num_heads, batch_first=True)
#         self.increments_mlp = nn.Sequential(
#             nn.Linear(h_dim, h_dim),
#             getattr(nn, non_linearity)(),
#             nn.Linear(h_dim, 1),
#         )

#         nn.init.constant_(self.increments_mlp[-1].weight, 0)
#         nn.init.constant_(self.increments_mlp[-1].bias, 0)

#         self.increments_init = (
#             torch.ones((self.num_entries), device=device) / self.num_entries
#         )

#     def update(self, r: Tensor, mask: Tensor | None) -> None:
#         # (batch_size, num_subtasks, context_size, h_dim)
#         # (batch_size, num_subtasks, context_size)

#         batch_size = r.shape[0]
#         num_subtasks = r.shape[1]
#         context_size = r.shape[2]

#         r = r.view(batch_size * num_subtasks, context_size, -1)
#         # (batch_size * num_subtasks, context_size, h_dim)

#         h: Tensor = self.proj_in(self.increments_init.unsqueeze(1))
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

#         increments = self.increments_mlp(h).squeeze(-1)
#         # (batch_size, num_subtasks, num_entries)

#         self.increments: Tensor = nn.functional.softplus(
#             increments + self.increments_init[None, None, :]
#         )  # (batch_size, num_subtasks, num_entries)

#         self.increments = self.increments / torch.sum(
#             self.increments, dim=-1, keepdim=True
#         )  # (batch_size, num_subtasks, num_entries)

#         self.betas = torch.cumsum(self.increments, dim=-1)
#         # (batch_size, num_subtasks, num_entries)

#     def get(self, n: int) -> Tensor:
#         beta_n = self.betas[:, :, n][:, :, None]
#         # (batch_size, num_subtasks, 1)

#         return beta_n
