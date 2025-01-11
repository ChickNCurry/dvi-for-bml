import torch
from torch import Tensor, nn

from src.components.schedule.base_schedule import BaseSchedule


class AnnealingSchedule(BaseSchedule, nn.Module):
    def __init__(
        self,
        num_steps: int,
        device: torch.device,
        requires_grad: bool = True,
    ) -> None:
        super(AnnealingSchedule, self).__init__()

        self.increments = nn.Parameter(
            torch.ones((num_steps), device=device),
            requires_grad=requires_grad,
        )

    def update(self, r: Tensor, mask: Tensor | None) -> None:
        params = torch.nn.functional.softplus(self.increments)
        self.betas = torch.cumsum(params, dim=0) / torch.sum(params, dim=0)

    def get(self, n: int) -> Tensor:
        beta_n = self.betas[n - 1]
        return beta_n


class AggrAnnealingSchedule(BaseSchedule, nn.Module):
    def __init__(
        self,
        h_dim: int,
        non_linearity: str,
        num_steps: int,
        device: torch.device,
    ) -> None:
        super(AggrAnnealingSchedule, self).__init__()

        self.num_steps = num_steps

        self.increments_mlp = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            getattr(nn, non_linearity)(),
            nn.Linear(h_dim, num_steps),
        )

        nn.init.constant_(self.increments_mlp[-1].weight, 0)
        nn.init.constant_(self.increments_mlp[-1].bias, 0)

        self.increments_init = torch.ones((num_steps), device=device) / num_steps

    def update(self, r: Tensor, mask: Tensor | None) -> None:
        # (batch_size, num_subtasks, h_dim)

        self.increments: Tensor = nn.functional.softplus(
            self.increments_mlp(r) + self.increments_init[None, None, :]
        )  # (batch_size, num_subtasks, num_steps)

        self.increments = self.increments / torch.sum(
            self.increments, dim=-1, keepdim=True
        )  # (batch_size, num_subtasks, num_steps)

        self.betas = torch.cumsum(self.increments, dim=-1)
        # (batch_size, num_subtasks, num_steps)

    def get(self, n: int) -> Tensor:
        beta_n = self.betas[:, :, n - 1][:, :, None]
        # (batch_size, num_subtasks, 1)

        return beta_n


class BCAAnnealingSchedule(BaseSchedule, nn.Module):
    def __init__(
        self,
        h_dim: int,
        non_linearity: str,
        num_steps: int,
        device: torch.device,
    ) -> None:
        super(BCAAnnealingSchedule, self).__init__()

        self.num_steps = num_steps

        self.proj_z_mu = nn.Linear(h_dim, h_dim)
        self.proj_z_var = nn.Linear(h_dim, h_dim)

        self.increments_mlp = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            getattr(nn, non_linearity)(),
            nn.Linear(h_dim, num_steps),
        )

        nn.init.constant_(self.increments_mlp[-1].weight, 0)
        nn.init.constant_(self.increments_mlp[-1].bias, 0)

        self.increments_init = torch.ones((num_steps), device=device) / num_steps

    def update(self, r: Tensor, mask: Tensor | None) -> None:
        # (batch_size, num_subtasks, h_dim)

        z_mu, z_var = r
        z_mu, z_var = self.proj_z_mu(z_mu), self.proj_z_var(z_var)
        # (batch_size, num_subtasks, h_dim)

        self.increments: Tensor = nn.functional.softplus(
            self.increments_mlp(z_mu + z_var) + self.increments_init[None, None, :]
        )  # (batch_size, num_subtasks, num_steps)

        self.increments = self.increments / torch.sum(
            self.increments, dim=-1, keepdim=True
        )  # (batch_size, num_subtasks, num_steps)

        self.betas = torch.cumsum(self.increments, dim=-1)
        # (batch_size, num_subtasks, num_steps)

    def get(self, n: int) -> Tensor:
        beta_n = self.betas[:, :, n - 1][:, :, None]
        # (batch_size, num_subtasks, 1)

        return beta_n


class MHAAnnealingSchedule(BaseSchedule, nn.Module):
    def __init__(
        self,
        h_dim: int,
        non_linearity: str,
        num_steps: int,
        device: torch.device,
        num_heads: int,
    ) -> None:
        super(MHAAnnealingSchedule, self).__init__()

        self.num_steps = num_steps

        self.proj_in = nn.Linear(1, h_dim)
        self.cross_attn = nn.MultiheadAttention(h_dim, num_heads, batch_first=True)
        self.increments_mlp = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            getattr(nn, non_linearity)(),
            nn.Linear(h_dim, 1),
        )

        nn.init.constant_(self.increments_mlp[-1].weight, 0)
        nn.init.constant_(self.increments_mlp[-1].bias, 0)

        self.increments_init = torch.ones((num_steps), device=device) / num_steps

    def update(self, r: Tensor, mask: Tensor | None) -> None:
        # (batch_size, num_subtasks, context_size, h_dim)
        # (batch_size, num_subtasks, context_size)

        batch_size = r.shape[0]
        num_subtasks = r.shape[1]
        context_size = r.shape[2]

        r = r.view(batch_size * num_subtasks, context_size, -1)
        # (batch_size * num_subtasks, context_size, h_dim)

        h: Tensor = self.proj_in(self.increments_init.unsqueeze(1))
        # (num_steps, h_dim)

        h = h.unsqueeze(0).expand(batch_size * num_subtasks, -1, -1)
        # (batch_size * num_subtasks, num_steps, h_dim)

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
        )  # (batch_size * num_subtasks, num_steps, h_dim)

        h = h.view(batch_size, num_subtasks, self.num_steps, -1)
        # (batch_size, num_subtasks, num_steps, h_dim)

        increments = self.increments_mlp(h).squeeze(-1)
        # (batch_size, num_subtasks, num_steps)

        self.increments: Tensor = nn.functional.softplus(
            increments + self.increments_init[None, None, :]
        )  # (batch_size, num_subtasks, num_steps)

        self.increments = self.increments / torch.sum(
            self.increments, dim=-1, keepdim=True
        )  # (batch_size, num_subtasks, num_steps)

        self.betas = torch.cumsum(self.increments, dim=-1)
        # (batch_size, num_subtasks, num_steps)

    def get(self, n: int) -> Tensor:
        beta_n = self.betas[:, :, n - 1][:, :, None]
        # (batch_size, num_subtasks, 1)

        return beta_n
