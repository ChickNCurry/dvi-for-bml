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

    def update(self, r: Tensor) -> None:
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

    def update(self, r: Tensor, mask: None) -> None:
        # (batch_size, num_subtasks, h_dim)

        self.increments: Tensor = nn.functional.softplus(
            self.increments_mlp(r) + self.increments_init[None, None, :]
        )

        self.increments = self.increments / torch.sum(
            self.increments, dim=-1, keepdim=True
        )

        self.betas = torch.cumsum(self.increments, dim=-1)

    def get(self, n: int) -> Tensor:
        beta_n = self.betas[:, :, n - 1][:, :, None]
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

    def update(self, r: Tensor, mask: None) -> None:
        # (batch_size, num_subtasks, h_dim)

        z_mu, z_var = r
        z_mu, z_var = self.proj_z_mu(z_mu), self.proj_z_var(z_var)

        self.increments: Tensor = nn.functional.softplus(
            self.increments_mlp(z_mu + z_var) + self.increments_init[None, None, :]
        )

        self.increments = self.increments / torch.sum(
            self.increments, dim=-1, keepdim=True
        )

        self.betas = torch.cumsum(self.increments, dim=-1)

    def get(self, n: int) -> Tensor:
        beta_n = self.betas[:, :, n - 1][:, :, None]
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

        self.cross_attn = nn.MultiheadAttention(h_dim, num_heads, batch_first=True)

        self.increments_mlp = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            getattr(nn, non_linearity)(),
            nn.Linear(h_dim, num_steps),
        )

        nn.init.constant_(self.increments_mlp[-1].weight, 0)
        nn.init.constant_(self.increments_mlp[-1].bias, 0)

        self.increments_init = torch.ones((num_steps), device=device) / num_steps

    def update(self, r: Tensor, mask: None) -> None:
        # (batch_size, num_subtasks, h_dim)

        z_mu, z_var = r
        z_mu, z_var = self.proj_z_mu(z_mu), self.proj_z_var(z_var)

        self.increments: Tensor = nn.functional.softplus(
            self.increments_mlp(z_mu + z_var) + self.increments_init[None, None, :]
        )

        self.increments = self.increments / torch.sum(
            self.increments, dim=-1, keepdim=True
        )

        self.betas = torch.cumsum(self.increments, dim=-1)

    def get(self, n: int) -> Tensor:
        beta_n = self.betas[:, :, n - 1][:, :, None]
        return beta_n
