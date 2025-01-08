import torch
from torch import Tensor, nn

from src.components.nn.schedule.base_schedule import BaseSchedule


class AnnealingSchedule(BaseSchedule, nn.Module):
    def __init__(
        self,
        num_steps: int,
        device: torch.device,
        requires_grad: bool = True,
    ) -> None:
        super(AnnealingSchedule, self).__init__()

        self.params = nn.Parameter(
            torch.ones((num_steps), device=device),
            requires_grad=requires_grad,
        )

    def update(self, r: Tensor) -> None:
        params = torch.nn.functional.softplus(self.params)
        self.betas = torch.cumsum(params, dim=0) / torch.sum(params, dim=0)

    def get(self, n: int) -> Tensor:
        beta_n = self.betas[n - 1]
        return beta_n


class ContextualAnnealingSchedule(BaseSchedule, nn.Module):
    def __init__(
        self,
        h_dim: int,
        non_linearity: str,
        num_steps: int,
        device: torch.device,
    ) -> None:
        super(ContextualAnnealingSchedule, self).__init__()

        self.num_steps = num_steps

        self.params_mlp = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            getattr(nn, non_linearity)(),
            nn.Linear(h_dim, num_steps),
        )

        nn.init.constant_(self.params_mlp[-1].weight, 0)
        nn.init.constant_(self.params_mlp[-1].bias, 0)

        self.params_init = torch.ones((num_steps), device=device) / num_steps

    def update(self, r: Tensor) -> None:
        self.params: Tensor = nn.functional.softplus(
            self.params_mlp(r) + self.params_init[None, None, :]
        )
        self.params = self.params / torch.sum(self.params, dim=-1, keepdim=True)
        self.betas = torch.cumsum(self.params, dim=-1)

    def get(self, n: int) -> Tensor:
        beta_n = self.betas[:, :, n - 1][:, :, None]
        return beta_n
