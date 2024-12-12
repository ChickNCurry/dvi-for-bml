from typing import Tuple

import torch
from torch import Tensor, nn
from torch.distributions import Distribution, Normal

from src.components.dvi.cdvi import CDVI
from src.components.nn.control import Control
from src.components.nn.schedule import CosineSchedule


class DIS(CDVI):
    def __init__(
        self,
        z_dim: int,
        num_steps: int,
        control: Control,
        device: torch.device,
        min: float = 0.1,
    ) -> None:
        super(DIS, self).__init__(
            z_dim=z_dim,
            num_steps=num_steps,
        )

        self.control = control

        self.beta_schedule = CosineSchedule(z_dim, num_steps, device, min)
        # (num_steps, z_dim)

        self.sigma = nn.ParameterList(
            [nn.Parameter(torch.ones((self.z_dim), device=device))]
        )  # (num_steps, z_dim)

    def get_prior(
        self, size: Tuple[int, int, int], device: torch.device
    ) -> Distribution:
        return Normal(  # type: ignore
            torch.zeros(size, device=device),
            torch.ones(size, device=device) * self.sigma[0],
        )

    def forward_kernel(
        self,
        t: int,
        z_prev: Tensor,
        r_aggr: Tensor | None,
        r_non_aggr: Tensor | None,
        mask: Tensor | None,
        prior: Distribution,
        target: Distribution,
    ) -> Distribution:
        # (batch_size, num_subtasks, z_dim)
        # (batch_size, num_subtasks, h_dim)

        beta_t = self.beta_schedule.get(t)
        # (z_dim)

        control_t = self.control(t, z_prev, r_aggr, r_non_aggr, mask)
        # (batch_size, num_subtasks, z_dim)

        z_mu = z_prev + (beta_t * z_prev + control_t) * self.delta_t
        # (batch_size, num_subtasks, z_dim)

        z_sigma = torch.sqrt(2 * beta_t * self.delta_t) * self.sigma[0]
        z_sigma = z_sigma.expand(z_mu.shape[0], z_mu.shape[1], -1)
        # (batch_size, num_subtasks, z_dim)

        return Normal(z_mu, z_sigma)  # type: ignore

    def backward_kernel(
        self,
        t: int,
        z_next: Tensor,
        r_aggr: Tensor | None,
        r_non_aggr: Tensor | None,
        mask: Tensor | None,
        prior: Distribution,
        target: Distribution,
    ) -> Distribution:
        # (batch_size, num_subtasks, z_dim)
        # (batch_size, num_subtasks, h_dim)

        beta_t = self.beta_schedule.get(t)
        # (z_dim)

        z_mu = z_next - (beta_t * z_next) * self.delta_t
        # (batch_size, num_subtasks, z_dim)

        z_sigma = torch.sqrt(2 * beta_t * self.delta_t) * self.sigma[0]
        z_sigma = z_sigma.expand(z_mu.shape[0], z_mu.shape[1], -1)
        # (batch_size, num_subtasks, z_dim)

        return Normal(z_mu, z_sigma)  # type: ignore
