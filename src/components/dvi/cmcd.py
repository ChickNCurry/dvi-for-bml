from typing import Tuple

import numpy as np
import torch
from torch import Tensor
from torch.distributions import Distribution, Normal

from src.components.dvi.cdvi import CDVI
from src.components.nn.control import Control
from src.components.nn.schedule import CosineSchedule, LinearSchedule


class CMCD(CDVI):
    def __init__(
        self,
        z_dim: int,
        num_steps: int,
        control: Control | None,
        device: torch.device,
        min_sigma: float = 0.2,
        min_annealing: float = 0.01,
    ) -> None:
        super(CMCD, self).__init__(
            z_dim=z_dim,
            num_steps=num_steps,
        )

        self.control = control

        self.sigma_schedule = CosineSchedule(z_dim, num_steps, device, min_sigma)
        self.annealing_schedule = LinearSchedule(
            z_dim, num_steps, device, min_annealing
        )
        # (num_steps, z_dim)

    def get_prior(
        self, size: Tuple[int, int, int], device: torch.device
    ) -> Distribution:
        return Normal(  # type: ignore
            torch.zeros(size, device=device),
            torch.ones(size, device=device),
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

        sigma_t = self.sigma_schedule.get(t)
        # (1)

        control_t = (
            self.control(t, z_prev, r_aggr, r_non_aggr, mask)
            if self.control is not None
            else 0
        )
        # (batch_size, num_subtasks, z_dim)

        grad_log = self.get_grad_log_geo_avg(t, z_prev, prior, target)
        # (batch_size, num_subtasks, z_dim)

        z_mu = z_prev + (sigma_t.pow(2) * grad_log + control_t) * self.delta_t
        # (batch_size, num_subtasks, z_dim)

        z_sigma = sigma_t * np.sqrt(self.delta_t)
        z_sigma = z_sigma.expand(z_mu.shape[0], z_mu.shape[1], -1)
        # (batch_size, num_subtasks, z_dim)

        z_mu = torch.nan_to_num(z_mu, nan=0.0)
        z_sigma = torch.nan_to_num(z_sigma, nan=0.0)

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

        sigma_t = self.sigma_schedule.get(t)
        # (1)

        control_t = (
            self.control(t, z_next, r_aggr, r_non_aggr, mask)
            if self.control is not None
            else 0
        )
        # (batch_size, num_subtasks, z_dim)

        grad_log = self.get_grad_log_geo_avg(t, z_next, prior, target)
        # (batch_size, num_subtasks, z_dim)

        z_mu = z_next + (sigma_t.pow(2) * grad_log - control_t) * self.delta_t
        # (batch_size, num_subtasks, z_dim)

        z_sigma = sigma_t * np.sqrt(self.delta_t)
        z_sigma = z_sigma.expand(z_mu.shape[0], z_mu.shape[1], -1)
        # (batch_size, num_subtasks, z_dim)

        z_mu = torch.nan_to_num(z_mu, nan=0.0)
        z_sigma = torch.nan_to_num(z_sigma, nan=0.0)

        return Normal(z_mu, z_sigma)  # type: ignore

    def get_grad_log_geo_avg(
        self,
        t: int,
        z: Tensor,
        prior: Distribution,
        p_z_T: Distribution,
    ) -> Tensor:

        z = z.requires_grad_(True)

        beta_t = self.annealing_schedule.get(t)

        log_geo_avg: Tensor = (1 - beta_t) * prior.log_prob(
            z
        ) + beta_t * p_z_T.log_prob(z)

        grad = torch.autograd.grad(
            outputs=log_geo_avg,
            inputs=z,
            grad_outputs=torch.ones_like(log_geo_avg),
            create_graph=True,
            retain_graph=True,
        )[0]

        return grad
