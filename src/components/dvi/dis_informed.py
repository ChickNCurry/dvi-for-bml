from typing import Tuple

import torch
from torch import Tensor, nn
from torch.distributions import Distribution, Normal

from src.components.dvi.cdvi import CDVI
from src.components.nn.control_informed import InformedControl
from src.components.nn.schedule import CosineSchedule


class InformedDIS(CDVI):
    def __init__(
        self,
        z_dim: int,
        num_steps: int,
        control: InformedControl,
        device: torch.device,
        min: float = 0.1,
    ) -> None:
        super(InformedDIS, self).__init__(
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

        z_sigma = torch.sqrt(2 * beta_t * self.delta_t) * self.sigma[0]
        # (batch_size, num_subtasks, z_dim)

        grad_log = self.get_grad_log_geo_avg(t, z_prev, prior, target).detach()
        # (batch_size, num_subtasks, z_dim)

        control_t = self.control(t, z_prev, r_aggr, r_non_aggr, mask, grad_log, z_sigma)
        # (batch_size, num_subtasks, z_dim)

        z_mu = z_prev + (beta_t * z_prev + control_t) * self.delta_t
        # (batch_size, num_subtasks, z_dim)

        z_sigma = z_sigma.expand(z_mu.shape[0], z_mu.shape[1], -1)
        # (batch_size, num_subtasks, z_dim)

        # z_mu = torch.nan_to_num(z_mu)
        # z_sigma = torch.nan_to_num(z_sigma)

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

        # z_mu = torch.nan_to_num(z_mu)
        # z_sigma = torch.nan_to_num(z_sigma)

        return Normal(z_mu, z_sigma)  # type: ignore

    def get_grad_log_geo_avg(
        self,
        t: int,
        z: Tensor,
        prior: Distribution,
        target: Distribution,
    ) -> Tensor:

        z = z.requires_grad_(True)

        log_geo_avg: Tensor = (t / self.num_steps) * prior.log_prob(z) + (
            1 - (t / self.num_steps)
        ) * target.log_prob(z)

        grad = torch.autograd.grad(
            outputs=log_geo_avg,
            inputs=z,
            grad_outputs=torch.ones_like(log_geo_avg),
        )[0]

        grad_norm = grad.norm(p=2)  # Compute the L2 norm of the gradient
        if grad_norm > 1:
            grad = grad * (1 / grad_norm)

        return grad