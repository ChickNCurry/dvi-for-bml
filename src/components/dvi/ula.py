import torch
from torch import Tensor
from torch.distributions import Distribution, Normal

from src.components.dvi.cdvi import CDVI
from src.components.nn.control import Control
from src.components.nn.schedule import (
    AnnealingSchedule,
    NoiseSchedule,
    StepSizeSchedule,
)


class ULA(CDVI):
    def __init__(
        self,
        z_dim: int,
        num_steps: int,
        step_size_schedule: StepSizeSchedule,
        noise_schedule: NoiseSchedule,
        annealing_schedule: AnnealingSchedule,
        device: torch.device,
    ) -> None:
        super(ULA, self).__init__(
            z_dim=z_dim,
            num_steps=num_steps,
            device=device,
        )

        self.step_size_schedule = step_size_schedule
        self.noise_schedule = noise_schedule
        self.annealing_schedule = annealing_schedule

    def contextualize(
        self,
        target: Distribution,
        r_aggr: Tensor | None,
        r_non_aggr: Tensor | None,
        mask: Tensor | None,
    ) -> None:
        super(ULA, self).contextualize(target, r_aggr, r_non_aggr, mask)

        self.step_size_schedule.update(r_aggr)
        self.noise_schedule.update(r_aggr)
        self.annealing_schedule.update(r_aggr)

    def forward_kernel(
        self,
        n: int,
        z_prev: Tensor,
    ) -> Distribution:
        # (batch_size, num_subtasks, z_dim)
        # (batch_size, num_subtasks, h_dim)

        delta_t_n = self.step_size_schedule.get(n)
        var_n = self.noise_schedule.get(n)
        grad_log = self.get_grad_log_geo_avg(n, z_prev)
        # (batch_size, num_subtasks, z_dim)

        z_mu = z_prev + (var_n * grad_log) * delta_t_n
        z_sigma = torch.sqrt(var_n * 2 * delta_t_n)
        # (batch_size, num_subtasks, z_dim)

        return Normal(z_mu, z_sigma)  # type: ignore

    def backward_kernel(
        self,
        n: int,
        z_next: Tensor,
    ) -> Distribution:
        # (batch_size, num_subtasks, z_dim)
        # (batch_size, num_subtasks, h_dim)

        delta_t_n = self.step_size_schedule.get(n)
        var_n = self.noise_schedule.get(n)
        grad_log = self.get_grad_log_geo_avg(n, z_next)
        # (batch_size, num_subtasks, z_dim)

        z_mu = z_next + (var_n * grad_log) * delta_t_n
        z_sigma = torch.sqrt(2 * var_n * delta_t_n)
        # (batch_size, num_subtasks, z_dim)

        return Normal(z_mu, z_sigma)  # type: ignore

    def get_grad_log_geo_avg(
        self,
        n: int,
        z: Tensor,
    ) -> Tensor:

        z = z.requires_grad_(True)

        beta_n = self.annealing_schedule.get(n)

        log_geo_avg = (1 - beta_n) * self.prior.log_prob(
            z
        ) + beta_n * self.target.log_prob(z)

        grad = torch.autograd.grad(
            outputs=log_geo_avg,
            inputs=z,
            grad_outputs=torch.ones_like(log_geo_avg),
            create_graph=True,
            retain_graph=True,
        )[0]

        grad = torch.nan_to_num(grad)

        return grad
