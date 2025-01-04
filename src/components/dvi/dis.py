import torch
from torch import Tensor
from torch.distributions import Distribution, Normal

from src.components.dvi.cdvi import CDVI
from src.components.nn.control import Control
from src.components.nn.schedule import Schedule


class DIS(CDVI):
    def __init__(
        self,
        z_dim: int,
        num_steps: int,
        control: Control,
        step_size_schedule: Schedule,
        noise_schedule: Schedule,
        device: torch.device,
    ) -> None:
        super(DIS, self).__init__(
            z_dim=z_dim,
            num_steps=num_steps,
            device=device,
        )

        self.control = control
        self.step_size_schedule = step_size_schedule
        self.noise_schedule = noise_schedule

    def contextualize(
        self,
        target: Distribution,
        r_aggr: Tensor | None,
        r_non_aggr: Tensor | None,
        mask: Tensor | None,
    ):
        super(DIS, self).contextualize(target, r_aggr, r_non_aggr, mask)

        self.step_size_schedule.update(r_aggr)
        self.noise_schedule.update(r_aggr)

    def forward_kernel(
        self,
        n: int,
        z_prev: Tensor,
    ) -> Distribution:
        # (batch_size, num_subtasks, z_dim)
        # (batch_size, num_subtasks, h_dim)

        delta_t_n = self.step_size_schedule.get(n)
        var_n = self.noise_schedule.get(n)
        control_n = self.control(n, z_prev, self.r_aggr, self.r_non_aggr, self.mask)
        # (batch_size, num_subtasks, z_dim)

        z_mu = z_prev + (var_n * z_prev + torch.sqrt(var_n) * control_n) * delta_t_n
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
        # (batch_size, num_subtasks, z_dim)

        z_mu = z_next - (var_n * z_next) * delta_t_n
        z_sigma = torch.sqrt(var_n * 2 * delta_t_n)
        # (batch_size, num_subtasks, z_dim)

        return Normal(z_mu, z_sigma)  # type: ignore
