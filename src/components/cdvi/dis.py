from typing import Tuple

import torch
from torch import Tensor
from torch.distributions import Distribution, Normal

from src.components.cdvi.cdvi import CDVI
from src.components.control.abstract_control import AbstractControl
from src.components.schedule.abstract_schedule import AbstractSchedule


class DIS(CDVI):
    def __init__(
        self,
        z_dim: int,
        num_steps: int,
        control: AbstractControl,
        step_size_schedule: AbstractSchedule,
        noise_schedule: AbstractSchedule,
        annealing_schedule: AbstractSchedule,
        use_score: bool,
        use_error: bool,
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
        self.annealing_schedule = annealing_schedule
        self.use_score = use_score
        self.use_error = use_error

    def contextualize(
        self,
        target: Distribution,
        r: Tensor | Tuple[Tensor, Tensor],
        mask: Tensor | None,
    ):
        super(DIS, self).contextualize(target, r, mask)

        self.step_size_schedule.update(r, mask)
        self.noise_schedule.update(r, mask)
        self.annealing_schedule.update(r, mask)

    def forward_kernel(self, n: int, z: Tensor) -> Distribution:
        # (batch_size, num_subtasks, z_dim)
        # (batch_size, num_subtasks, h_dim)

        delta_t_n = self.step_size_schedule.get(n)
        var_n = self.noise_schedule.get(n)

        score_n = (
            torch.sqrt(var_n) * self.compute_score(n, z) if self.use_score else None
        )
        error_n = self.target.log_prob(z).detach() if self.use_error else None

        control_n = self.control(n, z, self.r, self.mask, score_n, error_n)
        # (batch_size, num_subtasks, z_dim)

        z_mu = z + (-var_n * z + torch.sqrt(var_n) * control_n) * delta_t_n
        z_sigma = torch.sqrt(var_n * 2 * delta_t_n)
        # (batch_size, num_subtasks, z_dim)

        return Normal(z_mu, z_sigma)  # type: ignore

    def backward_kernel(self, n: int, z: Tensor) -> Distribution:
        # (batch_size, num_subtasks, z_dim)
        # (batch_size, num_subtasks, h_dim)

        delta_t_n = self.step_size_schedule.get(n)
        var_n = self.noise_schedule.get(n)
        # (batch_size, num_subtasks, z_dim)

        z_mu = z - (var_n * z) * delta_t_n
        z_sigma = torch.sqrt(var_n * 2 * delta_t_n)
        # (batch_size, num_subtasks, z_dim)

        return Normal(z_mu, z_sigma)  # type: ignore

    def compute_score(self, n: int, z: Tensor) -> Tensor:
        z = z.requires_grad_(True)

        beta_n = self.annealing_schedule.get(n)

        log_geo_avg = (1 - beta_n) * self.prior.log_prob(
            z
        ) + beta_n * self.target.log_prob(z)

        score_n = torch.autograd.grad(
            outputs=log_geo_avg,
            inputs=z,
            grad_outputs=torch.ones_like(log_geo_avg),
            create_graph=True,
            retain_graph=True,
        )[0]

        score_n = score_n.detach()
        # score_n = torch.nan_to_num(score_n)

        grad_norm = score_n.norm(p=2)
        if grad_norm > 1:
            score_n = score_n * (1 / grad_norm)

        return score_n
