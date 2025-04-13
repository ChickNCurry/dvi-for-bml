from typing import Tuple

import torch
from torch import Tensor
from torch.distributions import Distribution, Normal

from dviforbml.components.cdvi.cdvi import CDVI
from dviforbml.components.control.abstract_control import AbstractControl
from dviforbml.components.schedule.abstract_schedule import AbstractSchedule


class DIS(CDVI):
    def __init__(
        self,
        z_dim: int,
        num_steps: int,
        control: AbstractControl,
        step_size_schedule: AbstractSchedule,
        noise_schedule: AbstractSchedule,
        annealing_schedule: AbstractSchedule | None,
        use_score: bool,
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

    def contextualize(
        self,
        target: Distribution,
        r: Tensor | Tuple[Tensor, Tensor],
        mask: Tensor | None,
        s_emb: Tensor | None,
    ):
        super(DIS, self).contextualize(target, r, mask, s_emb)

        self.step_size_schedule.update(r, mask, s_emb)
        self.noise_schedule.update(r, mask, s_emb)

        if self.annealing_schedule is not None:
            self.annealing_schedule.update(r, mask, s_emb)

    def forward_kernel(self, n: int, z: Tensor) -> Distribution:
        # (batch_size, num_subtasks, z_dim)
        # (batch_size, num_subtasks, h_dim)

        delta_t_n = self.step_size_schedule.get(n)
        var_n = self.noise_schedule.get(n)

        # if torch.any(var_n < 0):
        #     raise ValueError("Negative variance detected")

        score_n = (
            None
            if not self.use_score
            else (torch.sqrt(var_n) * self.compute_score(n, z))
        )
        control_n = self.control(n, z, self.r, self.mask, self.s_emb, score_n)
        # (batch_size, num_subtasks, z_dim)

        # NO NUMERICAL TRICK
        z_mu = z + (var_n * z + torch.sqrt(var_n) * control_n) * delta_t_n
        z_sigma = torch.sqrt(2 * var_n * delta_t_n)
        # (batch_size, num_subtasks, z_dim)

        return Normal(z_mu, z_sigma)

    def backward_kernel(self, n: int, z: Tensor) -> Distribution:
        # (batch_size, num_subtasks, z_dim)
        # (batch_size, num_subtasks, h_dim)

        delta_t_n = self.step_size_schedule.get(n)
        var_n = self.noise_schedule.get(n)
        # (batch_size, num_subtasks, z_dim)

        # if torch.any(var_n < 0):
        #     raise ValueError("Negative variance detected")

        z_mu = z - (var_n * z) * delta_t_n
        z_sigma = torch.sqrt(2 * var_n * delta_t_n)
        # (batch_size, num_subtasks, z_dim)

        return Normal(z_mu, z_sigma)

    def compute_score(self, n: int, z: Tensor) -> Tensor:
        assert self.annealing_schedule is not None

        z = z.requires_grad_(True)

        beta_n = self.annealing_schedule.get(n)

        prior_log_prob = self.prior.log_prob(z)
        target_log_prob = self.target.log_prob(z)
        log_geo_avg = (1 - beta_n) * prior_log_prob + beta_n * target_log_prob

        score_n = torch.autograd.grad(
            outputs=log_geo_avg,
            inputs=z,
            grad_outputs=torch.ones_like(log_geo_avg),
            create_graph=True,
            retain_graph=True,
        )[0]

        score_n = score_n.detach()

        thresh = 10
        grad_norm = score_n.norm(p=2)
        if grad_norm > thresh:
            score_n = score_n * (thresh / grad_norm)

        return score_n
