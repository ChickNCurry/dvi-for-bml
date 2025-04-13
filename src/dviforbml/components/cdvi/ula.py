from typing import Tuple

import torch
from torch import Tensor
from torch.distributions import Distribution, Normal

from dviforbml.components.cdvi.cdvi import CDVI
from dviforbml.components.schedule.abstract_schedule import AbstractSchedule


class ULA(CDVI):
    def __init__(
        self,
        z_dim: int,
        num_steps: int,
        step_size_schedule: AbstractSchedule,
        noise_schedule: AbstractSchedule,
        annealing_schedule: AbstractSchedule,
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
        r: Tensor | Tuple[Tensor, Tensor],
        mask: Tensor | None,
        s_emb: Tensor | None,
    ) -> None:
        super(ULA, self).contextualize(target, r, mask, s_emb)

        self.step_size_schedule.update(r, mask, s_emb)
        self.noise_schedule.update(r, mask, s_emb)
        self.annealing_schedule.update(r, mask, s_emb)

    def forward_kernel(self, n: int, z: Tensor) -> Distribution:
        # (batch_size, num_subtasks, z_dim)
        # (batch_size, num_subtasks, h_dim)

        delta_t_n = self.step_size_schedule.get(n)
        var_n = self.noise_schedule.get(n)
        score_n = self.compute_score(n, z)
        # (batch_size, num_subtasks, z_dim)

        # # check for nans ins score
        # if torch.any(torch.isnan(score_n)):
        #     raise ValueError("Nan detected in score")

        # if torch.any(var_n < 0):
        #     raise ValueError("Negative variance detected")

        z_mu = z + (var_n * score_n) * delta_t_n
        z_sigma = torch.sqrt(2 * var_n * delta_t_n)
        # (batch_size, num_subtasks, z_dim)

        return Normal(z_mu, z_sigma)

    def backward_kernel(self, n: int, z: Tensor) -> Distribution:
        # (batch_size, num_subtasks, z_dim)
        # (batch_size, num_subtasks, h_dim)

        delta_t_n = self.step_size_schedule.get(n)
        var_n = self.noise_schedule.get(n)
        score_n = self.compute_score(n, z)
        # (batch_size, num_subtasks, z_dim)

        # if torch.any(torch.isnan(score_n)):
        #     raise ValueError("Nan detected in score")

        # if torch.any(var_n < 0):
        #     raise ValueError("Negative variance detected")

        z_mu = z - (var_n * score_n) * delta_t_n
        z_sigma = torch.sqrt(2 * var_n * delta_t_n)
        # (batch_size, num_subtasks, z_dim)

        return Normal(z_mu, z_sigma)

    def compute_score(self, n: int, z: Tensor) -> Tensor:
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

        return score_n
