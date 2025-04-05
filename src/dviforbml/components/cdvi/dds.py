from typing import Tuple

import torch
from torch import Tensor
from torch.distributions import Distribution, Normal
from torch import nn

from dviforbml.components.cdvi.cdvi import CDVI
from dviforbml.components.control.abstract_control import AbstractControl


def create_cos_square_scheduler(
    C_start: Tensor, C_end: Tensor, T: float, device: torch.device
) -> Tuple:
    """
    Create a cosine-squared scheduler.

    Args:
        C_start (float): Starting diffusion coefficient
        C_end (float): Ending diffusion coefficient
        T (float): Total time duration

    Returns:
        Tuple of three functions:
        1. Scheduler function that returns the diffusion coefficient at time t
        2. Forward integrated scheduler function that returns \int_0^s scheduler(t) dt
        3. Backward integrated scheduler function that returns \int_s^T scheduler(t) dt
    """
    a = C_start - C_end
    b = torch.tensor([3.14159265358979323846], device=device) / (2 * T)
    c = C_end

    def scheduler(t: Tensor) -> Tensor:
        """
        Cosine-squared scheduler that decreases smoothly from C_start to C_end
        Follows the shape: C_end + (C_start - C_end) * cos(π * t / T)^2
        """
        return a * (torch.cos(b * t) ** 2) + c

    def forward_integrated_scheduler(t: Tensor) -> Tensor:
        """
        Compute the forward integral of the scheduler from 0 to t.
        This is the analytical integral of the linear scheduler from 0 to t.
        """
        return (a * torch.sin(2 * b * t) / (4 * b)) + (a * t / 2) + (c * t)

    def delta_integrated_scheduler(t_prev: Tensor, t: Tensor) -> Tensor:
        """
        Compute the integral of the scheduler from t_prev to t.
        This is the analytical integral of the cosine-squared scheduler from t_prev to t.
        """
        # We can use the forward integrated scheduler and take the difference
        forward_t = (a * torch.sin(2 * b * t) / (4 * b)) + (a * t / 2) + (c * t)
        forward_t_prev = (
            (a * torch.sin(2 * b * t_prev) / (4 * b)) + (a * t_prev / 2) + (c * t_prev)
        )
        return forward_t - forward_t_prev

    def backward_integrated_scheduler(s: Tensor) -> Tensor:
        """
        Compute the backward integral of the scheduler from s to T.
        This is the analytical integral of the linear scheduler from s to T.
        """
        return (
            -(a * (torch.sin(2 * b * s)) / (4 * b)) + (a * (T - s) / 2) + (c * (T - s))
        )

    return (
        scheduler,
        forward_integrated_scheduler,
        delta_integrated_scheduler,
        backward_integrated_scheduler,
    )


class DDS(CDVI, nn.Module):
    def __init__(
        self,
        z_dim: int,
        num_steps: int,
        control: AbstractControl,
        device: torch.device,
        meta_batch_size: int,
        num_subtasks: int,
        init_std: float = 1,
        T: float = 1.0,
        beta_max: float = 10.0,
        beta_min: float = 0.01,
    ) -> None:
        super(DDS, self).__init__(
            z_dim=z_dim,
            num_steps=num_steps,
            device=device,
        )
        self.device = device
        self.dim = z_dim
        self.control = control
        self.meta_batch_size = meta_batch_size
        self.num_subtasks = num_subtasks
        self.batch_size = meta_batch_size * num_subtasks

        self.eta = init_std
        self.T = T
        self.dt = T / num_steps

        self.step_tensor = (
            torch.Tensor([step for step in range(self.num_steps + 1)])
            .unsqueeze(1)
            .to(device)
        )

        self.prior = Normal(
            torch.zeros(self.dim).to(self.device),
            torch.ones(self.dim).to(self.device) * self.eta,
        )

        self.prior_score = lambda x: -x / self.eta**2

        self.beta_max = nn.Parameter(torch.tensor(beta_max, device=self.device))
        self.beta_min = nn.Parameter(torch.tensor(beta_min, device=self.device))

    def update_schedules(self):
        # print(self.beta_max.data, self.beta_min.data)

        (
            scheduler,
            forward_integrated_scheduler,
            delta_integrated_scheduler,
            self.backward_integrated_scheduler,
        ) = create_cos_square_scheduler(
            C_start=torch.nn.functional.softplus(self.beta_max),
            C_end=torch.nn.functional.softplus(self.beta_min),
            T=self.T,
            device=self.device,
        )

        self.noise_bwd_int = torch.exp(
            -torch.Tensor(
                [
                    self.backward_integrated_scheduler(self.dt * step)
                    for step in range(self.num_steps)
                ]
            ).to(self.device)
        )

        self.alpha_vals = 1 - torch.exp(
            -2
            * torch.Tensor(
                [
                    delta_integrated_scheduler(step * self.dt, (step + 1) * self.dt)
                    for step in range(self.num_steps + 1)
                ]
            )
        ).to(self.device)

        self.kappa_vals = torch.Tensor(
            [
                (self.eta * (1 - torch.sqrt(1 - self.alpha_vals[step]))) ** 2
                / self.alpha_vals[step]
                for step in range(self.num_steps + 1)
            ]
        ).to(self.device)

    def sde_integrate(
        self,
        target: Distribution,
        r: Tensor | Tuple[Tensor, Tensor],
        mask: Tensor | None,
        size: Tensor | None,
    ):
        self.meta_batch_size = r.shape[0] if not isinstance(r, tuple) else r[0].shape[0]
        self.num_subtasks = r.shape[1] if not isinstance(r, tuple) else r[0].shape[1]
        self.batch_size = self.meta_batch_size * self.num_subtasks

        self.update_schedules()

        x_t = torch.zeros(
            (self.batch_size, self.num_steps + 1, self.dim),
            device=self.device,
        )

        log_weights = torch.zeros(
            (self.batch_size,),
            device=self.device,
        )

        noises = torch.normal(
            torch.zeros(
                (self.batch_size, self.num_steps, self.dim),
                device=self.device,
            ),
            torch.ones(
                (self.batch_size, self.num_steps, self.dim),
                device=self.device,
            ),
        )

        x = self.prior.sample((self.batch_size,)).to(self.device)
        x_t[:, 0] = x
        # (batch_size, num_entries, z_dim)

        for s in range(self.num_steps):
            alpha = self.alpha_vals[s]
            kappa = self.kappa_vals[s]

            score = self.absorb_num_subtasks(
                self.control(s, self.release_num_subtasks(x), r, mask, size, None)
            )

            means = x + (1 - torch.sqrt(1 - alpha)) * (2 * self.eta**2 * score - x)
            std = self.eta * torch.sqrt(alpha)
            x = means + std * noises[:, s]

            log_weights_deterministic = -2 * kappa * torch.sum(score**2, -1)

            log_weights_stochastic = (
                -2 * torch.sqrt(kappa) * torch.sum(score * noises[:, s], -1)
            )

            log_weights += log_weights_deterministic + log_weights_stochastic

            x_t[:, s + 1] = x

        log_weights += self.absorb_num_subtasks(
            target.log_prob(self.release_num_subtasks(x))
        ).sum(-1) - self.prior.log_prob(x).sum(-1)

        nabla_g = self.prior_score(x) - self.compute_target_score(target, x)

        return x_t.detach(), log_weights.detach(), nabla_g.detach()

    def compute_target_score(self, target: Distribution, z: Tensor) -> Tensor:
        z = z.requires_grad_(True)

        log_prob = self.absorb_num_subtasks(
            target.log_prob(self.release_num_subtasks(z))
        )

        score = torch.autograd.grad(
            outputs=log_prob,
            inputs=z,
            grad_outputs=torch.ones_like(log_prob),
        )[0]

        score = score.detach()

        return score

    def compute_loss(
        self,
        target: Distribution,
        r: Tensor | Tuple[Tensor, Tensor],
        mask: Tensor | None,
        size: Tensor | None,
    ) -> Tuple[Tensor, Tensor]:
        self.meta_batch_size = r.shape[0] if not isinstance(r, tuple) else r[0].shape[0]
        self.num_subtasks = r.shape[1] if not isinstance(r, tuple) else r[0].shape[1]
        self.batch_size = self.meta_batch_size * self.num_subtasks

        x, log_weights, nabla_g = self.sde_integrate(target, r, mask, size)

        scaled_log_weights = log_weights

        self_normalized = scaled_log_weights - torch.logsumexp(
            scaled_log_weights, dim=0
        )

        importance_weight = torch.exp(self_normalized)

        scaled_adj_states = self.noise_bwd_int.unsqueeze(1) * nabla_g.unsqueeze(
            1
        ).repeat(1, self.noise_bwd_int.shape[0], 1)

        scores = []

        for i in range(self.num_steps + 1):
            scores.append(
                self.absorb_num_subtasks(
                    self.control(
                        i, self.release_num_subtasks(x[:, i, :]), r, mask, size, None
                    )
                )
            )

        scores = torch.cat(scores, dim=1).view(x.shape)

        adjoint_err = (
            2
            * (torch.sum((scaled_adj_states + scores[:, :-1]) ** 2, axis=(1, 2)))
            * importance_weight
        )

        loss = 0.5 * torch.sum(adjoint_err)

        return loss, x.view(
            self.meta_batch_size, self.num_subtasks, self.num_steps + 1, self.dim
        )

    def absorb_num_subtasks(self, x: Tensor) -> Tensor:
        # (batch_size, num_subtasks, z_dim)
        return x.view(self.meta_batch_size * self.num_subtasks, -1)

    def release_num_subtasks(self, x: Tensor) -> Tensor:
        # (batch_size * num_subtasks, z_dim)
        return x.view(self.meta_batch_size, self.num_subtasks, -1)

    def forward_kernel(self, n, z):
        raise NotImplementedError

    def backward_kernel(self, n, z):
        return NotImplementedError
