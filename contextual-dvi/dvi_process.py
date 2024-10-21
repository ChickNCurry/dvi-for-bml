import math
from abc import ABC, abstractmethod
from typing import List, Tuple

import numpy as np
import torch
from score_function import ScoreFunction
from torch import Tensor, nn
from torch.distributions import Distribution, Normal


class DiffusionVIProcess(nn.Module, ABC):
    def __init__(
        self,
        z_dim: int,
        num_steps: int,
    ) -> None:
        super(DiffusionVIProcess, self).__init__()

        assert num_steps > 0

        self.z_dim = z_dim
        self.num_steps = num_steps

        self.delta_t = 1.0 / num_steps
        self.sigmas: nn.ParameterList

    @abstractmethod
    def forward_kernel(self, z_prev: Tensor, t: int, c: Tensor) -> Distribution:
        pass

    @abstractmethod
    def backward_kernel(self, z_next: Tensor, t: int) -> Distribution:
        pass

    def run_chain(
        self,
        p_z_0: Distribution,
        p_z_T: Distribution,
        context: Tensor,
    ) -> Tuple[Tensor, List[Tensor]]:

        z_samples = [p_z_0.sample()]

        fwd_log_probs = []
        bwd_log_probs = []

        for i in range(0, self.num_steps):

            t = i + 1

            fwd_kernel = self.forward_kernel(z_samples[t - 1], t, context)
            z_samples.append(fwd_kernel.rsample())

            bwd_kernel = self.backward_kernel(z_samples[t], t - 1)

            fwd_log_probs.append(fwd_kernel.log_prob(z_samples[t]))
            bwd_log_probs.append(bwd_kernel.log_prob(z_samples[t - 1]))

        fwd_log_probs.append(p_z_0.log_prob(z_samples[0]))
        bwd_log_probs.append(p_z_T.log_prob(z_samples[-1]))

        assert len(fwd_log_probs) == self.num_steps + 1
        assert len(bwd_log_probs) == self.num_steps + 1

        log_w = torch.stack(
            [
                (bwd_log_probs[i] - fwd_log_probs[i]).mean(dim=0).sum()
                for i in range(self.num_steps + 1)
            ]
        ).sum(dim=0)

        return log_w, z_samples


class DIS(DiffusionVIProcess):
    def __init__(
        self,
        z_dim: int,
        num_steps: int,
        score_function: ScoreFunction,
        device: torch.device,
    ) -> None:
        super(DIS, self).__init__(
            z_dim=z_dim,
            num_steps=num_steps,
        )

        self.score_function = score_function

        self.beta_schedule = [
            3 * np.pow(np.cos(math.pi * (1 - t) / 2), 2)  # t
            for t in np.linspace(1, 0.01, num_steps)
        ]

        self.betas = nn.ParameterList(
            [
                nn.Parameter(torch.tensor([beta], dtype=torch.float, device=device))
                for beta in self.beta_schedule
            ]
        )

        self.sigmas = nn.ParameterList([nn.Parameter(torch.ones(z_dim, device=device))])

    def forward_kernel(self, z_prev: Tensor, t: int, c: Tensor) -> Distribution:
        # (batch_size, z_dim), (1), (batch_size, c_dim)

        beta_t = self.betas[t - 1]
        # (1)

        score = self.score_function(z_prev, t, c)
        # (batch_size, z_dim)

        z_mu = z_prev + (beta_t * z_prev + score) * self.delta_t

        z_sigma = torch.sqrt(2 * beta_t * self.delta_t)  # * self.sigmas[0]
        z_sigma = z_sigma.repeat(z_mu.shape[0], 1)
        # (batch_size, z_dim)

        return Normal(z_mu, z_sigma)  # type: ignore

    def backward_kernel(self, z_next: Tensor, t: int) -> Distribution:
        # (batch_size, z_dim), (1)

        beta_t = self.betas[t - 1]
        # (1)

        z_mu = z_next - (beta_t * z_next) * self.delta_t
        z_sigma = torch.sqrt(2 * beta_t * self.delta_t)  # * self.sigmas[0]
        z_sigma = z_sigma.repeat(z_mu.shape[0], 1)
        # (batch_size, z_dim)

        return Normal(z_mu, z_sigma)  # type: ignore
