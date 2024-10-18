from abc import ABC, abstractmethod
from typing import List, Tuple

import torch
from score_function import ScoreFunction
from torch import Tensor, nn
from torch.distributions import Distribution, Normal
import math


class DiffusionVIProcess(nn.Module, ABC):
    def __init__(
        self,
        z_dim: int,
        num_steps: int,
        score_function: ScoreFunction,
    ) -> None:
        super(DiffusionVIProcess, self).__init__()

        assert num_steps > 0

        self.z_dim = z_dim
        self.num_steps = num_steps
        self.score_function = score_function

        self.delta_t = torch.tensor(1.0 / num_steps)
        self.sigmas: nn.ParameterList

    @abstractmethod
    def forward_p_z(self, z_prev: Tensor, t: int, c: Tensor) -> Distribution:
        pass

    @abstractmethod
    def backward_p_z(self, z_next: Tensor, t: int) -> Distribution:
        pass

    def forward_chain(
        self,
        p_z_0: Distribution,
        context: Tensor,
    ) -> Tuple[List[Distribution], List[Tensor]]:

        p_z_list = [p_z_0]
        z_list = [p_z_0.sample()]

        for i in range(0, self.num_steps):

            t = i + 1

            p_z = self.forward_p_z(z_list[i], t, context)
            z = p_z.rsample()

            p_z_list.append(p_z)
            z_list.append(z)

        assert len(p_z_list) == len(z_list) == self.num_steps + 1

        return p_z_list, z_list

    def backward_chain(
        self,
        p_z_T: Distribution,
        z_samples: List[Tensor],
    ) -> List[Distribution]:

        assert len(z_samples) == self.num_steps + 1

        p_z_list = [p_z_T]

        for i in range(self.num_steps, 0, -1):

            t = i - 1

            p_z = self.backward_p_z(z_samples[i], t)

            p_z_list.append(p_z)

        assert len(p_z_list) == self.num_steps + 1

        p_z_list.reverse()

        return p_z_list


class DIS(DiffusionVIProcess):
    def __init__(
        self,
        z_dim: int,
        num_steps: int,
        score_function: ScoreFunction,
    ) -> None:
        super(DIS, self).__init__(
            z_dim=z_dim,
            num_steps=num_steps,
            score_function=score_function,
        )

        self.betas = nn.ParameterList(
            [
                nn.Parameter(torch.tensor([t]))
                for t in torch.linspace(0.01, 1, num_steps).tolist()
            ]
        )

        self.sigmas = nn.ParameterList([nn.Parameter(torch.ones(z_dim))])

    def forward_p_z(self, z_prev: Tensor, t: int, c: Tensor) -> Distribution:
        # (batch_size, z_dim), (1), (batch_size, c_dim)

        beta_t = self.betas[t - 1]
        # (1)

        score = self.score_function(z_prev, t, c)
        # (batch_size, z_dim)

        z_mu = z_prev + (beta_t * z_prev + score) * self.delta_t
        z_sigma = torch.sqrt(2 * beta_t * self.delta_t) * self.sigmas[0]
        z_sigma = z_sigma.repeat(z_mu.shape[0], 1)
        # (batch_size, z_dim)

        return Normal(z_mu, z_sigma)  # type: ignore

    def backward_p_z(self, z_next: Tensor, t: int) -> Distribution:
        # (batch_size, z_dim), (1)

        beta_t = self.betas[t - 1]
        # (1)

        z_mu = (z_next - beta_t * z_next) * self.delta_t
        z_sigma = torch.sqrt(2 * beta_t * self.delta_t) * self.sigmas[0]
        z_sigma = z_sigma.repeat(z_mu.shape[0], 1)
        # (batch_size, z_dim)

        return Normal(z_mu, z_sigma)  # type: ignore


class DDS(DiffusionVIProcess):
    def __init__(
        self,
        z_dim: int,
        num_steps: int,
        score_function: ScoreFunction,
    ) -> None:
        super(DDS, self).__init__(
            z_dim=z_dim,
            num_steps=num_steps,
            score_function=score_function,
        )

        self.betas = [
            torch.tensor([t]) for t in torch.linspace(1e-4, 0.02, num_steps).tolist()
        ]

        self.sigmas = nn.ParameterList([nn.Parameter(torch.ones(z_dim))])

    def forward_p_z(self, z_prev: Tensor, t: int, c: Tensor) -> Distribution:
        # (batch_size, z_dim), (1), (batch_size, c_dim)

        beta_t = self.betas[t - 1].to(z_prev.device)
        # (1)

        score = self.score_function(z_prev, t, c)
        # (batch_size, z_dim)

        z_mu = (torch.sqrt(1 - beta_t) * z_prev + score) * self.delta_t
        z_sigma = torch.sqrt(beta_t * self.delta_t) * self.sigmas[0]
        z_sigma = z_sigma.repeat(z_mu.shape[0], 1)
        # (batch_size, z_dim)

        return Normal(z_mu, z_sigma)  # type: ignore

    def backward_p_z(self, z_next: Tensor, t: int) -> Distribution:
        # (batch_size, z_dim), (1)

        beta_t = self.betas[t - 1].to(z_next.device)
        # (1)

        z_mu = (torch.sqrt(1 - beta_t) * z_next) * self.delta_t
        z_sigma = torch.sqrt(beta_t * self.delta_t) * self.sigmas[0]
        z_sigma = z_sigma.repeat(z_mu.shape[0], 1)
        # (batch_size, z_dim)

        return Normal(z_mu, z_sigma)  # type: ignore


class PIS(DiffusionVIProcess):
    def __init__(
        self,
        z_dim: int,
        num_steps: int,
        score_function: ScoreFunction,
    ) -> None:
        super(PIS, self).__init__(
            z_dim=z_dim, num_steps=num_steps, score_function=score_function
        )

        self.sigma_max = 1
        self.sigma_schedule = [
            self.sigma_max * torch.cos(math.pi * (1 - t) / 2).pow(2)
            for t in torch.linspace(0.1, 0.9, num_steps + 1)
        ]

    def forward_p_z(self, z_prev: Tensor, t: int, c: Tensor) -> Distribution:
        # (batch_size, z_dim), (1), (batch_size, c_dim)

        sigma_t = self.sigma_schedule[t].to(z_prev.device)
        # (1)

        score = self.score_function(z_prev, t, c)
        # (batch_size, z_dim)

        z_mu = z_prev + score * self.delta_t
        z_sigma = sigma_t * torch.sqrt(self.delta_t)
        z_sigma = z_sigma.repeat(z_mu.shape[0], 1)
        # (batch_size, z_dim)

        return Normal(z_mu, z_sigma)  # type: ignore

    def backward_p_z(self, z_next: Tensor, t: int) -> Distribution:
        # (batch_size, z_dim), (1)

        sigma_t = self.sigma_schedule[t].to(z_next.device)
        # (1)

        z_mu = ((t - self.delta_t) / t) * z_next
        z_sigma = torch.sqrt(((t - self.delta_t) / t) * self.delta_t) * sigma_t
        z_sigma = z_sigma.repeat(z_mu.shape[0], 1)
        # (batch_size, z_dim)

        return Normal(z_mu, z_sigma)  # type: ignore
