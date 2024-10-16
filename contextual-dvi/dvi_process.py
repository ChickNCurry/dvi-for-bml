from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple

import torch
from score_function import ScoreFunction
from torch import Tensor, nn


@dataclass
class zTuple:
    def __init__(self, z: Tensor, z_mu: Tensor, z_sigma: Tensor) -> None:
        self.z = z
        self.z_mu = z_mu
        self.z_sigma = z_sigma


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
        self.delta_t = torch.tensor(1 / num_steps)

    @abstractmethod
    def forward_z_mu_and_z_sigma(
        self, z_prev: Tensor, t: int, h: Tensor
    ) -> Tuple[Tensor, Tensor]:
        pass

    @abstractmethod
    def backward_z_mu_and_z_sigma(
        self, z_next: Tensor, t: int
    ) -> Tuple[Tensor, Tensor]:
        pass

    def reparameterize(self, z_mu: Tensor, z_sigma: Tensor) -> Tensor:
        # (batch_size, z_dim)

        eps = torch.randn_like(z_sigma)
        z = z_mu + z_sigma * eps
        # (batch_size, z_dim)

        return z

    def forward_process(
        self,
        z_0_tuple: zTuple,
        context: Tensor,
        z_samples: List[Tensor] | None,
    ) -> List[zTuple]:

        z_tuples = [z_0_tuple]

        if z_samples is not None:
            assert len(z_samples) == self.num_steps + 1

        for t in range(0, self.num_steps):

            z_prev = z_tuples[-1].z if z_samples is None else z_samples[t]

            z_mu, z_sigma = self.forward_z_mu_and_z_sigma(z_prev, t, context)
            z = (
                self.reparameterize(z_mu, z_sigma)
                if z_samples is None
                else z_samples[t + 1]
            )

            z_tuples.append(zTuple(z, z_mu, z_sigma))

        assert len(z_tuples) == self.num_steps + 1

        return z_tuples

    def backward_process(
        self,
        z_T_tuple: zTuple,
        z_samples: List[Tensor] | None,
    ) -> List[zTuple]:

        z_tuples = [z_T_tuple]

        if z_samples is not None:
            assert len(z_samples) == self.num_steps + 1

        for t in range(self.num_steps - 1, -1, -1):

            z_next = z_tuples[-1].z if z_samples is None else z_samples[t + 1]

            z_mu, z_sigma = self.backward_z_mu_and_z_sigma(z_next, t - 1)
            z = (
                self.reparameterize(z_mu, z_sigma)
                if z_samples is None
                else z_samples[t]
            )

            z_tuples.append(zTuple(z, z_mu, z_sigma))

        assert len(z_tuples) == self.num_steps + 1

        return z_tuples


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

        # self.betas = nn.ParameterList(
        #     [
        #         nn.Parameter(torch.tensor([t]))
        #         for t in torch.linspace(0.001, 10, num_steps).tolist()
        #     ]
        # )

        self.betas = nn.ParameterList(
            [nn.Parameter(torch.tensor([1.0])) for _ in range(num_steps)]
        )
        self.sigmas = nn.ParameterList([nn.Parameter(torch.ones(z_dim))])

    def forward_z_mu_and_z_sigma(
        self, z_prev: Tensor, t: int, c: Tensor
    ) -> Tuple[Tensor, Tensor]:
        # (batch_size, z_dim), (1)

        beta_t = self.betas[t - 1]
        # (1)

        score = self.score_function(z_prev, t, c)
        z_mu = z_prev + (beta_t * z_prev + score) * self.delta_t
        # (batch_size, z_dim)

        z_sigma = torch.sqrt(2 * beta_t * self.delta_t) * self.sigmas[0]
        z_sigma = z_sigma.repeat(z_mu.shape[0], 1)
        # (batch_size, z_dim)

        return z_mu, z_sigma

    def backward_z_mu_and_z_sigma(
        self, z_next: Tensor, t: int
    ) -> Tuple[Tensor, Tensor]:
        # (batch_size, z_dim), (1)

        beta_t = self.betas[t - 1]
        # (1)

        z_mu = (z_next - beta_t * z_next) * self.delta_t
        # (batch_size, z_dim)

        z_sigma = torch.sqrt(2 * beta_t * self.delta_t) * self.sigmas[0]
        z_sigma = z_sigma.repeat(z_mu.shape[0], 1)
        # (batch_size, z_dim)

        return z_mu, z_sigma
