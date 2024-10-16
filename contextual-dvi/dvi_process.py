from abc import ABC, abstractmethod
from typing import List, Tuple

import torch
from score_function import ScoreFunction
from torch import Tensor, nn
from torch.distributions import Distribution, Normal


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

    # def reparametrize(self, z_mu: Tensor, z_sigma: Tensor) -> Tensor:
    #     # (batch_size, z_dim), (batch_size, z_dim)

    #     eps = torch.randn_like(z_mu)
    #     # (batch_size, z_dim)

    #     z = z_mu + eps * z_sigma
    #     # (batch_size, z_dim)

    #     return z

    @abstractmethod
    def forward_p_z(self, z_prev: Tensor, t: int, c: Tensor | None) -> Distribution:
        pass

    @abstractmethod
    def backward_p_z(self, z_next: Tensor, t: int) -> Distribution:
        pass

    def forward_chain(
        self,
        p_z_0: Distribution,
        context: Tensor | None,
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

    def forward_p_z(self, z_prev: Tensor, t: int, c: Tensor | None) -> Distribution:
        # (batch_size, z_dim), (1), (batch_size, c_dim)

        beta_t = self.betas[t - 1]
        # (1)

        score = self.score_function(z_prev, t, c)
        z_mu = z_prev + (beta_t * z_prev + score) * self.delta_t
        z_sigma = torch.sqrt(2 * beta_t * self.delta_t) * self.sigmas[0]
        z_sigma = z_sigma.repeat(z_mu.shape[0], 1)
        # (batch_size, z_dim)

        p_z = Normal(z_mu, z_sigma)  # type: ignore

        return p_z

    def backward_p_z(self, z_next: Tensor, t: int) -> Distribution:
        # (batch_size, z_dim), (1)

        beta_t = self.betas[t - 1]
        # (1)

        z_mu = (z_next - beta_t * z_next) * self.delta_t
        z_sigma = torch.sqrt(2 * beta_t * self.delta_t) * self.sigmas[0]
        z_sigma = z_sigma.repeat(z_mu.shape[0], 1)
        # (batch_size, z_dim)

        p_z = Normal(z_mu, z_sigma)  # type: ignore

        return p_z
