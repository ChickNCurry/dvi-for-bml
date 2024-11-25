import math
from abc import ABC, abstractmethod
from typing import List, Tuple

import numpy as np
import torch
from torch import Tensor, nn
from torch.distributions import Distribution, Normal

from src.components.control import Control
from src.components.hyper_net import HyperNet


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

    @abstractmethod
    def get_prior(self, batch_size: int, device: torch.device) -> Distribution:
        pass

    @abstractmethod
    def forward_kernel(
        self,
        z_prev: Tensor,
        t: int,
        context_embedding: Tensor,
        mask: Tensor | None,
        p_z_0: Distribution,
        p_z_T: Distribution,
    ) -> Distribution:
        pass

    @abstractmethod
    def backward_kernel(
        self,
        z_next: Tensor,
        t: int,
        context_embedding: Tensor,
        mask: Tensor | None,
        p_z_0: Distribution,
        p_z_T: Distribution,
    ) -> Distribution:
        pass

    def run_chain(
        self,
        p_z_T: Distribution,
        context_embedding: Tensor,
        mask: Tensor | None,
    ) -> Tuple[Tensor, List[Tensor]]:

        p_z_0 = self.get_prior(
            batch_size=context_embedding.shape[0], device=context_embedding.device
        )

        z = [p_z_0.sample()]

        log_w: Tensor = torch.zeros(
            size=(context_embedding.shape[0], self.z_dim),
            device=context_embedding.device,
        )

        for i in range(0, self.num_steps):

            t = i + 1

            fwd_kernel = self.forward_kernel(
                z[t - 1], t, context_embedding, mask, p_z_0, p_z_T
            )

            z.append(fwd_kernel.rsample())

            bwd_kernel = self.backward_kernel(
                z[t], t - 1, context_embedding, mask, p_z_0, p_z_T
            )

            log_w += bwd_kernel.log_prob(z[t - 1]) - fwd_kernel.log_prob(z[t])

        log_w += p_z_T.log_prob(z[-1]) - p_z_0.log_prob(z[0])

        log_w = log_w.mean(dim=0).sum()

        return log_w, z


class DIS(DiffusionVIProcess):
    def __init__(
        self,
        device: torch.device,
        z_dim: int,
        num_steps: int,
        control: Control,
        hyper_net: HyperNet | None,
        min: float = 0.1,
    ) -> None:
        super(DIS, self).__init__(
            z_dim=z_dim,
            num_steps=num_steps,
        )

        self.control = control
        self.hyper_net = hyper_net

        self.beta_schedule = nn.ParameterList(
            [
                nn.Parameter(torch.tensor([beta], dtype=torch.float, device=device))
                for beta in [
                    (1 - min) * np.cos(math.pi * (1 - t) * 0.5) ** 2 + min
                    for t in np.linspace(1, 0, num_steps)
                ]
            ]
        )

        self.sigma_schedule = nn.ParameterList(
            [
                nn.Parameter(torch.tensor([sigma], dtype=torch.float, device=device))
                for sigma in [1]
            ]
        )

    def get_prior(self, batch_size: int, device: torch.device) -> Distribution:
        return Normal(  # type: ignore
            torch.zeros((batch_size, self.z_dim), device=device),
            torch.ones((batch_size, self.z_dim), device=device)
            * self.sigma_schedule[0],
        )

    def forward_kernel(
        self,
        z_prev: Tensor,
        t: int,
        context_embedding: Tensor,
        mask: Tensor | None,
        p_z_0: Distribution,
        p_z_T: Distribution,
    ) -> Distribution:
        # (batch_size, z_dim), (1), (batch_size, h_dim)

        beta_t = (
            self.beta_schedule[t - 1]
            if self.hyper_net is None
            else nn.Softplus()(
                0.9 * self.beta_schedule[t - 1]
                + 0.1 * self.hyper_net(t, context_embedding, mask)
            )
        )
        # (1)

        control = self.control(t, z_prev, context_embedding, mask)
        # (batch_size, z_dim)

        z_mu = z_prev + (beta_t * z_prev + control) * self.delta_t
        # (batch_size, z_dim)

        z_sigma = torch.sqrt(2 * beta_t * self.delta_t) * self.sigma_schedule[0]
        z_sigma = z_sigma.expand(z_mu.shape[0], -1)
        # (batch_size, z_dim)

        return Normal(z_mu, z_sigma)  # type: ignore

    def backward_kernel(
        self,
        z_next: Tensor,
        t: int,
        context_embedding: Tensor,
        mask: Tensor | None,
        p_z_0: Distribution,
        p_z_T: Distribution,
    ) -> Distribution:
        # (batch_size, z_dim), (1)

        beta_t = (
            self.beta_schedule[t - 1]
            if self.hyper_net is None
            else nn.Softplus()(
                0.9 * self.beta_schedule[t - 1]
                + 0.1 * self.hyper_net(t, context_embedding, mask)
            )
        )
        # (1)

        z_mu = z_next - (beta_t * z_next) * self.delta_t
        # (batch_size, z_dim)

        z_sigma = torch.sqrt(2 * beta_t * self.delta_t) * self.sigma_schedule[0]
        z_sigma = z_sigma.expand(z_mu.shape[0], -1)
        # (batch_size, z_dim)

        return Normal(z_mu, z_sigma)  # type: ignore


class CMCD(DiffusionVIProcess):
    def __init__(
        self,
        device: torch.device,
        z_dim: int,
        num_steps: int,
        control: Control,
        min: float = 0.2,
    ) -> None:
        super(CMCD, self).__init__(
            z_dim=z_dim,
            num_steps=num_steps,
        )

        self.control = control

        self.sigma_schedule = nn.ParameterList(
            [
                nn.Parameter(torch.tensor([sigma], dtype=torch.float, device=device))
                for sigma in [
                    (1 - min) * np.cos(math.pi * (1 - t) * 0.5) ** 2 + min
                    for t in np.linspace(1, 0, num_steps)
                ]
            ]
        )

        self.annealing_schedule = nn.ParameterList(
            [
                nn.Parameter(torch.tensor([beta], dtype=torch.float, device=device))
                for beta in np.linspace(min, 1 - min, num_steps)
            ]
        )

    def get_prior(self, batch_size: int, device: torch.device) -> Distribution:
        return Normal(  # type: ignore
            torch.zeros((batch_size, self.z_dim), device=device),
            torch.ones((batch_size, self.z_dim), device=device),
        )

    def forward_kernel(
        self,
        z_prev: Tensor,
        t: int,
        context_embedding: Tensor,
        mask: Tensor | None,
        p_z_0: Distribution,
        p_z_T: Distribution,
    ) -> Distribution:
        # (batch_size, z_dim), (1), (batch_size, h_dim)

        sigma_t = self.sigma_schedule[t - 1]
        # (1)

        control = self.control(t, z_prev, context_embedding, mask)
        # (batch_size, z_dim)

        grad_log = self.get_grad_log_geo_avg(z_prev, t, p_z_0, p_z_T)
        # (batch_size, z_dim)

        z_mu = z_prev + (sigma_t.pow(2) * grad_log + control) * self.delta_t
        # (batch_size, z_dim)

        z_sigma = sigma_t * np.sqrt(self.delta_t)
        z_sigma = z_sigma.expand(z_mu.shape[0], -1)
        # (batch_size, z_dim)

        return Normal(z_mu, z_sigma)  # type: ignore

    def backward_kernel(
        self,
        z_next: Tensor,
        t: int,
        context_embedding: Tensor,
        mask: Tensor | None,
        p_z_0: Distribution,
        p_z_T: Distribution,
    ) -> Distribution:
        # (batch_size, z_dim), (1), (batch_size, h_dim)

        sigma_t = self.sigma_schedule[t - 1]
        # (1)

        control = self.control(t, z_next, context_embedding, mask)
        # (batch_size, z_dim)

        grad_log = self.get_grad_log_geo_avg(z_next, t, p_z_0, p_z_T)
        # (batch_size, z_dim)

        z_mu = z_next + (sigma_t.pow(2) * grad_log - control) * self.delta_t
        # (batch_size, z_dim)

        z_sigma = sigma_t * np.sqrt(self.delta_t)
        z_sigma = z_sigma.expand(z_mu.shape[0], -1)
        # (batch_size, z_dim)

        return Normal(z_mu, z_sigma)  # type: ignore

    def get_grad_log_geo_avg(
        self,
        z: Tensor,
        t: int,
        p_z_0: Distribution,
        p_z_T: Distribution,
    ) -> Tensor:

        z = z.requires_grad_(True)

        beta_t = self.annealing_schedule[t - 1]

        log_geo_avg: Tensor = (1 - beta_t) * p_z_0.log_prob(
            z
        ) + beta_t * p_z_T.log_prob(z)

        grad = torch.autograd.grad(
            outputs=log_geo_avg,
            inputs=z,
            grad_outputs=torch.ones_like(log_geo_avg),
            create_graph=True,
            retain_graph=True,
        )[0]

        return grad
