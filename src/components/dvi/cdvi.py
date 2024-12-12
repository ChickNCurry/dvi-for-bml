from abc import ABC, abstractmethod
from typing import List, Tuple

import torch
from torch import Tensor, nn
from torch.distributions import Distribution


class CDVI(nn.Module, ABC):
    def __init__(
        self,
        z_dim: int,
        num_steps: int,
    ) -> None:
        super(CDVI, self).__init__()

        assert num_steps > 0

        self.z_dim = z_dim
        self.num_steps = num_steps
        self.delta_t = 1.0 / num_steps

    @abstractmethod
    def get_prior(
        self, size: Tuple[int, int, int], device: torch.device
    ) -> Distribution:
        pass

    @abstractmethod
    def forward_kernel(
        self,
        t: int,
        z_prev: Tensor,
        r_aggr: Tensor | None,
        r_non_aggr: Tensor | None,
        mask: Tensor | None,
        prior: Distribution,
        target: Distribution,
    ) -> Distribution:
        pass

    @abstractmethod
    def backward_kernel(
        self,
        t: int,
        z_next: Tensor,
        r_aggr: Tensor | None,
        r_non_aggr: Tensor | None,
        mask: Tensor | None,
        prior: Distribution,
        target: Distribution,
    ) -> Distribution:
        pass

    def run_chain(
        self,
        target: Distribution,
        r_aggr: Tensor | None,
        r_non_aggr: Tensor | None,
        mask: Tensor | None,
    ) -> Tuple[Tensor, Tensor, List[Tensor]]:

        if r_aggr is not None:
            device = r_aggr.device
            batch_size = r_aggr.shape[0]
            num_subtasks = r_aggr.shape[1]

        elif r_non_aggr is not None:
            device = r_non_aggr.device
            batch_size = r_non_aggr.shape[0]
            num_subtasks = r_non_aggr.shape[1]

        prior = self.get_prior((batch_size, num_subtasks, self.z_dim), device)
        # (batch_size, num_subtasks, z_dim)

        z = [prior.sample()]

        elbo: Tensor = torch.zeros(
            size=(batch_size, num_subtasks, self.z_dim), device=device
        )  # (batch_size, num_subtasks, z_dim)

        for i in range(0, self.num_steps):

            t = i + 1

            fwd_kernel = self.forward_kernel(
                t, z[t - 1], r_aggr, r_non_aggr, mask, prior, target
            )

            z.append(fwd_kernel.rsample())

            bwd_kernel = self.backward_kernel(
                t - 1, z[t], r_aggr, r_non_aggr, mask, prior, target
            )

            elbo += bwd_kernel.log_prob(z[t - 1]) - fwd_kernel.log_prob(z[t])
            # (batch_size, num_subtasks, z_dim or 1)

        log_like = target.log_prob(z[-1])
        elbo += log_like - prior.log_prob(z[0])
        # (batch_size, num_subtasks, z_dim or 1)

        elbo = elbo.mean(dim=0).mean(dim=0).sum()
        log_like = log_like.mean(dim=0).mean(dim=0).sum()
        # (1)

        return elbo, log_like, z
