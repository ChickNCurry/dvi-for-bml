from abc import ABC, abstractmethod
from typing import List, Tuple

import torch
from torch import Tensor, nn
from torch.distributions import Distribution, Normal


class CDVI(nn.Module, ABC):
    def __init__(
        self,
        z_dim: int,
        num_steps: int,
        device: torch.device,
    ) -> None:
        super(CDVI, self).__init__()

        assert num_steps > 0

        self.z_dim = z_dim
        self.num_steps = num_steps
        self.device = device

    def contextualize(
        self,
        target: Distribution,
        r_aggr: Tensor | None,
        r_non_aggr: Tensor | None,
        mask: Tensor | None,
    ) -> None:

        self.target = target
        self.r_aggr = r_aggr
        self.r_non_aggr = r_non_aggr
        self.mask = mask

        if r_aggr is not None:
            device = r_aggr.device
            batch_size = r_aggr.shape[0]
            num_subtasks = r_aggr.shape[1]

        elif r_non_aggr is not None:
            device = r_non_aggr.device
            batch_size = r_non_aggr.shape[0]
            num_subtasks = r_non_aggr.shape[1]

        self.device = device
        self.size = (batch_size, num_subtasks, self.z_dim)

        self.prior: Distribution = Normal(  # type: ignore
            torch.zeros(self.size, device=device),
            torch.ones(self.size, device=device),
        )

    @abstractmethod
    def forward_kernel(
        self,
        n: int,
        z_prev: Tensor,
    ) -> Distribution:
        pass

    @abstractmethod
    def backward_kernel(
        self,
        n: int,
        z_next: Tensor,
    ) -> Distribution:
        pass

    def run_chain(
        self,
        target: Distribution,
        r_aggr: Tensor | None,
        r_non_aggr: Tensor | None,
        mask: Tensor | None,
    ) -> Tuple[Tensor, Tensor, List[Tensor]]:

        self.contextualize(target, r_aggr, r_non_aggr, mask)

        z = [self.prior.sample()]

        elbo: Tensor = torch.zeros(self.size, device=self.device)
        # (batch_size, num_subtasks, z_dim)

        for i in range(0, self.num_steps):
            n = i + 1

            fwd_kernel = self.forward_kernel(n, z[n - 1])
            z.append(fwd_kernel.rsample())
            bwd_kernel = self.backward_kernel(n - 1, z[n])

            elbo += bwd_kernel.log_prob(z[n - 1]) - fwd_kernel.log_prob(z[n])
            # (batch_size, num_subtasks, z_dim or 1)

        elbo += self.target.log_prob(z[-1]) - self.prior.log_prob(z[0])
        # (batch_size, num_subtasks, z_dim or 1)

        elbo = elbo.sum(-1).mean()
        # (1)

        return elbo, torch.zeros_like(elbo), z
