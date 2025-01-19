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
        r: Tensor | Tuple[Tensor, Tensor],
        mask: Tensor | None,
    ) -> None:

        self.target = target
        self.r = r
        self.mask = mask

        device = r[0].device if isinstance(r, tuple) else r.device
        batch_size = r[0].shape[0] if isinstance(r, tuple) else r.shape[0]
        num_subtasks = r[0].shape[1] if isinstance(r, tuple) else r.shape[1]

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
        z: Tensor,
    ) -> Distribution:
        pass

    @abstractmethod
    def backward_kernel(
        self,
        n: int,
        z: Tensor,
    ) -> Distribution:
        pass

    def run_chain(
        self,
        target: Distribution,
        r: Tensor | Tuple[Tensor, Tensor],
        mask: Tensor | None,
    ) -> Tuple[Tensor, Tensor, List[Tensor]]:

        self.contextualize(target, r, mask)

        z = [self.prior.sample()]

        elbo: Tensor = torch.zeros(self.size, device=self.device)
        # (batch_size, num_subtasks, z_dim)

        for n in range(0, self.num_steps):

            fwd_kernel = self.forward_kernel(n, z[n])
            z.append(fwd_kernel.rsample())
            bwd_kernel = self.backward_kernel(n + 1, z[n + 1])

            elbo += bwd_kernel.log_prob(z[n]) - fwd_kernel.log_prob(z[n + 1])
            # (batch_size, num_subtasks, z_dim or 1)

        elbo += self.target.log_prob(z[-1]) - self.prior.log_prob(z[0])
        # (batch_size, num_subtasks, z_dim or 1)

        elbo = elbo.sum(-1).mean()
        # (1)

        return elbo, torch.zeros_like(elbo), z
