from abc import ABC, abstractmethod
from typing import List, Tuple

import torch
from torch import Tensor, nn
from torch.distributions import Distribution, Normal
from torch.distributions.kl import kl_divergence


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
        s: Tensor | None,
    ) -> None:
        self.target = target
        self.r = r
        self.mask = mask
        self.s = s

        device = r[0].device if isinstance(r, tuple) else r.device
        batch_size = r[0].shape[0] if isinstance(r, tuple) else r.shape[0]
        num_subtasks = r[0].shape[1] if isinstance(r, tuple) else r.shape[1]

        self.device = device
        self.size = (batch_size, num_subtasks)

        self.prior: Distribution = Normal(
            torch.zeros((batch_size, num_subtasks, self.z_dim), device=device),
            torch.ones((batch_size, num_subtasks, self.z_dim), device=device),
        )  # (batch_size, num_subtasks, z_dim)

    @abstractmethod
    def forward_kernel(
        self,
        n: int,
        z: Tensor,
    ) -> Distribution:
        raise NotImplementedError

    @abstractmethod
    def backward_kernel(
        self,
        n: int,
        z: Tensor,
    ) -> Distribution:
        raise NotImplementedError

    def run_forward_process(
        self,
        target: Distribution,
        r: Tensor,
        mask: Tensor | None,
        s: Tensor | None,
        other_zs: List[Tensor] | None,
    ) -> Tuple[Tensor, List[Tensor] | None]:
        self.contextualize(target, r, mask, s)

        zs = [self.prior.sample()] if other_zs is None else other_zs
        # (batch_size, num_subtasks, z_dim)

        log_prob = self.prior.log_prob(zs[0]).sum(-1)
        # (batch_size, num_subtasks, z_dim)

        for n in range(0, self.num_steps):
            fwd_kernel = self.forward_kernel(n, zs[n])

            if other_zs is None:
                zs.append(fwd_kernel.rsample())

            log_prob += fwd_kernel.log_prob(zs[n + 1]).sum(-1)
            # (batch_size, num_subtasks)

        log_prob = log_prob.mean()
        # (1)

        return log_prob, zs

    def run_backward_process(
        self,
        target: Distribution,
        r: Tensor,
        mask: Tensor | None,
        s: Tensor | None,
        zs: List[Tensor],
    ) -> Tensor:
        self.contextualize(target, r, mask, s)

        log_prob = self.target.log_prob(zs[-1]).sum(-1)
        # (batch_size, num_subtasks)

        for n in range(0, self.num_steps):
            bwd_kernel = self.backward_kernel(n + 1, zs[n + 1])
            log_prob += bwd_kernel.log_prob(zs[n]).sum(-1)
            # (batch_size, num_subtasks)

        log_prob = log_prob.mean()
        # (1)

        return log_prob

    def run_2_forward_processes(
        self,
        target: Distribution,
        r_data: Tensor | Tuple[Tensor, Tensor],
        r_context: Tensor | Tuple[Tensor, Tensor],
        mask_context: Tensor | None,
        s_data: Tensor | None,
        s_context: Tensor | None,
    ) -> Tuple[Tensor, List[Tensor]]:
        self.contextualize(target, r_data, None, s_data)

        z_dists_data = []
        z_dists_context = []
        zs = [self.prior.sample()]
        # (batch_size, num_subtasks, z_dim)

        for n in range(0, self.num_steps):
            fwd_kernel_data = self.forward_kernel(n, zs[n])
            z_dists_data.append(fwd_kernel_data)
            zs.append(fwd_kernel_data.rsample())

        self.contextualize(target, r_context, mask_context, s_context)

        for n in range(0, self.num_steps):
            fwd_kernel_context = self.forward_kernel(n, zs[n])
            z_dists_context.append(fwd_kernel_context)

        log_prob = (
            torch.stack(
                [
                    kl_divergence(zd, zc).sum(-1)
                    for zd, zc in zip(z_dists_data, z_dists_context)
                ],
                dim=0,
            )
            .sum(0)
            .mean()
        )  # (1)

        return log_prob, zs

    def run_both_processes(
        self,
        target: Distribution,
        r: Tensor | Tuple[Tensor, Tensor],
        mask: Tensor | None,
        s: Tensor | None,
    ) -> Tuple[Tensor, List[Tensor]]:
        self.contextualize(target, r, mask, s)

        zs = [self.prior.sample()]
        # (batch_size, num_subtasks, z_dim)

        elbo = torch.zeros(self.size, device=self.device)
        # (batch_size, num_subtasks)

        for n in range(0, self.num_steps):
            fwd_kernel = self.forward_kernel(n, zs[n])
            zs.append(fwd_kernel.rsample())
            bwd_kernel = self.backward_kernel(n + 1, zs[n + 1])

            elbo += bwd_kernel.log_prob(zs[n]).sum(-1)
            elbo -= fwd_kernel.log_prob(zs[n + 1]).sum(-1)
            # (batch_size, num_subtasks)

        elbo += self.target.log_prob(zs[-1]).sum(-1)
        elbo -= self.prior.log_prob(zs[0]).sum(-1)
        # (batch_size, num_subtasks)

        elbo = elbo.mean()
        # (1)

        return elbo, zs
