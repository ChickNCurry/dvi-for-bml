from typing import Tuple

import torch
from torch import Tensor, nn
from torch.distributions.normal import Normal

from src.components.decoder.decoder import Decoder
from src.components.encoder.aggr_encoder import AggrEncoder
from src.components.encoder.bca_encoder import BCAEncoder


class AGGRCNP(nn.Module):
    def __init__(self, encoder: AggrEncoder, decoder: Decoder):
        super(AGGRCNP, self).__init__()

        self.encoder = encoder

        self.proj_r = nn.Linear(encoder.h_dim, decoder.z_dim)

        self.decoder = decoder

    def forward(
        self, context: Tensor, mask: Tensor | None, x_target: Tensor
    ) -> Tuple[Normal]:
        # (batch_size, num_subtasks, context_size, c_dim)
        # (batch_size, num_subtasks, context_size)
        # (batch_size, num_subtasks, target_size, x_dim)

        r = self.encoder(context, mask)
        # (batch_size, num_subtasks, h_dim)

        r = self.proj_r(r)
        # (batch_size, num_subtasks, z_dim)

        y_dist = self.decoder(r, x_target)
        # (batch_size, num_subtasks, target_size, y_dim)

        return (y_dist,)

    def loss(self, y_dist: Normal, y_target: Tensor, mask: Tensor | None) -> Tensor:
        return cnp_loss(y_dist, y_target, mask)


class BCACNP(nn.Module):
    def __init__(self, encoder: BCAEncoder, decoder: Decoder):
        super(BCACNP, self).__init__()

        self.encoder = encoder

        self.proj_r = nn.Linear(encoder.h_dim, decoder.z_dim)

        self.decoder = decoder

    def forward(
        self, context: Tensor, mask: Tensor | None, x_target: Tensor
    ) -> Tuple[Normal]:
        # (batch_size, num_subtasks, context_size, c_dim)
        # (batch_size, num_subtasks, context_size)
        # (batch_size, num_subtasks, target_size, x_dim)

        z_mu, z_var = self.encoder(context, mask)
        # (batch_size, num_subtasks, h_dim)

        r = self.proj_r(z_mu + z_var)
        # (batch_size, num_subtasks, z_dim)

        y_dist = self.decoder(r, x_target)
        # (batch_size, num_subtasks, target_size, y_dim)

        return (y_dist,)

    def loss(self, y_dist: Normal, y_target: Tensor, mask: Tensor | None) -> Tensor:
        return cnp_loss(y_dist, y_target, mask)


def cnp_loss(y_dist: Normal, y_target: Tensor, mask: Tensor | None) -> Tensor:
    # (batch_size, num_subtasks, target_size, y_dim)
    # (batch_size, num_subtasks, target_size, y_dim)
    # (batch_size, num_subtasks, context_size)

    log_like: Tensor = y_dist.log_prob(y_target)
    # (batch_size, num_subtasks, target_size, y_dim)

    if mask is not None:
        mask = mask.unsqueeze(-1).expand(-1, -1, -1, log_like.shape[-1])
        # (batch_size, num_subtasks, target_size, y_dim)

        log_like = log_like * mask

    log_like = log_like.sum(dim=-1).sum(dim=-1).mean()
    # (1)

    return -log_like


class AGGRNP(nn.Module):
    def __init__(self, encoder: AggrEncoder, decoder: Decoder):
        super(AGGRNP, self).__init__()

        self.encoder = encoder

        self.proj_z_mu = nn.Linear(encoder.h_dim, decoder.z_dim)
        self.proj_z_logvar = nn.Linear(encoder.h_dim, decoder.z_dim)

        self.decoder = decoder

    def forward(
        self, context: Tensor, mask: Tensor | None, x_target: Tensor
    ) -> Tuple[Normal, Normal, Tensor]:
        # (batch_size, num_subtasks, context_size, c_dim)
        # (batch_size, num_subtasks, context_size)
        # (batch_size, num_subtasks, target_size, x_dim)

        r = self.encoder(context, mask)
        # (batch_size, num_subtasks, h_dim)

        z_mu, z_log_var = self.proj_z_mu(r), self.proj_z_logvar(r)
        z_dist = Normal(z_mu, torch.exp(0.5 * z_log_var))
        z = z_dist.rsample()
        # (batch_size, num_subtasks, z_dim)

        y_dist = self.decoder(z, x_target)
        # (batch_size, num_subtasks, target_size, y_dim)

        return y_dist, z_dist, z

    def loss(
        self,
        y_dist: Normal,
        z_dist: Normal,
        z: Tensor,
        y_target: Tensor,
        mask: Tensor | None,
    ) -> Tensor:
        return np_loss(y_dist, z_dist, z, y_target, mask)


class BCANP(nn.Module):
    def __init__(self, encoder: BCAEncoder, decoder: Decoder):
        super(BCANP, self).__init__()

        self.encoder = encoder

        self.proj_z_mu = nn.Linear(encoder.h_dim, decoder.z_dim)
        self.proj_z_logvar = nn.Linear(encoder.h_dim, decoder.z_dim)

        self.decoder = decoder

    def forward(
        self, context: Tensor, mask: Tensor | None, x_target: Tensor
    ) -> Tuple[Normal, Normal, Tensor]:
        # (batch_size, num_subtasks, context_size, c_dim)
        # (batch_size, num_subtasks, context_size)
        # (batch_size, num_subtasks, target_size, x_dim)

        z_mu, z_var = self.encoder(context, mask)
        # (batch_size, num_subtasks, h_dim)

        z_mu, z_log_var = self.proj_z_mu(z_mu), self.proj_z_logvar(z_var)
        z_dist = Normal(z_mu, torch.exp(0.5 * z_log_var))
        z = z_dist.rsample()
        # (batch_size, num_subtasks, z_dim)

        y_dist = self.decoder(z, x_target)
        # (batch_size, num_subtasks, target_size, y_dim)

        return y_dist, z_dist, z

    def loss(
        self,
        y_dist: Normal,
        z_dist: Normal,
        z: Tensor,
        y_target: Tensor,
        mask: Tensor | None,
    ) -> Tensor:
        return np_loss(y_dist, z_dist, z, y_target, mask)


def np_loss(
    y_dist: Normal,
    z_dist: Normal,
    z: Tensor,
    y_target: Tensor,
    mask: Tensor | None,
) -> Tensor:
    # (batch_size, num_subtasks, target_size, y_dim)
    # (batch_size, num_subtasks, target_size, y_dim)
    # (batch_size, num_subtasks, context_size)
    # (batch_size, num_subtasks, z_dim)

    batch_size = y_target.shape[0]
    num_subtasks = y_target.shape[1]
    device = y_target.device
    z_dim = z.shape[-1]

    prior_dist = Normal(  # type: ignore
        torch.zeros((batch_size, num_subtasks, z_dim), device=device),
        torch.ones((batch_size, num_subtasks, z_dim), device=device),
    )

    log_prior: Tensor = prior_dist.log_prob(z)  # type: ignore
    log_prior = log_prior.sum(dim=-1, keepdim=True)
    # (batch_size, num_subtasks, 1)

    log_tp: Tensor = z_dist.log_prob(z)
    log_tp = log_tp.sum(dim=-1, keepdim=True)
    # (batch_size, num_subtasks, 1)

    log_like: Tensor = y_dist.log_prob(y_target)
    # (batch_size, num_subtasks, target_size, y_dim)

    if mask is not None:
        mask = mask.unsqueeze(-1).expand(-1, -1, -1, log_like.shape[-1])
        # (batch_size, num_subtasks, target_size, y_dim)

        log_like = log_like * mask

    log_like = log_like.sum(dim=-1).sum(dim=-1, keepdim=True)
    # (batch_size, num_subtasks, 1)

    return -(log_like + log_prior - log_tp).sum(dim=-1).mean()
