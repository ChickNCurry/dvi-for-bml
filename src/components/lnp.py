from typing import Tuple

from torch import Tensor, nn
import torch
from torch.distributions.normal import Normal

from src.components.decoder.decoder import Decoder
from src.components.encoder.aggr_encoder import AggrEncoder
from src.components.encoder.bca_encoder import BCAEncoder
from torch.nn.functional import softplus


class AggrLNP(nn.Module):
    def __init__(
        self,
        encoder: AggrEncoder,
        decoder: Decoder,
    ):
        super(AggrLNP, self).__init__()

        self.encoder = encoder
        self.proj_z_mu = nn.Linear(encoder.h_dim, decoder.z_dim)
        self.proj_z_sigma = nn.Linear(encoder.h_dim, decoder.z_dim)
        self.decoder = decoder

    def encode(self, context: Tensor, mask: Tensor | None) -> Tuple[Normal, Tensor]:
        # (batch_size, num_subtasks, data_size, c_dim)
        # (batch_size, num_subtasks, data_size)

        r = self.encoder(context, mask)
        # (batch_size, num_subtasks, h_dim)

        # z_mu, z_log_var = self.proj_z_mu(r), self.proj_z_logvar(r)
        # z_dist = Normal(z_mu, torch.exp(0.5 * z_log_var))

        z_mu = self.proj_z_mu(r)
        z_sigma = softplus(self.proj_z_sigma(r))
        z_dist = Normal(z_mu, z_sigma)

        z = z_dist.rsample()
        # (batch_size, num_subtasks, z_dim)

        return z_dist, z

    def forward(
        self, context: Tensor, mask: Tensor | None, x: Tensor
    ) -> Tuple[Normal, Normal, Tensor]:
        # (batch_size, num_subtasks, data_size, c_dim)
        # (batch_size, num_subtasks, data_size)
        # (batch_size, num_subtasks, data_size, x_dim)

        z_dist, z = self.encode(context, mask)
        # (batch_size, num_subtasks, z_dim)

        y_dist = self.decoder(z, x)
        # (batch_size, num_subtasks, data_size, y_dim)

        return y_dist, z_dist, z


class BcaLNP(nn.Module):
    def __init__(
        self,
        encoder: BCAEncoder,
        decoder: Decoder,
    ):
        super(BcaLNP, self).__init__()

        self.encoder = encoder
        self.decoder = decoder

    def encode(self, context: Tensor, mask: Tensor | None) -> Tuple[Normal, Tensor]:
        # (batch_size, num_subtasks, data_size, c_dim)
        # (batch_size, num_subtasks, data_size)

        z_mu, z_var = self.encoder(context, mask)
        z_dist = Normal(z_mu, torch.sqrt(z_var))
        z = z_dist.rsample()
        # (batch_size, num_subtasks, z_dim)

        return z_dist, z

    def forward(
        self, context: Tensor, mask: Tensor | None, x: Tensor
    ) -> Tuple[Normal, Normal, Tensor]:
        # (batch_size, num_subtasks, data_size, c_dim)
        # (batch_size, num_subtasks, data_size)
        # (batch_size, num_subtasks, data_size, x_dim)

        z_dist, z = self.encode(context, mask)
        # (batch_size, num_subtasks, z_dim)

        y_dist = self.decoder(z, x)
        # (batch_size, num_subtasks, data_size, y_dim)

        return y_dist, z_dist, z
