from abc import ABC, abstractmethod
from typing import Tuple

import torch
from torch import Tensor, nn
from torch.distributions.distribution import Distribution
from torch.distributions.normal import Normal
from torch.nn.functional import softplus

from src.architectures.np import NP
from src.components.decoder.decoder import Decoder
from src.components.decoder.decoder_times_prior import DecoderTimesPrior
from src.components.encoder.abstract_encoder import AbstractEncoder
from src.components.encoder.aggr_encoder import AggrEncoder
from src.components.encoder.bca_encoder import BCAEncoder


class LNP(NP, ABC):
    def __init__(self, encoder: AbstractEncoder, decoder: Decoder) -> None:
        super(LNP, self).__init__(encoder, decoder)

    @abstractmethod
    def forward(
        self, context: Tensor, mask: Tensor | None, x: Tensor
    ) -> Tuple[Normal, Normal, Tensor]:
        raise NotImplementedError

    def inference(
        self, x_context: Tensor, y_context: Tensor, mask: Tensor | None, x_data: Tensor
    ) -> Tuple[Normal, Tensor | None]:
        context = torch.cat([x_context, y_context], dim=-1)

        y_dist, _, z = self(context, mask, x_data)

        assert isinstance(y_dist, Normal)

        return y_dist, z

    def get_target_dist(
        self, x_context: Tensor, y_context: Tensor, mask: Tensor | None
    ) -> Distribution | None:
        target_dist = DecoderTimesPrior(
            decoder=self.decoder,
            x=x_context,
            y=y_context,
            mask=mask,
        )

        return target_dist


class AggrLNP(LNP):
    def __init__(
        self,
        encoder: AggrEncoder,
        decoder: Decoder,
    ):
        super(AggrLNP, self).__init__(encoder, decoder)

        self.proj_z_mu = nn.Linear(encoder.h_dim, decoder.z_dim)
        self.proj_z_sigma = nn.Linear(encoder.h_dim, decoder.z_dim)

    def encode(self, context: Tensor, mask: Tensor | None) -> Tuple[Normal, Tensor]:
        # (batch_size, num_subtasks, data_size, c_dim)
        # (batch_size, num_subtasks, data_size)

        r = self.encoder(context, mask)
        # (batch_size, num_subtasks, h_dim)

        # z_mu, z_log_var = self.proj_z_mu(r), self.proj_z_logvar(r)
        # z_dist = Normal(z_mu, torch.exp(0.5 * z_log_var))

        z_mu = self.proj_z_mu(r)
        z_sigma = softplus(torch.clamp(self.proj_z_sigma(r), min=1e-6, max=1e2))
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


class BCALNP(LNP):
    def __init__(
        self,
        encoder: BCAEncoder,
        decoder: Decoder,
    ):
        super(BCALNP, self).__init__(encoder, decoder)

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
