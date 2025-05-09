from abc import ABC, abstractmethod
from typing import Tuple

import torch
from torch import Tensor, nn
from torch.distributions.distribution import Distribution
from torch.distributions.normal import Normal
from torch.nn.functional import softplus

from dviforbml.architectures.np import NP
from dviforbml.components.decoder.decoder import Decoder
from dviforbml.components.decoder.decoder_times_prior import DecoderTimesPrior
from dviforbml.components.encoder.abstract_encoder import AbstractEncoder
from dviforbml.components.encoder.aggr_encoder import AggrEncoder
from dviforbml.components.encoder.bca_encoder import BCAEncoder


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

        input_dim = encoder.h_dim + (
            encoder.z_dim if encoder.max_context_size is not None else 0
        )

        self.proj_z_mu = nn.Linear(
            input_dim,
            decoder.z_dim,
        )

        self.proj_z_sigma = nn.Linear(
            input_dim,
            decoder.z_dim,
        )

    def encode(self, context: Tensor, mask: Tensor | None) -> Tuple[Normal, Tensor]:
        # (batch_size, num_subtasks, data_size, c_dim)
        # (batch_size, num_subtasks, data_size)

        r, s = self.encoder(context, mask)
        # (batch_size, num_subtasks, h_dim)
        # (batch_size, num_subtasks, z_dim)

        input = torch.cat([r, s], dim=-1) if s is not None else r
        # (batch_size, num_subtasks, h_dim + z_dim)

        z_mu = self.proj_z_mu(input)
        z_sigma = softplus(torch.clamp(self.proj_z_sigma(input), min=1e-6, max=1e2))
        # z_sigma = softplus(self.proj_z_sigma(input)) + 1e-6
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

        (z_mu, z_var), _ = self.encoder(context, mask)

        if torch.any(z_var < 0):
            raise ValueError("Negative variance detected")

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
