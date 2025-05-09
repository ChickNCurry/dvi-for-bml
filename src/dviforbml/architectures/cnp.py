from abc import ABC, abstractmethod
from typing import Tuple

import torch
from torch import Tensor, nn
from torch.distributions.distribution import Distribution
from torch.distributions.normal import Normal

from dviforbml.architectures.np import NP
from dviforbml.components.decoder.decoder import Decoder
from dviforbml.components.encoder.abstract_encoder import AbstractEncoder
from dviforbml.components.encoder.aggr_encoder import AggrEncoder
from dviforbml.components.encoder.bca_encoder import BCAEncoder


class CNP(NP, ABC):
    def __init__(self, encoder: AbstractEncoder, decoder: Decoder) -> None:
        super(CNP, self).__init__(encoder, decoder)

    @abstractmethod
    def forward(self, context: Tensor, mask: Tensor | None, x: Tensor) -> Normal:
        raise NotImplementedError

    def inference(
        self, x_context: Tensor, y_context: Tensor, mask: Tensor | None, x_data: Tensor
    ) -> Tuple[Normal, Tensor | None]:
        context = torch.cat([x_context, y_context], dim=-1)

        y_dist: Normal = self(context, mask, x_data)

        return y_dist, None

    def get_target_dist(
        self, x_context: Tensor, y_context: Tensor, mask: Tensor | None
    ) -> Distribution | None:
        return None


class AggrCNP(CNP):
    def __init__(
        self,
        encoder: AggrEncoder,
        decoder: Decoder,
    ):
        super(AggrCNP, self).__init__(encoder, decoder)

        self.proj_z = nn.Linear(
            encoder.h_dim + encoder.z_dim
            if encoder.max_context_size is not None
            else encoder.h_dim,
            decoder.z_dim,
        )

    def forward(self, context: Tensor, mask: Tensor | None, x: Tensor) -> Normal:
        # (batch_size, num_subtasks, data_size, c_dim)
        # (batch_size, num_subtasks, data_size)
        # (batch_size, num_subtasks, data_size, x_dim)

        r, s = self.encoder(context, mask)
        # (batch_size, num_subtasks, h_dim)
        # (batch_size, num_subtasks, z_dim)

        input = torch.cat([r, s], dim=-1) if s is not None else r
        # (batch_size, num_subtasks, h_dim + z_dim)

        z = self.proj_z(input)
        # (batch_size, num_subtasks, z_dim)

        y_dist = self.decoder(z, x)
        # (batch_size, num_subtasks, data_size, y_dim)

        return y_dist


class BCACNP(CNP):
    def __init__(
        self,
        encoder: BCAEncoder,
        decoder: Decoder,
    ):
        super(BCACNP, self).__init__(encoder, decoder)

        self.proj_r = nn.Linear(
            encoder.bca_dim * 2 + encoder.z_dim
            if encoder.max_context_size is not None
            else encoder.bca_dim * 2,
            decoder.z_dim,
        )

    def forward(self, context: Tensor, mask: Tensor | None, x: Tensor) -> Normal:
        # (batch_size, num_subtasks, data_size, c_dim)
        # (batch_size, num_subtasks, data_size)
        # (batch_size, num_subtasks, data_size, x_dim)

        (z_mu, z_var), s = self.encoder(context, mask)
        # (batch_size, num_subtasks, bca_dim)
        # (batch_size, num_subtasks, z_dim)

        input = torch.cat([e for e in [z_mu, z_var, s] if e is not None], dim=-1)
        # (batch_size, num_subtasks, 2 * bca_dim + z_dim)

        z = self.proj_r(input)
        # (batch_size, num_subtasks, z_dim)

        y_dist = self.decoder(z, x)
        # (batch_size, num_subtasks, data_size, y_dim)

        return y_dist
