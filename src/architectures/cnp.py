from abc import ABC, abstractmethod
from typing import Tuple

import torch
from torch import Tensor, nn
from torch.distributions.normal import Normal

from src.architectures.np import NP
from src.components.decoder.decoder import Decoder
from src.components.encoder.aggr_encoder import AggrEncoder
from src.components.encoder.bca_encoder import BCAEncoder


class CNP(NP, ABC):
    def __init__(self) -> None:
        super(CNP, self).__init__()

    @abstractmethod
    def forward(self, context: Tensor, mask: Tensor | None, x: Tensor) -> Normal:
        raise NotImplementedError

    def inference(
        self, x_context: Tensor, y_context: Tensor, mask: Tensor | None, x_data: Tensor
    ) -> Tuple[Normal, Tensor | None]:
        context = torch.cat([x_context, y_context], dim=-1)

        y_dist: Normal = self(context, mask, x_data)

        return y_dist, None


class AggrCNP(CNP):
    def __init__(
        self,
        encoder: AggrEncoder,
        decoder: Decoder,
    ):
        super(AggrCNP, self).__init__()

        self.encoder = encoder
        self.proj_r = nn.Linear(encoder.h_dim, decoder.z_dim)
        self.decoder = decoder

    def forward(self, context: Tensor, mask: Tensor | None, x: Tensor) -> Normal:
        # (batch_size, num_subtasks, data_size, c_dim)
        # (batch_size, num_subtasks, data_size)
        # (batch_size, num_subtasks, data_size, x_dim)

        r = self.encoder(context, mask)
        # (batch_size, num_subtasks, h_dim)

        r = self.proj_r(r)
        # (batch_size, num_subtasks, z_dim)

        y_dist = self.decoder(r, x)
        # (batch_size, num_subtasks, data_size, y_dim)

        return y_dist


class BcaCNP(CNP):
    def __init__(
        self,
        encoder: BCAEncoder,
        decoder: Decoder,
    ):
        super(BcaCNP, self).__init__()

        self.encoder = encoder
        self.proj_r = nn.Linear(encoder.z_dim * 2, decoder.z_dim)
        self.decoder = decoder

    def forward(self, context: Tensor, mask: Tensor | None, x: Tensor) -> Normal:
        # (batch_size, num_subtasks, data_size, c_dim)
        # (batch_size, num_subtasks, data_size)
        # (batch_size, num_subtasks, data_size, x_dim)

        z_mu, z_var = self.encoder(context, mask)
        # (batch_size, num_subtasks, z_dim)

        input = torch.cat([z_mu, z_var], dim=-1)
        # (batch_size, num_subtasks, 2 * z_dim)

        r = self.proj_r(input)
        # (batch_size, num_subtasks, z_dim)

        y_dist = self.decoder(r, x)
        # (batch_size, num_subtasks, data_size, y_dim)

        return y_dist
