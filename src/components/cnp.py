from typing import Tuple

from torch import Tensor, nn
from torch.distributions.normal import Normal

from src.components.decoder.decoder import Decoder
from src.components.encoder.aggr_encoder import AggrEncoder
from src.components.encoder.bca_encoder import BCAEncoder


class AggrCNP(nn.Module):
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


class BcaCNP(nn.Module):
    def __init__(
        self,
        encoder: BCAEncoder,
        decoder: Decoder,
    ):
        super(BcaCNP, self).__init__()

        self.encoder = encoder
        self.proj_r = nn.Linear(encoder.h_dim, decoder.z_dim)
        self.decoder = decoder

    def forward(self, context: Tensor, mask: Tensor | None, x: Tensor) -> Normal:
        # (batch_size, num_subtasks, data_size, c_dim)
        # (batch_size, num_subtasks, data_size)
        # (batch_size, num_subtasks, data_size, x_dim)

        z_mu, z_var = self.encoder(context, mask)
        # (batch_size, num_subtasks, h_dim)

        r = self.proj_r(z_mu + z_var)
        # (batch_size, num_subtasks, z_dim)

        y_dist = self.decoder(r, x)
        # (batch_size, num_subtasks, data_size, y_dim)

        return y_dist
