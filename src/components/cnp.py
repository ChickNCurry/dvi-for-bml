from torch import Tensor, nn
from torch.distributions.normal import Normal

from components.encoder.bca_encoder import BCAEncoder
from src.components.decoder.decoder import Decoder
from src.components.encoder.aggr_encoder import AggrEncoder


class AGGRCNP(nn.Module):
    def __init__(self, encoder: AggrEncoder, decoder: Decoder):
        super(AGGRCNP, self).__init__()

        self.encoder = encoder

        self.proj_r = nn.Linear(encoder.h_dim, decoder.z_dim)

        self.decoder = decoder

    def forward(self, context: Tensor, mask: Tensor | None, x_target: Tensor) -> Normal:
        # (batch_size, num_subtasks, context_size, c_dim)
        # (batch_size, num_subtasks, context_size)
        # (batch_size, num_subtasks, target_size, x_dim)

        r = self.encoder(context, mask)
        # (batch_size, num_subtasks, h_dim)

        r = self.proj_r(r)
        # (batch_size, num_subtasks, z_dim)

        y_dist = self.decoder(r, x_target)
        # (batch_size, num_subtasks, target_size, y_dim)

        return y_dist


class BCACNP(nn.Module):
    def __init__(self, encoder: BCAEncoder, decoder: Decoder):
        super(BCACNP, self).__init__()

        self.encoder = encoder

        self.proj_r = nn.Linear(encoder.h_dim, decoder.z_dim)

        self.decoder = decoder

    def forward(self, context: Tensor, mask: Tensor | None, x_target: Tensor) -> Normal:
        # (batch_size, num_subtasks, context_size, c_dim)
        # (batch_size, num_subtasks, context_size)
        # (batch_size, num_subtasks, target_size, x_dim)

        z_mu, z_var = self.encoder(context, mask)
        # (batch_size, num_subtasks, h_dim)

        r = self.proj_r(z_mu + z_var)
        # (batch_size, num_subtasks, z_dim)

        y_dist = self.decoder(r, x_target)
        # (batch_size, num_subtasks, target_size, y_dim)

        return y_dist


def loss(y_dist: Normal):
    # (batch_size, num_subtasks, target_size, y_dim)

    
