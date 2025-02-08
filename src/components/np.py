from torch import Tensor, nn
import torch
from torch.distributions.normal import Normal

from components.encoder.bca_encoder import BCAEncoder
from src.components.decoder.decoder import Decoder
from src.components.encoder.aggr_encoder import AggrEncoder


class AGGRNP(nn.Module):
    def __init__(self, encoder: AggrEncoder, decoder: Decoder):
        super(AGGRNP, self).__init__()

        self.encoder = encoder

        self.proj_z_mu = nn.Linear(encoder.h_dim, decoder.z_dim)
        self.proj_z_logvar = nn.Linear(encoder.h_dim, decoder.z_dim)

        self.decoder = decoder

    def forward(self, context: Tensor, mask: Tensor | None, x_target: Tensor) -> Normal:
        # (batch_size, num_subtasks, context_size, c_dim)
        # (batch_size, num_subtasks, context_size)
        # (batch_size, num_subtasks, target_size, x_dim)

        r = self.encoder(context, mask)
        # (batch_size, num_subtasks, h_dim)

        z_mu, z_log_var = self.proj_z_mu(r), self.proj_z_logvar(r)
        z = Normal(z_mu, torch.exp(0.5 * z_log_var)).rsample()
        # (batch_size, num_subtasks, z_dim)

        y_dist = self.decoder(z, x_target)
        # (batch_size, num_subtasks, target_size, y_dim)

        return y_dist


class BCANP(nn.Module):
    def __init__(self, encoder: BCAEncoder, decoder: Decoder):
        super(BCANP, self).__init__()

        self.encoder = encoder

        self.proj_z_mu = nn.Linear(encoder.h_dim, decoder.z_dim)
        self.proj_z_logvar = nn.Linear(encoder.h_dim, decoder.z_dim)

        self.decoder = decoder

    def forward(self, context: Tensor, mask: Tensor | None, x_target: Tensor) -> Normal:
        # (batch_size, num_subtasks, context_size, c_dim)
        # (batch_size, num_subtasks, context_size)
        # (batch_size, num_subtasks, target_size, x_dim)

        z_mu, z_var = self.encoder(context, mask)
        # (batch_size, num_subtasks, h_dim)

        z_mu, z_log_var = self.proj_z_mu(z_mu), self.proj_z_logvar(z_var)
        z = Normal(z_mu, torch.exp(0.5 * z_log_var)).rsample()
        # (batch_size, num_subtasks, z_dim)

        y_dist = self.decoder(z, x_target)
        # (batch_size, num_subtasks, target_size, y_dim)

        return y_dist
