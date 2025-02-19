import torch
from torch import Tensor, nn
from torch.distributions import Distribution, Normal
from torch.nn.functional import softplus


class Decoder(nn.Module):
    def __init__(
        self,
        x_dim: int,
        z_dim: int,
        h_dim: int,
        y_dim: int,
        num_layers: int,
        non_linearity: str,
    ) -> None:
        super(Decoder, self).__init__()

        self.z_dim = z_dim

        self.mlp = nn.Sequential(
            nn.Linear(x_dim + z_dim, h_dim),
            *[
                layer
                for layer in (getattr(nn, non_linearity)(), nn.Linear(h_dim, h_dim))
                for _ in range(num_layers - 2)
            ],
            getattr(nn, non_linearity)(),
        )

        self.proj_y_mu = nn.Linear(h_dim, y_dim)
        self.proj_y_var = nn.Linear(h_dim, y_dim)

    def forward(
        self,
        z: Tensor,
        x: Tensor,
    ) -> Distribution:
        # (batch_size, num_subtasks, z_dim)
        # (batch_size, num_subtasks, data_size, x_dim)

        z = z.unsqueeze(2).expand(-1, -1, x.shape[2], -1)
        # (batch_size, num_subtasks, data_size, z_dim)

        input = torch.cat([x, z], dim=-1)
        # (batch_size, num_subtasks, data_size, x_dim + z_dim)

        h = self.mlp(input)
        # (batch_size, num_subtasks, data_size, h_dim)

        y_mu = self.proj_y_mu(h)
        y_sigma = softplus(self.proj_y_var(h))
        # y_sigma = torch.exp(0.5 * self.proj_y_logvar(h))
        # (batch_size, num_subtasks, data_size, y_dim)

        return Normal(y_mu, y_sigma)  # type: ignore
