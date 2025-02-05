from torch import Tensor, nn
import torch
from torch.distributions import Distribution, Normal


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

        self.proj_x = nn.Linear(x_dim, h_dim)
        self.proj_z = nn.Linear(z_dim, h_dim)

        self.mlp = nn.Sequential(
            *[
                layer
                for layer in (getattr(nn, non_linearity)(), nn.Linear(h_dim, h_dim))
                for _ in range(num_layers - 2)
            ],
            getattr(nn, non_linearity)(),
        )

        self.proj_y_mu = nn.Linear(h_dim, y_dim)
        self.proj_y_logvar = nn.Linear(h_dim, y_dim)

    def forward(
        self,
        z: Tensor,
        x_target: Tensor,
    ) -> Distribution:
        # (batch_size, num_subtasks, z_dim)
        # (batch_size, num_subtasks, target_size, x_dim)
        # (batch_size, num_subtasks, context_size, x_dim)

        z = z.unsqueeze(2).expand(-1, -1, x_target.shape[2], -1)
        # (batch_size, num_subtasks, target_size, z_dim)

        h = self.mlp(self.proj_x(x_target) + self.proj_z(z))
        # (batch_size, num_subtasks, target_size, h_dim)

        y_mu = self.proj_y_mu(h)
        # y_sigma = 0.1 + 0.9 * nn.functional.softplus(self.proj_y_w(h))
        y_sigma = torch.exp(0.5 * self.proj_y_logvar(h))
        # (batch_size, num_subtasks, target_size, y_dim)

        return Normal(y_mu, y_sigma)  # type: ignore
