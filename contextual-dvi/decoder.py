import torch
from torch import Tensor, nn
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
        has_det_path: bool,
    ) -> None:
        super(Decoder, self).__init__()

        self.has_det_path = has_det_path

        self.mlp = nn.Sequential(
            nn.Linear(x_dim + z_dim + h_dim if has_det_path else x_dim + z_dim, h_dim),
            *[
                layer
                for layer in (getattr(nn, non_linearity)(), nn.Linear(h_dim, h_dim))
                for _ in range(num_layers - 1)
            ]
        )

        self.proj_y_mu = nn.Linear(h_dim, y_dim)
        self.proj_y_w = nn.Linear(h_dim, y_dim)

    def forward(
        self,
        x_target: Tensor,
        z: Tensor,
        mask: Tensor | None,
        context_embedding: Tensor | None,
    ) -> Distribution:
        # (batch_size, target_size, x_dim)
        # (batch_size, z_dim)
        # (batch_size, target_size)
        # (batch_size, h_dim)

        z = z.unsqueeze(1).expand(-1, x_target.shape[1], -1)
        # (batch_size, target_size, z_dim)

        context_embedding = (
            context_embedding.unsqueeze(1).expand(-1, x_target.shape[1], -1)
            if self.has_det_path and context_embedding is not None
            else None
        )
        # (batch_size, target_size, z_dim)

        h = (
            torch.cat([x_target, z, context_embedding], dim=-1)
            if self.has_det_path and context_embedding is not None
            else torch.cat([x_target, z], dim=-1)
        )
        # (batch_size, target_size, x_dim + z_dim + h_dim)

        h = self.mlp(h)
        # (batch_size, target_size, h_dim)

        y_mu = self.proj_y_mu(h)
        y_std = 0.1 + 0.9 * nn.Softplus()(self.proj_y_w(h))
        # (batch_size, target_size, y_dim)

        y_dist = Normal(y_mu, y_std)  # type: ignore

        return y_dist
