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
        self.proj_y_sigma = nn.Linear(h_dim, y_dim)

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
        y_sigma = softplus(torch.clamp(self.proj_y_sigma(h), min=1e-6, max=1e2))
        # (batch_size, num_subtasks, data_size, y_dim)

        return Normal(y_mu, y_sigma)  # type: ignore

    def get_mse(self, z: Tensor, x: Tensor, y: Tensor, mask: Tensor) -> Tensor:
        # (batch_size, num_subtasks, z_dim)
        # (batch_size, num_subtasks, data_size, x_dim)
        # (batch_size, num_subtasks, data_size, y_dim)
        # (batch_size, num_subtasks, data_size)

        y_dist: Normal = self(z, x)
        y_pred: Tensor = y_dist.mean
        # (batch_size, num_subtasks, data_size, y_dim)

        mse: Tensor = ((y_pred - y) ** 2).sum(3)
        # (batch_size, num_subtasks, data_size)

        if mask is not None:
            mse = mse * mask

        mse = mse.mean(2, keepdim=True)
        # (batch_size, num_subtasks, 1)

        return mse

    def get_log_like(self, z: Tensor, x: Tensor, y: Tensor, mask: Tensor) -> Tensor:
        # (batch_size, num_subtasks, z_dim)
        # (batch_size, num_subtasks, data_size, x_dim)
        # (batch_size, num_subtasks, data_size, y_dim)
        # (batch_size, num_subtasks, data_size)

        y_dist: Normal = self(z, x)
        # (batch_size, num_subtasks, data_size, y_dim)

        log_like: Tensor = y_dist.log_prob(y).sum(3)  # type: ignore
        # (batch_size, num_subtasks, data_size)

        if mask is not None:
            log_like = log_like * mask

        log_like = log_like.sum(2, keepdim=True)
        # (batch_size, num_subtasks, 1)

        return log_like
