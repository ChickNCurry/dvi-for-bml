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
        has_lat_path: bool,
        has_det_path: bool,
        is_cross_attentive: bool,
        num_heads: int,
    ) -> None:
        super(Decoder, self).__init__()

        self.z_dim = z_dim
        self.has_lat_path = has_lat_path
        self.has_det_path = has_det_path
        self.is_cross_attentive = is_cross_attentive
        self.num_heads = num_heads

        if self.is_cross_attentive:
            self.mlp_query = nn.Sequential(
                nn.Linear(x_dim, h_dim),
                getattr(nn, non_linearity)(),
                nn.Linear(h_dim, h_dim),
            )
            self.mlp_key = nn.Sequential(
                nn.Linear(x_dim, h_dim),
                getattr(nn, non_linearity)(),
                nn.Linear(h_dim, h_dim),
            )
            self.cross_attn = nn.MultiheadAttention(
                h_dim, num_heads=self.num_heads, batch_first=True
            )

        self.mlp = nn.Sequential(
            nn.Linear(
                x_dim + (z_dim if has_lat_path else 0) + (h_dim if has_det_path else 0),
                h_dim,
            ),
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
        z: Tensor | None,
        context_emb: Tensor | None,
        mask: Tensor | None,
    ) -> Distribution:
        # (batch_size, num_subtasks, target_size, x_dim)
        # (batch_size, num_subtasks, z_dim)
        # (batch_size, num_subtasks, h_dim) or (batch_size, num_subtasks, context_size, h_dim)
        # (batch_size, num_subtasks, context_size)

        c: Tensor | None = None

        if self.has_det_path and context_emb is not None:
            if self.is_cross_attentive:
                mask = mask.unsqueeze(2) if mask is not None else None
                # (batch_size, num_subtasks, 1, context_size)

                query = self.mlp_query(x_target)
                # (batch_size, num_subtasks, target_size, h_dim)

                key = self.mlp_key(
                    context_emb[:, :, :, 0:1]
                )  # (batch_size, num_subtasks, context_size, h_dim)

                value = context_emb
                # (batch_size, num_subtasks, context_size, h_dim)

                c, _ = self.cross_attn(
                    query=query, key=key, value=value, attn_mask=mask
                )
            else:
                c = context_emb.unsqueeze(2).expand(-1, -1, x_target.shape[1], -1)
            # (batch_size, num_subtasks, target_size, h_dim)

        if self.has_lat_path and z is not None:
            z = z.unsqueeze(2).expand(-1, -1, x_target.shape[2], -1)
            # (batch_size, num_subtasks, target_size, z_dim)

        h = torch.cat([t for t in [x_target, c, z] if t is not None], dim=-1)
        # (batch_size, num_subtasks, target_size, x_dim + z_dim + h_dim)

        h = self.mlp(h)
        # (batch_size, num_subtasks, target_size, h_dim)

        y_mu = self.proj_y_mu(h)
        y_std = 0.1 + 0.9 * nn.Softplus()(self.proj_y_w(h))
        # (batch_size, num_subtasks, target_size, y_dim)

        y_dist = Normal(y_mu, y_std)  # type: ignore

        return y_dist


class LikelihoodTimesPrior(Distribution, nn.Module):
    def __init__(
        self,
        decoder: Decoder,
        x_target: Tensor,
        y_target: Tensor,
        context_emb: Tensor | None,
        mask: Tensor | None,
    ) -> None:
        super(LikelihoodTimesPrior, self).__init__(validate_args=False)

        device = x_target.device
        z_dim = decoder.z_dim
        batch_size = x_target.shape[0]
        num_subtasks = x_target.shape[1]

        self.prior = Normal(  # type: ignore
            torch.zeros((batch_size, num_subtasks, z_dim), device=device),
            torch.ones((batch_size, num_subtasks, z_dim), device=device),
        )

        self.decoder = decoder
        self.x_target = x_target
        self.y_target = y_target
        self.context_emb = context_emb
        self.mask = mask

    def mse(
        self,
        z: Tensor,
        x_data: Tensor,
        y_data: Tensor,
    ) -> Tensor:
        # (batch_size, num_subtasks, z_dim)

        y_pred: Tensor = self.decoder(x_data, z, self.context_emb, self.mask).mean
        # (batch_size, num_subtasks, target_size, y_dim)

        mse: Tensor = ((y_pred - y_data) ** 2).sum(dim=2).sum(dim=-1, keepdim=True)
        # (batch_size, num_subtasks, 1)

        return mse

    def log_like(
        self,
        z: Tensor,
        x_target: Tensor,
        y_target: Tensor,
        mask: Tensor | None,
    ) -> Tensor:
        log_like: Tensor = self.decoder(
            x_target, z, self.context_emb, self.mask
        ).log_prob(y_target)
        # (batch_size, num_subtasks, target_size, y_dim)

        if mask is not None:
            log_like = log_like * mask.unsqueeze(-1).expand(
                -1, -1, -1, log_like.shape[-1]
            )  # (batch_size, num_subtasks, target_size, y_dim)

        log_like = log_like.sum(dim=2).sum(dim=-1, keepdim=True)
        # (batch_size, num_subtasks, 1)

        return log_like

    def log_prob(self, z: Tensor) -> Tensor:
        # (batch_size, num_subtasks, z_dim)

        log_likelihood: Tensor = self.log_like(
            z, self.x_target, self.y_target, self.mask
        )  # (batch_size, num_subtasks, 1)

        log_prior: Tensor = self.prior.log_prob(z).sum(dim=2, keepdim=True)  # type: ignore
        # (batch_size, num_subtasks, 1)

        return log_likelihood + log_prior
