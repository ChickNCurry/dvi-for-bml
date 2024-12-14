import numpy as np
import torch
from torch import Tensor, nn
from torch.distributions import Distribution, Normal


class Decoder(nn.Module):
    def __init__(
        self,
        x_dim: int,
        z_dim: int | None,
        h_dim: int,
        y_dim: int,
        num_layers: int,
        non_linearity: str,
        has_lat_path: bool,
        has_det_path: bool,
        is_cross_attentive: bool,
        num_heads: int | None,
    ) -> None:
        super(Decoder, self).__init__()

        self.z_dim = z_dim
        self.has_lat_path = has_lat_path
        self.has_det_path = has_det_path
        self.is_cross_attentive = is_cross_attentive
        self.num_heads = num_heads

        self.proj_x = nn.Linear(x_dim, h_dim)

        if self.has_lat_path:
            assert z_dim is not None
            self.proj_z = nn.Linear(z_dim, h_dim)

        if self.has_det_path and self.is_cross_attentive:
            assert self.num_heads is not None
            self.cross_attn = nn.MultiheadAttention(h_dim, num_heads, batch_first=True)

        self.mlp = nn.Sequential(
            *[
                layer
                for layer in (getattr(nn, non_linearity)(), nn.Linear(h_dim, h_dim))
                for _ in range(num_layers - 2)
            ],
            getattr(nn, non_linearity)(),
        )

        self.proj_y_mu = nn.Linear(h_dim, y_dim)
        self.proj_y_w = nn.Linear(h_dim, y_dim)

    def forward(
        self,
        z: Tensor | None,
        x_target: Tensor,
        x_context: Tensor | None,
        r_aggr: Tensor | None,
        r_non_aggr: Tensor | None,
        mask: Tensor | None,
    ) -> Distribution:
        # (batch_size, num_subtasks, z_dim)
        # (batch_size, num_subtasks, target_size, x_dim)
        # (batch_size, num_subtasks, context_size, x_dim)
        # (batch_size, num_subtasks, h_dim)
        # (batch_size, num_subtasks, context_size, h_dim)
        # (batch_size, num_subtasks, context_size)

        batch_size = x_target.shape[0]
        num_subtasks = x_target.shape[1]
        target_size = x_target.shape[2]

        r: Tensor | None = None

        if self.has_det_path:

            if self.is_cross_attentive:
                assert (
                    x_context is not None
                    and r_non_aggr is not None
                    and self.num_heads is not None
                )

                mask = (
                    mask.unsqueeze(2)
                    .expand(-1, -1, target_size, -1)
                    .view(batch_size * num_subtasks, -1, -1)
                    .repeat(self.num_heads, 1, 1)
                    if mask is not None
                    else None
                )  # (batch_size * num_subtasks * num_heads, target_size, context_size)

                query = self.proj_x(x_target.view(batch_size * num_subtasks, -1, -1))
                # (batch_size * num_subtasks, target_size, h_dim)

                key = self.proj_x(x_context.view(batch_size * num_subtasks, -1, -1))
                # (batch_size * num_subtasks, context_size, h_dim)

                value = r_non_aggr.view(batch_size * num_subtasks, -1, -1)
                # (batch_size * num_subtasks, context_size, h_dim)

                r, _ = self.cross_attn(
                    query=query, key=key, value=value, attn_mask=mask
                )
                # (batch_size * num_subtasks, target_size, h_dim)

                assert r is not None

                r = r.view(batch_size, num_subtasks, -1, -1)
                # (batch_size, num_subtasks, target_size, h_dim)

            else:
                assert r_aggr is not None

                r = r_aggr.unsqueeze(2).expand(-1, -1, target_size, -1)
                # (batch_size, num_subtasks, target_size, h_dim)

        if self.has_lat_path:
            assert z is not None

            z = z.unsqueeze(2).expand(-1, -1, target_size, -1)
            # (batch_size, num_subtasks, target_size, z_dim)

            z = self.proj_z(z)
            # (batch_size, num_subtasks, target_size, h_dim)

        x = self.proj_x(x_target)
        h = self.mlp(x + z if z is not None else 0 + r if r is not None else 0)
        # (batch_size, num_subtasks, target_size, h_dim)

        y_mu = self.proj_y_mu(h)
        y_sigma = 0.1 + 0.9 * nn.Softplus()(self.proj_y_w(h))
        # (batch_size, num_subtasks, target_size, y_dim)

        # y_mu = torch.nan_to_num(y_mu)
        # y_sigma = torch.nan_to_num(y_sigma)

        return Normal(y_mu, y_sigma)  # type: ignore


class DecoderTimesPrior(Distribution, nn.Module):
    def __init__(
        self,
        decoder: Decoder,
        x_target: Tensor,
        y_target: Tensor,
        x_context: Tensor | None,
        r_aggr: Tensor | None,
        r_non_aggr: Tensor | None,
        mask: Tensor | None,
    ) -> None:
        # (batch_size, num_subtasks, z_dim)
        # (batch_size, num_subtasks, target_size, x_dim)
        # (batch_size, num_subtasks, target_size, y_dim)
        # (batch_size, num_subtasks, context_size, x_dim)
        # (batch_size, num_subtasks, h_dim)
        # (batch_size, num_subtasks, context_size, h_dim)
        # (batch_size, num_subtasks, context_size)

        super(DecoderTimesPrior, self).__init__(validate_args=False)

        self.decoder = decoder
        self.x_target = x_target
        self.y_target = y_target
        self.x_context = x_context
        self.r_aggr = r_aggr
        self.r_non_aggr = r_non_aggr
        self.mask = mask

        device = x_target.device
        batch_size = x_target.shape[0]
        num_subtasks = x_target.shape[1]
        z_dim = decoder.z_dim

        assert z_dim is not None

        self.prior = Normal(  # type: ignore
            torch.zeros((batch_size, num_subtasks, z_dim), device=device),
            torch.ones((batch_size, num_subtasks, z_dim), device=device),
        )

    def mse(
        self,
        z: Tensor,
        x_data: Tensor,
        y_data: Tensor,
    ) -> Tensor:
        # (batch_size, num_subtasks, z_dim)
        # (batch_size, num_subtasks, target_size, x_dim)
        # (batch_size, num_subtasks, target_size, y_dim)

        y_pred: Tensor = self.decoder(
            z, x_data, self.x_context, self.r_aggr, self.r_non_aggr, self.mask
        ).mean
        # (batch_size, num_subtasks, target_size, y_dim)

        mse: Tensor = torch.median(((y_pred - y_data) ** 2).sum(3).sum(2).mean(1))
        # (1)

        return mse

    def lmpl(
        self,
        z: Tensor,
        x_data: Tensor,
        y_data: Tensor,
    ) -> Tensor:
        # (batch_size, num_subtasks, z_dim)
        # (batch_size, num_subtasks, target_size, x_dim)
        # (batch_size, num_subtasks, target_size, y_dim)

        lmpl: Tensor = self.decoder(
            z, x_data, self.x_context, self.r_aggr, self.r_non_aggr, self.mask
        ).log_prob(y_data)
        # (batch_size, num_subtasks, target_size, y_dim)

        lmpl = torch.median(
            torch.logsumexp(lmpl.sum(3).sum(2), dim=1) - np.log(x_data.shape[1])
        )  # (1)

        return lmpl

    def log_prob(self, z: Tensor) -> Tensor:
        # (batch_size, num_subtasks, z_dim)

        log_like: Tensor = self.decoder(
            z,
            self.x_target,
            self.x_context,
            self.r_aggr,
            self.r_non_aggr,
            self.mask,
        ).log_prob(self.y_target)
        # (batch_size, num_subtasks, target_size, y_dim)

        if self.mask is not None:
            mask = self.mask.unsqueeze(-1).expand(-1, -1, -1, log_like.shape[-1])
            # (batch_size, num_subtasks, target_size, y_dim)

            log_like = log_like * mask

        log_like = log_like.sum(dim=2).sum(dim=-1, keepdim=True)
        # (batch_size, num_subtasks, 1)

        log_prior: Tensor = self.prior.log_prob(z)  # type: ignore
        log_prior = log_prior.sum(dim=2, keepdim=True)
        # (batch_size, num_subtasks, 1)

        return log_like + log_prior
