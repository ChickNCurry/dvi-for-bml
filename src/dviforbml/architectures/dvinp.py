from typing import Tuple

import torch
from torch import Tensor
from torch.distributions.distribution import Distribution
from torch.distributions.normal import Normal

from dviforbml.architectures.np import NP
from dviforbml.components.cdvi.cdvi import CDVI
from dviforbml.components.decoder.decoder import Decoder
from dviforbml.components.decoder.decoder_times_prior import DecoderTimesPrior
from dviforbml.components.encoder.abstract_encoder import AbstractEncoder


class DVINP(NP):
    def __init__(
        self,
        encoder: AbstractEncoder,
        cdvi: CDVI,
        decoder: Decoder,
    ) -> None:
        super(DVINP, self).__init__(encoder, decoder)

        self.cdvi = cdvi

    def inference(
        self, x_context: Tensor, y_context: Tensor, mask: Tensor | None, x_data: Tensor
    ) -> Tuple[Normal, Tensor | None]:
        context = torch.cat([x_context, y_context], dim=-1)
        r_context, s_context = self.encoder(context, mask)

        target_dist = self.get_target_dist(x_context, y_context, mask)

        _, zs = self.cdvi.run_forward_process(
            target_dist, r_context, mask, s_context, None
        )

        assert zs is not None

        y_dist_data: Normal = self.decoder(zs[-1], x_data)

        return y_dist_data, zs[-1]

    def get_target_dist(
        self, x_context: Tensor, y_context: Tensor, mask: Tensor | None
    ) -> Distribution | None:
        return DecoderTimesPrior(
            decoder=self.decoder,
            x=x_context,
            y=y_context,
            mask=mask,
        )

    def freeze_cdvi(self) -> None:
        for param in self.cdvi.parameters():
            param.requires_grad = False

    def unfreeze_cdvi(self) -> None:
        for param in self.cdvi.parameters():
            param.requires_grad = True
