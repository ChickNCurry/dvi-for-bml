from torch import Tensor
from torch.distributions.normal import Normal

from components.decoder.decoder_times_prior import DecoderTimesPrior
from src.architectures.np import NP
from src.components.cdvi.cdvi import CDVI
from src.components.decoder.decoder import Decoder
from src.components.encoder.base_encoder import BaseEncoder


class DVINP(NP):
    def __init__(
        self,
        encoder: BaseEncoder,
        cdvi: CDVI,
        decoder: Decoder,
    ) -> None:
        super(DVINP).__init__()

        self.encoder = encoder
        self.cdvi = cdvi
        self.decoder = decoder

    def freeze(self, only_decoder: bool) -> None:
        if only_decoder:
            for param in self.decoder.parameters():
                param.requires_grad = False
            for param in self.encoder.parameters():
                param.requires_grad = True
            for param in self.cdvi.parameters():
                param.requires_grad = True
        else:
            for param in self.decoder.parameters():
                param.requires_grad = True
            for param in self.encoder.parameters():
                param.requires_grad = False
            for param in self.cdvi.parameters():
                param.requires_grad = False

    def inference(
        self, x_context: Tensor, y_context: Tensor, context: Tensor, x_data: Tensor
    ) -> Normal:
        r = self.encoder(context, None)

        target = DecoderTimesPrior(
            decoder=self.decoder,
            x=x_context,
            y=y_context,
            mask=None,
        )

        _, zs = self.cdvi.run_forward_process(target, r, None, None)

        assert zs is not None

        y_dist_data: Normal = self.decoder(zs[-1], x_data)

        return y_dist_data
