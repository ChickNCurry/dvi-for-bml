from torch import nn

from src.components.cdvi.cdvi import CDVI
from src.components.decoder.decoder import Decoder
from src.components.encoder.base_encoder import BaseEncoder


class DVINP(nn.Module):
    def __init__(
        self,
        encoder: BaseEncoder,
        cdvi: CDVI,
        decoder: Decoder,
    ) -> None:
        super().__init__()

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


# class AggrDVINP(DVINP):
#     def __init__(
#         self,
#         encoder: BaseEncoder,
#         cdvi: CDVI,
#         decoder: Decoder | None,
#     ) -> None:
#         super().__init__(encoder, cdvi, decoder)

#     def forward(
#         self, context: Tensor, mask: Tensor | None, x: Tensor, y: Tensor
#     ) -> Tensor:
#         # (batch_size, num_subtasks, data_size, c_dim)
#         # (batch_size, num_subtasks, data_size)
#         # (batch_size, num_subtasks, data_size, x_dim)
#         # (batch_size, num_subtasks, data_size, y_dim)

#         r = self.encoder(context, mask)
#         # (batch_size, num_subtasks, h_dim)

#         target = DecoderTimesPrior(decoder=self.decoder, x=x, y=y, mask=mask)

#         elbo, _, z = self.cdvi.run_chain(target, r, mask)
#         # (1)
#         # (num_steps, batch_size, num_subtasks, z_dim)
