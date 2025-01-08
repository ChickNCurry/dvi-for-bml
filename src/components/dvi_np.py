from typing import Callable

from torch import Tensor, nn
from torch.distributions import Distribution

from src.components.cdvi.cdvi import CDVI
from src.components.decoder.decoder import Decoder
from src.components.encoder.base_encoder import BaseEncoder


class DVINP(nn.Module):
    def __init__(
        self,
        encoder: BaseEncoder,
        cdvi: CDVI,
        decoder: Decoder | None,
        contextual_target: Callable[[Tensor, Tensor | None], Distribution] | None,
    ) -> None:
        super().__init__()

        self.encoder = encoder
        self.cdvi = cdvi
        self.decoder = decoder
        self.contextual_target = contextual_target

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
