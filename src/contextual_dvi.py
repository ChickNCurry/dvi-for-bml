from torch import nn

from src.decoder import Decoder
from src.dvi_process import DiffusionVIProcess
from src.encoder import Encoder


class ContextualDVI(nn.Module):
    def __init__(
        self,
        encoder: Encoder,
        dvi_process: DiffusionVIProcess,
        decoder: Decoder | None,
    ) -> None:
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.dvi_process = dvi_process
