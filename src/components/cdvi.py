from typing import Callable, Tuple

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch import Tensor, nn
from torch.distributions import Distribution
from torch.optim import Optimizer
from torch.optim.adamw import AdamW
from torch.utils.data import DataLoader, random_split

from src.components.control import Control
from src.components.decoder import Decoder
from src.components.cdvi_process import CDVIProcess
from src.components.encoder import Encoder, SetEncoder
from src.components.hyper_net import HyperNet
from src.utils.datasets import MetaLearningDataset


class CDVI(nn.Module):
    def __init__(
        self,
        encoder: Encoder,
        dvi_process: CDVIProcess,
        decoder: Decoder | None,
        contextual_target: Callable[[Tensor, Tensor | None], Distribution] | None,
    ) -> None:
        super().__init__()

        self.encoder = encoder
        self.dvi_process = dvi_process
        self.decoder = decoder
        self.contextual_target = contextual_target

    def freeze(self, only_decoder: bool) -> None:
        if only_decoder:
            for param in self.decoder.parameters():
                param.requires_grad = False
            for param in self.encoder.parameters():
                param.requires_grad = True
            for param in self.dvi_process.parameters():
                param.requires_grad = True
        else:
            for param in self.decoder.parameters():
                param.requires_grad = True
            for param in self.encoder.parameters():
                param.requires_grad = False
            for param in self.dvi_process.parameters():
                param.requires_grad = False
