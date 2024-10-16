import math

import numpy as np
import torch
from dvi_process import zTuple
from torch import Tensor
from torch.distributions import Normal
from torch.utils.data import Dataset


def contextual_gaussian(context: float, dim: int) -> Normal:
    mu = 5 * math.sin(context) * torch.ones(dim)
    std = torch.ones(dim)
    return Normal(mu, std)  # type: ignore


def contextual_gaussian_tuple(context: Tensor) -> zTuple:
    mu = 5 * torch.sin(context)
    std = torch.ones_like(mu, device=context.device)
    return zTuple(torch.normal(mu, std), mu, std)


class ContextualGaussianDataset(Dataset[Tensor]):
    def __init__(self, size: int) -> None:
        super(ContextualGaussianDataset, self).__init__()
        self.contexts = np.ones(size) * math.pi / 2

        # np.arange(0, 2 * math.pi, 2 * math.pi / size)

    def __len__(self) -> int:
        return len(self.contexts)

    def __getitem__(self, idx: int) -> Tensor:
        return torch.tensor([self.contexts[idx]]).float()
