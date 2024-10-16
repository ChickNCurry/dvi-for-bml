import math

import numpy as np
import torch
from torch import Tensor
from torch.distributions import Normal
from torch.utils.data import Dataset


def contextual_gaussian(context: Tensor) -> Normal:
    mu = 5 * torch.sin(context)
    std = torch.ones_like(mu, device=context.device)
    return Normal(mu, std)  # type: ignore


class ContextualGaussianDataset(Dataset[Tensor]):
    def __init__(self, size: int) -> None:
        super(ContextualGaussianDataset, self).__init__()

        self.contexts = np.arange(0, 2 * math.pi, 2 * math.pi / size)
        # self.contexts = np.ones(size) * math.pi / 2

    def __len__(self) -> int:
        return len(self.contexts)

    def __getitem__(self, idx: int) -> Tensor:
        return torch.tensor([self.contexts[idx]]).float()
