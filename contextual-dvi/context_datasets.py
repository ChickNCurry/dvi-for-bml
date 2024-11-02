import random

import torch
from torch import Tensor
from torch.utils.data import Dataset


class ContextTestDataset(Dataset[Tensor]):
    def __init__(
        self,
        size: int,
        c_dim: int,
        sampling_factor: float,
    ) -> None:
        super(ContextTestDataset, self).__init__()

        self.size = size
        self.c_dim = c_dim
        self.sampling_factor = sampling_factor

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> Tensor:
        context = self.sampling_factor * torch.ones((1, self.c_dim))
        return context


class ContextSetDataset(Dataset[Tensor]):
    def __init__(
        self,
        size: int,
        c_dim: int,
        sampling_factor: float,
        max_context_size: int,
    ) -> None:
        super(ContextSetDataset, self).__init__()

        self.size = size
        self.c_dim = c_dim
        self.max_context_size = max_context_size
        self.sampling_factor = sampling_factor

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> Tensor:
        context = self.sampling_factor * torch.rand((self.max_context_size, self.c_dim))

        choices = [random.choice([1, -1]) for _ in range(self.c_dim)]
        for i in range(self.c_dim):
            context[:, i] = context[:, i] * choices[i]

        return context
