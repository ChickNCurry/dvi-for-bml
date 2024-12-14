import random
from typing import Tuple

import torch
from metalearning_benchmarks import MetaLearningBenchmark  # type: ignore
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


class MetaLearningDataset(Dataset[Tuple[Tensor, Tensor]]):
    def __init__(
        self,
        benchmark: MetaLearningBenchmark,
        max_context_size: int,
    ) -> None:
        self.benchmark = benchmark
        self.max_context_size = max_context_size

        assert self.max_context_size <= self.benchmark.n_datapoints_per_task

    def __len__(self) -> int:
        return self.benchmark.n_task  # type: ignore

    def __getitem__(self, task_idx: int) -> Tuple[Tensor, Tensor]:
        task = self.benchmark.get_task_by_index(task_index=task_idx)

        x_context = Tensor(task.x)
        y_context = Tensor(task.y)

        assert x_context.shape[0] == y_context.shape[0]
        perm = torch.randperm(x_context.shape[0])

        x_context = x_context[perm]
        y_context = y_context[perm]

        return x_context, y_context
