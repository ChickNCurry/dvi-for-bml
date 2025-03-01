import hashlib
import random
from typing import Any, Tuple

import numpy as np
import torch
from metalearning_benchmarks import MetaLearningBenchmark  # type: ignore
from torch import Tensor
from torch.utils.data import Dataset
from metalearning_benchmarks.parametric_benchmark import ParametricBenchmark  # type: ignore


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
        generator: torch.Generator | None = None,
    ) -> None:
        self.benchmark = benchmark
        self.max_context_size = max_context_size
        self.generator = generator

        assert self.max_context_size <= self.benchmark.n_datapoints_per_task

    def __len__(self) -> int:
        return self.benchmark.n_task  # type: ignore

    def __getitem__(self, task_idx: int) -> Tuple[Tensor, Tensor]:
        task = self.benchmark.get_task_by_index(task_index=task_idx)

        x_data = Tensor(task.x)
        y_data = Tensor(task.y)

        assert x_data.shape[0] == y_data.shape[0]
        perm = torch.randperm(x_data.shape[0], generator=self.generator)

        x_data = x_data[perm]
        y_data = y_data[perm]

        # print(hash_tensor(x_context), hash_tensor(y_context))

        return x_data, y_data


def hash_tensor(t: Tensor) -> str:
    tensor_bytes = t.numpy().tobytes()
    return hashlib.sha256(tensor_bytes).hexdigest()


class Sinusoid1DFreq(ParametricBenchmark):  # type: ignore
    # cf. MAML paper, section "5.1. Regression"
    d_param = 1
    d_x = 1
    d_y = 1
    is_dynamical_system = False

    freq_bounds = np.array([0.1, 3])
    param_bounds = np.array([freq_bounds])
    x_bounds = np.array([[-5.0, 5.0]])

    def __init__(
        self,
        n_task: int,
        n_datapoints_per_task: int,
        output_noise: float,
        seed_task: int,
        seed_x: int,
        seed_noise: float,
    ):
        super().__init__(
            n_task=n_task,
            n_datapoints_per_task=n_datapoints_per_task,
            output_noise=output_noise,
            seed_task=seed_task,
            seed_x=seed_x,
            seed_noise=seed_noise,
        )

    def __call__(self, x: np.ndarray, param: np.ndarray) -> np.ndarray[Any, Any]:  # type: ignore
        freq = param
        y: np.ndarray[Any, Any] = np.sin(freq * x)
        return y
