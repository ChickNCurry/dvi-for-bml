import itertools
import random
from typing import List, Tuple

import numpy as np
import torch
from torch import Size, Tensor
from torch.distributions import Distribution, Normal
from torch.utils.data import Dataset


class ContextualGaussian(Distribution):
    def __init__(self, context: Tensor, scale: float) -> None:
        # (batch_size, context_size, z_dim)

        assert context.shape[1] == 1

        super(ContextualGaussian, self).__init__()

        mu = torch.mean(context, dim=1)
        sigma = torch.ones_like(mu, device=mu.device) * scale
        # (batch_size, z_dim)

        self.gaussian = Normal(mu, sigma)  # type: ignore

    def sample(self, sample_shape: Size = torch.Size([])) -> Tensor:
        return self.gaussian.sample()  # type: ignore

    def log_prob(self, x: Tensor) -> Tensor:
        return self.gaussian.log_prob(x)  # type: ignore


class ContextualGMM(Distribution):
    def __init__(
        self,
        context: Tensor,
        offsets: Tuple[float, float] = (5, -5),
        scales: Tuple[float, float] = (1, 1),
        weights: Tuple[float, float] = (0.3, 0.7),
    ) -> None:
        # (batch_size, context_size, z_dim)

        assert context.shape[1] == 1

        super(ContextualGMM, self).__init__()

        self.batch_size = context.shape[0]

        mu = torch.mean(context, dim=1)
        sigma = torch.ones_like(mu, device=mu.device)
        # (batch_size, z_dim)

        self.gaussian_a = Normal(mu + offsets[0], sigma * scales[0])  # type: ignore
        self.gaussian_b = Normal(mu + offsets[1], sigma * scales[1])  # type: ignore

        self.weights = torch.tensor([weights[0], weights[1]])

    def sample(self, sample_shape: Size = torch.Size([])) -> Tensor:
        gaussian_indices = torch.multinomial(self.weights, self.batch_size, True)
        # (batch_size)

        samples_a = self.gaussian_a.sample()  # type: ignore
        samples_b = self.gaussian_b.sample()  # type: ignore
        # (batch_size, z_dim)

        samples = torch.stack(
            [
                samples_a[i] if gaussian_indices[i] == 0 else samples_b[i]
                for i in range(self.batch_size)
            ],
        )
        # (batch_size, z_dim)

        return samples

    def log_prob(self, x: Tensor) -> Tensor:
        return torch.logsumexp(
            torch.stack(
                [
                    torch.log(self.weights[0]) + self.gaussian_a.log_prob(x),  # type: ignore
                    torch.log(self.weights[1]) + self.gaussian_b.log_prob(x),  # type: ignore
                ]
            ),
            dim=0,
        )


class ContextualLatentSpaceGMM(Distribution):
    def __init__(self, context: Tensor):
        # (batch_size, context_size, z_dim)

        super(ContextualLatentSpaceGMM, self).__init__()

        self.batch_size = context.shape[0]
        self.context_size = context.shape[1]
        self.z_dim = context.shape[2]

        self.mu = torch.mean(context, dim=1)
        self.sigma = self.exp_decay(self.context_size, 1) * torch.ones_like(
            self.mu, device=self.mu.device
        )
        # (batch_size, z_dim)

        self.gaussians = self.get_gaussians_list()
        self.weights = torch.tensor(self.get_weights_list())

    def get_gaussians_list(self) -> List[Normal]:
        gaussians_list = []

        for permutation in list(itertools.product([1, -1], repeat=self.z_dim)):

            modified_mu = self.mu.clone()

            for dim in range(self.z_dim):
                modified_mu[:, dim] = modified_mu[:, dim] * permutation[dim]

            modified_gaussian = Normal(modified_mu, self.sigma)  # type: ignore

            gaussians_list.append(modified_gaussian)

        return gaussians_list

    def get_weights_list(self) -> List[float]:
        calibration = (len(self.gaussians) - 1) / len(self.gaussians)
        rest_weight = self.exp_decay(self.context_size, calibration)

        main_weight = 1 - rest_weight
        norm_rest_weight = rest_weight / (len(self.gaussians) - 1)

        weights_list = [main_weight] + [norm_rest_weight] * (len(self.gaussians) - 1)

        return weights_list

    def exp_decay(self, x: int, calibration: float, a: float = 0.2) -> float:
        val: float = np.exp(-a * x) + calibration - np.exp(-a)
        return val

    def sample(self, sample_shape: Size = torch.Size([])) -> Tensor:
        possible_samples = torch.stack(
            [gaussian.sample() for gaussian in self.gaussians]  # type: ignore
        )
        # (num_gaussians, batch_size, z_dim)

        gaussian_indices = torch.multinomial(self.weights, self.batch_size, True)
        # (batch_size)

        samples = torch.stack(
            [
                possible_samples[gaussian_indices[i], i, :]
                for i in range(self.batch_size)
            ],
        )
        # (batch_size, z_dim)

        return samples

    def log_prob(self, x: Tensor) -> Tensor:
        return torch.logsumexp(
            torch.stack(
                [
                    torch.log(self.weights[i]) + self.gaussians[i].log_prob(x)  # type: ignore
                    for i in range(len(self.gaussians))
                ],
            ),
            dim=0,
        )


class ContextDataset(Dataset[Tensor]):
    def __init__(
        self,
        size: int,
        c_dim: int,
        max_context_size: int,
        sampling_factor: float,
        variably_sized_context: bool,
    ) -> None:
        super(ContextDataset, self).__init__()

        self.size = size

        self.c_dim = c_dim
        self.max_context_size = max_context_size
        self.sampling_factor = sampling_factor

        self.variably_sized_context = variably_sized_context

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> Tensor:
        if self.variably_sized_context:
            context = self.sampling_factor * torch.rand(
                (self.max_context_size, self.c_dim)
            )

            choices = [random.choice([1, -1]) for _ in range(self.c_dim)]
            for i in range(self.c_dim):
                context[:, i] = context[:, i] * choices[i]

        else:
            context = torch.zeros((self.max_context_size, self.c_dim))

        return context
