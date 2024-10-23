from typing import List, Tuple
import numpy as np
import torch
from torch import Size, Tensor
from torch.distributions import Distribution, Normal
from torch.utils.data import Dataset
import itertools


class ContextualGaussian(Distribution):
    def __init__(self, context: Tensor, scale: float) -> None:
        # (batch_size, z_dim)

        super(ContextualGaussian, self).__init__()

        sigma = torch.ones_like(context, device=context.device) * scale

        self.gaussian = Normal(context, sigma)  # type: ignore

    def sample(self, sample_shape: Size = torch.Size([])) -> Tensor:
        return self.gaussian.sample(sample_shape)  # type: ignore

    def log_prob(self, x: Tensor) -> Tensor:
        return self.gaussian.log_prob(x)  # type: ignore


class ContextualGMM(Distribution):
    def __init__(
        self,
        context: Tensor,
        offsets: Tuple[float, float],
        scales: Tuple[float, float],
        weights: Tuple[float, float],
    ) -> None:
        # (batch_size, z_dim)

        super(ContextualGMM, self).__init__()

        self.batch_size = context.shape[0]

        sigma = torch.ones_like(context, device=context.device)

        self.gaussian_a = Normal(context + offsets[0], sigma * scales[0])  # type: ignore
        self.gaussian_b = Normal(context + offsets[1], sigma * scales[1])  # type: ignore

        self.weights = torch.tensor([weights[0], weights[1]])

    def sample(self, sample_shape: Size = torch.Size([])) -> Tensor:
        components = torch.multinomial(self.weights, self.batch_size, True)

        samples_a = self.gaussian_a.sample(sample_shape)  # type: ignore
        samples_b = self.gaussian_b.sample(sample_shape)  # type: ignore

        return torch.stack(
            [
                samples_a[i] if components[i] == 0 else samples_b[i]
                for i in range(self.batch_size)
            ],
        )

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
        # (context_size, z_dim)

        super(ContextualLatentSpaceGMM, self).__init__()

        self.context_size = context.shape[0]
        self.z_dim = context.shape[1]

        self.mu = torch.mean(context, dim=0)
        self.sigma = self.exp_decay(self.context_size, 1) * torch.ones_like(
            self.mu, device=self.mu.device
        )
        # (z_dim)

        self.gaussians = self.get_gaussians_list()
        self.weights = torch.tensor(self.get_weights_list())

    def get_gaussians_list(self) -> List[Normal]:
        gaussians_list = []

        for permutation in list(itertools.product([1, -1], repeat=self.z_dim)):

            modified_mu = self.mu.clone()

            for dim in range(self.z_dim):
                modified_mu[dim] = modified_mu[dim] * permutation[dim]

            modified_gaussian = Normal(modified_mu, self.sigma)  # type: ignore

            gaussians_list.append(modified_gaussian)

        return gaussians_list

    def get_weights_list(self) -> List[float]:
        rest_weight = self.exp_decay(self.context_size, 0.75)

        main_weight = 1 - rest_weight
        norm_rest_weight = rest_weight / (len(self.gaussians) - 1)

        weights_list = [main_weight] + [norm_rest_weight] * (len(self.gaussians) - 1)

        return weights_list

    def exp_decay(self, x: int, calibration: float, a: float = 0.2) -> float:
        val: float = np.exp(-a * x) + calibration - np.exp(-a)
        return val

    def sample(self, sample_shape: Size = torch.Size([])) -> Tensor:
        components = torch.multinomial(self.weights, sample_shape[0], True)

        samples = torch.stack(
            [gaussian.sample(sample_shape) for gaussian in self.gaussians], dim=0  # type: ignore
        )

        return torch.stack(
            [samples[components[i], i] for i in range(sample_shape[0])],
        )

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
    def __init__(self, size: int) -> None:
        super(ContextDataset, self).__init__()

        # self.contexts = np.linspace(-5, 5, size)

        self.contexts = np.zeros(size)

    def __len__(self) -> int:
        return len(self.contexts)

    def __getitem__(self, idx: int) -> Tensor:
        return torch.tensor([self.contexts[idx]]).float()
