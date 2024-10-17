import math

import numpy as np
import torch
from torch import Size, Tensor
from torch.distributions import Normal, Distribution
from torch.utils.data import Dataset


def contextual_gaussian_sine(context: Tensor) -> Normal:
    mu = 5 * torch.sin(context)
    std = torch.ones_like(mu, device=context.device)
    return Normal(mu, std)  # type: ignore


class ContextualGMM(Distribution):
    def __init__(self, context: Tensor) -> None:
        super(ContextualGMM, self).__init__()

        self.batch_size = context.shape[0]

        std = torch.ones_like(context, device=context.device)

        self.gaussian_a = Normal(5 + context, std)  # type: ignore
        self.gaussian_b = Normal(-5 + context, std)  # type: ignore

        self.weights = torch.tensor([0.5, 0.5])

    def sample(self, sample_shape: Size = torch.Size([])) -> Tensor:
        components = torch.multinomial(self.weights, self.batch_size, True)

        print(components.shape)

        samples_a = self.gaussian_a.sample()  # type: ignore
        samples_b = self.gaussian_b.sample()  # type: ignore

        print(samples_a.shape, samples_b.shape)
        print(samples_a[1].shape)

        samples = torch.stack(
            [
                samples_a[i] if components[i] == 0 else samples_b[i]
                for i in range(self.batch_size)
            ],
        )

        print(samples.shape)

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


class ContextDataset(Dataset[Tensor]):
    def __init__(self, size: int) -> None:
        super(ContextDataset, self).__init__()

        # self.contexts = np.arange(0, 2 * math.pi, 2 * math.pi / size)
        # self.contexts = np.ones(size) * math.pi / 2

        self.contexts = np.arange(-5, 5, 10 / size)

    def __len__(self) -> int:
        return len(self.contexts)

    def __getitem__(self, idx: int) -> Tensor:
        return torch.tensor([self.contexts[idx]]).float()
