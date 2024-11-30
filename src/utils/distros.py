import itertools
from typing import List, Tuple

import torch
from torch import Size, Tensor
from torch.distributions import Distribution, Normal


class ContextualGaussian(Distribution):
    def __init__(self, context: Tensor, scale: float) -> None:
        # (batch_size, context_size, z_dim)

        super(ContextualGaussian, self).__init__(validate_args=False)

        assert context.shape[1] == 1

        mu = torch.mean(context, dim=1)
        sigma = torch.ones_like(mu, device=mu.device) * scale
        # (batch_size, z_dim)

        self.gaussian = Normal(mu, sigma)  # type: ignore

    def sample(self, sample_shape: Size | None = None) -> Tensor:
        return self.gaussian.sample()  # type: ignore

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
        # (batch_size, context_size, z_dim)

        super(ContextualGMM, self).__init__(validate_args=False)

        assert context.shape[1] == 1

        self.batch_size = context.shape[0]

        mu = torch.mean(context, dim=1)
        sigma = torch.ones_like(mu, device=mu.device)
        # (batch_size, z_dim)

        self.component_a = Normal(mu + offsets[0], sigma * scales[0])  # type: ignore
        self.component_b = Normal(mu + offsets[1], sigma * scales[1])  # type: ignore
        self.weights = torch.tensor([weights[0], weights[1]])

    def sample(self, sample_shape: Size | None = None) -> Tensor:
        component_indices = torch.multinomial(self.weights, self.batch_size, True)
        # (batch_size)

        samples_a = self.component_a.sample()  # type: ignore
        samples_b = self.component_b.sample()  # type: ignore
        # (batch_size, z_dim)

        samples = torch.stack(
            [
                samples_a[i] if component_indices[i] == 0 else samples_b[i]
                for i in range(self.batch_size)
            ],
        )
        # (batch_size, z_dim)

        return samples

    def log_prob(self, x: Tensor) -> Tensor:
        return torch.logsumexp(
            torch.stack(
                [
                    torch.log(self.weights[0]) + self.component_a.log_prob(x),  # type: ignore
                    torch.log(self.weights[1]) + self.component_b.log_prob(x),  # type: ignore
                ]
            ),
            dim=0,
        )


class TaskPosteriorGMM(Distribution):
    def __init__(self, context: Tensor, mask: Tensor | None) -> None:
        # (batch_size, context_size, z_dim)
        # (batch_size, context_size)

        super(TaskPosteriorGMM, self).__init__(validate_args=False)

        self.batch_size = context.shape[0]
        self.context_size = context.shape[1]
        self.z_dim = context.shape[2]

        if mask is not None:
            counts = mask.sum(dim=1, keepdim=True)
            # (batch_size, 1)

            sum = context.sum(dim=1)
            # (batch_size, z_dim)

            self.mu = sum / counts
            # (batch_size, z_dim)

            self.sigma = torch.ones_like(
                self.mu, device=self.mu.device
            ) * self.exp_decay(counts, 1)
            # (batch_size, z_dim)

            self.components = self.get_components()
            self.weights = self.get_weights_via_counts(counts)

        else:
            counts = (
                torch.tensor([self.context_size], device=context.device)
                .unsqueeze(0)
                .expand(self.batch_size, -1)
            )
            # (batch_size, 1)

            self.mu = torch.mean(context, dim=1)
            # (batch_size, z_dim)

            self.sigma = torch.ones_like(
                self.mu, device=self.mu.device
            ) * self.exp_decay(counts, 1)
            # (batch_size, z_dim)

            self.components = self.get_components()
            self.weights = self.get_weights()

    def get_components(self) -> List[Normal]:
        components = []

        for permutation in list(itertools.product([1, -1], repeat=self.z_dim)):

            modified_mu = self.mu.clone()

            for dim in range(self.z_dim):
                modified_mu[:, dim] = modified_mu[:, dim] * permutation[dim]

            modified_component = Normal(modified_mu, self.sigma)  # type: ignore

            components.append(modified_component)

        return components

    def get_weights_via_counts(self, counts: Tensor) -> Tensor:
        # (batch_size, 1)

        calibration = (len(self.components) - 1) / len(self.components)

        rest_weights = self.exp_decay(counts, calibration)
        norm_rest_weight = rest_weights / (len(self.components) - 1)
        main_weight = 1 - rest_weights
        # (batch_size, 1)

        weights = torch.stack(
            tensors=[main_weight] + [norm_rest_weight] * (len(self.components) - 1),
        )
        # (num_components, batch_size)

        return weights

    def get_weights(self) -> Tensor:
        calibration = (len(self.components) - 1) / len(self.components)
        rest_weight = self.exp_decay(
            torch.tensor([self.context_size], device=self.mu.device), calibration
        )

        main_weight = 1 - rest_weight
        norm_rest_weight = rest_weight / (len(self.components) - 1)

        weights_list = [main_weight] + [norm_rest_weight] * (len(self.components) - 1)
        weights = torch.tensor(weights_list, device=self.mu.device)

        return weights

    def exp_decay(self, x: Tensor, calibration: float, a: float = 0.2) -> Tensor:
        return torch.exp(-a * x) + calibration - torch.exp(-a * torch.ones_like(x))

    def sample(self, sample_shape: Size | None = None) -> Tensor:
        possible_samples = torch.stack([c.sample() for c in self.components])  # type: ignore
        # (num_components, batch_size, z_dim)

        component_indices = torch.multinomial(self.weights, self.batch_size, True)
        # (batch_size)

        samples = torch.stack(
            [
                possible_samples[component_indices[i], i, :]
                for i in range(self.batch_size)
            ],
        )
        # (batch_size, z_dim)

        return samples

    def log_prob(self, x: Tensor) -> Tensor:
        return torch.logsumexp(
            torch.stack(
                [
                    torch.log(self.weights[i]) + self.components[i].log_prob(x)  # type: ignore
                    for i in range(len(self.components))
                ],
            ),
            dim=0,
        )
