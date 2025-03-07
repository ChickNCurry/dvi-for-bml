from typing import List, Tuple

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import Tensor
from torch.utils.data import DataLoader


def visualize_task(dataloader: DataLoader[Tuple[Tensor, Tensor]]) -> None:
    plt.figure(figsize=(8, 4))

    for batch in dataloader:
        x_data, y_data = batch
        # (batch_size, data_size, x_dim)
        # (batch_size, data_size, y_dim)

        x_data_sorted, indices = x_data.sort(dim=1)
        y_data_sorted = y_data.gather(1, indices)

        x_data, y_data = x_data.numpy(), y_data.numpy()

        for i in range(x_data.shape[0]):
            plt.plot(x_data_sorted[i, :, 0], y_data_sorted[i, :, 0])

        # plt.scatter(x_data[:, :, 0], y_data[:, :, 0])

    plt.show()
