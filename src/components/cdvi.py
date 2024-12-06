from typing import Callable, Tuple

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch import Tensor, nn
from torch.distributions import Distribution
from torch.optim import Optimizer
from torch.optim.adamw import AdamW
from torch.utils.data import DataLoader, random_split

from src.components.control import Control
from src.components.decoder import Decoder
from components.cdvi_process import CDVIProcess
from src.components.encoder import Encoder, SetEncoder
from src.components.hyper_net import HyperNet
from src.utils.datasets import MetaLearningDataset


class CDVI(nn.Module):
    def __init__(
        self,
        encoder: Encoder,
        dvi_process: CDVIProcess,
        decoder: Decoder | None,
        contextual_target: Callable[[Tensor, Tensor | None], Distribution] | None,
    ) -> None:
        super().__init__()

        self.encoder = encoder
        self.dvi_process = dvi_process
        self.decoder = decoder
        self.contextual_target = contextual_target

    def freeze(self, only_decoder: bool) -> None:
        if only_decoder:
            for param in self.decoder.parameters():
                param.requires_grad = False
            for param in self.encoder.parameters():
                param.requires_grad = True
            for param in self.dvi_process.parameters():
                param.requires_grad = True
        else:
            for param in self.decoder.parameters():
                param.requires_grad = True
            for param in self.encoder.parameters():
                param.requires_grad = False
            for param in self.dvi_process.parameters():
                param.requires_grad = False


def load_cdvi_for_bml(
    cfg: DictConfig, device: torch.device
) -> Tuple[CDVI, Optimizer, DataLoader, DataLoader]:
    benchmark = instantiate(cfg.benchmark)
    dataset = MetaLearningDataset(benchmark=benchmark)

    num_val_tasks = 32
    train_set, val_set = random_split(
        dataset, [len(dataset) - num_val_tasks, num_val_tasks]
    )

    train_loader = DataLoader(train_set, cfg.training.batch_size, True)
    val_loader = DataLoader(val_set, num_val_tasks, False)

    set_encoder = SetEncoder(
        c_dim=cfg.common.c_dim,
        h_dim=cfg.common.h_dim,
        num_layers=cfg.common.num_layers,
        non_linearity=cfg.common.non_linearity,
        is_attentive=cfg.set_encoder.is_attentive,
        is_aggregative=not cfg.control_and_hyper_net.is_cross_attentive
        or not cfg.decoder.is_cross_attentive,
        is_non_aggregative=cfg.control_and_hyper_net.is_cross_attentive
        or cfg.decoder.is_cross_attentive,
        use_context_size=cfg.set_encoder.use_context_size,
        aggregation=cfg.set_encoder.aggregation,
        max_context_size=dataset.max_context_size,
    )

    control = Control(
        h_dim=cfg.common.h_dim,
        z_dim=cfg.common.z_dim,
        num_layers=cfg.common.num_layers,
        non_linearity=cfg.common.non_linearity,
        num_steps=cfg.dvi_process.num_steps,
        is_cross_attentive=cfg.control_and_hyper_net.is_cross_attentive,
        num_heads=cfg.control_and_hyper_net.num_heads,
    )

    hyper_net = (
        HyperNet(
            h_dim=cfg.common.h_dim,
            z_dim=cfg.common.z_dim,
            non_linearity=cfg.common.non_linearity,
            num_steps=cfg.dvi_process.num_steps,
            is_cross_attentive=cfg.control_and_hyper_net.is_cross_attentive,
            num_heads=cfg.control_and_hyper_net.num_heads,
        )
        if cfg.control_and_hyper_net.use_hyper_net
        else None
    )

    dvi_process: CDVIProcess = instantiate(
        cfg.dvi_process,
        z_dim=cfg.common.z_dim,
        control=control,
        hyper_net=hyper_net,
        device=device,
    )

    decoder = Decoder(
        x_dim=cfg.common.x_dim,
        z_dim=cfg.common.z_dim,
        h_dim=cfg.common.h_dim,
        y_dim=cfg.common.y_dim,
        num_layers=cfg.common.num_layers,
        non_linearity=cfg.common.non_linearity,
        has_lat_path=cfg.decoder.has_lat_path,
        has_det_path=cfg.decoder.has_det_path,
        is_cross_attentive=cfg.decoder.is_cross_attentive,
        num_heads=cfg.decoder.num_heads,
    )

    cdvi = CDVI(
        encoder=set_encoder,
        dvi_process=dvi_process,
        decoder=decoder,
        contextual_target=None,
    ).to(device)

    optimizer = AdamW(cdvi.parameters(), lr=cfg.training.learning_rate)

    return cdvi, optimizer, train_loader, val_loader
