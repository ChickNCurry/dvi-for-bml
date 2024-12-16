import os
import random
from typing import Tuple

import numpy as np
import torch
from hydra.utils import instantiate
from metalearning_benchmarks import MetaLearningBenchmark  # type: ignore
from omegaconf import DictConfig
from torch.optim.adamw import AdamW
from torch.utils.data import DataLoader, random_split

from src.components.nn.control_informed import InformedControl
from src.components.dvi.cdvi import CDVI
from src.components.dvinp import DVINP
from src.components.nn.control import Control
from src.components.nn.decoder import Decoder
from src.components.nn.encoder import Aggr, SetEncoder
from src.train.train import AbstractTrainer
from src.train.train_bml import BetterBMLTrainer
from src.train.train_bml_alternating import AlternatingBMLTrainer
from src.utils.datasets import MetaLearningDataset


def load_dvinp(
    cfg: DictConfig,
    dir: str,
    device: torch.device,
    decoder_only: bool = False,
    train_decoder: bool = True,
) -> Tuple[DVINP, AbstractTrainer]:

    torch.manual_seed(cfg.training.seed)
    random.seed(cfg.training.seed)
    np.random.seed(cfg.training.seed)

    g = torch.Generator()
    g.manual_seed(cfg.training.seed)

    benchmark: MetaLearningBenchmark = instantiate(cfg.benchmark)
    dataset = MetaLearningDataset(benchmark, cfg.training.max_context_size)

    if cfg.training.alternating_ratio is None:

        train_set, val_set = random_split(
            dataset,
            [len(dataset) - cfg.training.num_val_tasks, cfg.training.num_val_tasks],
        )

        train_loader = DataLoader(
            dataset=train_set,
            batch_size=cfg.training.batch_size,
            shuffle=True,
            generator=g,
        )

    else:

        len_train_set = len(dataset) - cfg.training.num_val_tasks
        len_train_decoder_set = int(len_train_set * cfg.training.alternating_ratio)
        len_train_cdvi_set = len_train_set - len_train_decoder_set

        train_decoder_set, train_cdvi_set, val_set = random_split(
            dataset,
            [len_train_decoder_set, len_train_cdvi_set, cfg.training.num_val_tasks],
        )

        train_decoder_loader = DataLoader(
            dataset=train_decoder_set,
            batch_size=cfg.training.batch_size,
            shuffle=True,
            generator=g,
        )

        train_cdvi_loader = DataLoader(
            dataset=train_cdvi_set,
            batch_size=cfg.training.batch_size,
            shuffle=True,
            generator=g,
        )

    val_loader = DataLoader(
        dataset=val_set,
        batch_size=cfg.training.num_val_tasks,
        shuffle=False,
        generator=g,
    )

    set_encoder = SetEncoder(
        c_dim=cfg.common.c_dim,
        h_dim=cfg.common.h_dim,
        num_layers=cfg.common.num_layers,
        non_linearity=cfg.common.non_linearity,
        is_attentive=cfg.set_encoder.is_attentive,
        num_heads=cfg.set_encoder.num_heads,
        is_non_aggregative=cfg.control.is_cross_attentive
        or cfg.decoder.is_cross_attentive,
        is_aggregative=not cfg.control.is_cross_attentive
        or not cfg.decoder.is_cross_attentive,
        aggregation=Aggr(cfg.set_encoder.aggregation),
        use_context_size_emb=cfg.set_encoder.use_context_size_emb,
        max_context_size=dataset.max_context_size,
    )

    control: Control | InformedControl = instantiate(
        cfg.control,
        h_dim=cfg.common.h_dim,
        z_dim=cfg.common.z_dim,
        num_steps=cfg.cdvi.num_steps,
        num_layers=3,  # cfg.common.num_layers,
        non_linearity=cfg.common.non_linearity,
    )

    cdvi: CDVI = instantiate(
        cfg.cdvi,
        z_dim=cfg.common.z_dim,
        control=control,
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

    dvinp = DVINP(
        encoder=set_encoder,
        cdvi=cdvi,
        decoder=decoder,
        contextual_target=None,
    ).to(device)

    optimizer = AdamW(
        (
            dvinp.parameters()
            if train_decoder
            else list(dvinp.encoder.parameters()) + list(dvinp.cdvi.parameters())
        ),
        lr=cfg.training.learning_rate,
    )

    trainer: AbstractTrainer = (
        BetterBMLTrainer(
            device=device,
            dvinp=dvinp,
            dataset=dataset,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            scheduler=None,
            wandb_logging=cfg.wandb.logging,
            num_subtasks=cfg.training.num_subtasks,
            sample_size=cfg.training.sample_size,
        )
        if cfg.training.alternating_ratio is None
        else AlternatingBMLTrainer(
            device=device,
            dvinp=dvinp,
            dataset=dataset,
            train_decoder_loader=train_decoder_loader,
            train_cdvi_loader=train_cdvi_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            wandb_logging=cfg.wandb.logging,
            num_subtasks=cfg.training.num_subtasks,
        )
    )

    dvinp_path = f"{dir}/dvinp.pth"
    optim_path = f"{dir}/optim.pth"

    if os.path.exists(dvinp_path):

        dvinp_state_dict = torch.load(
            dvinp_path, map_location=torch.device("cpu"), weights_only=True
        )

        if decoder_only:
            dvinp_state_dict = {
                k: v for k, v in dvinp_state_dict.items() if "decoder" in k
            }

        dvinp.load_state_dict(dvinp_state_dict)
        print(f"loaded dvinp from {dvinp_path}")

    if os.path.exists(optim_path):
        optim_state_dict = torch.load(
            optim_path, map_location=torch.device("cpu"), weights_only=True
        )

        if decoder_only:
            optim_state_dict = {
                k: v for k, v in optim_state_dict.items() if "decoder" in k
            }

        trainer.optimizer.load_state_dict(optim_state_dict)
        print(f"loaded optim from {optim_path}")

    return dvinp, trainer
