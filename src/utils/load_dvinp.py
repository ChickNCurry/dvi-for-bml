import os
import random
from enum import Enum
from typing import Tuple

import numpy as np
import torch
from hydra.utils import instantiate
from metalearning_benchmarks import MetaLearningBenchmark  # type: ignore
from omegaconf import DictConfig
from torch.optim.adamw import AdamW
from torch.utils.data import DataLoader, random_split

from src.components.cdvi.cdvi import CDVI
from src.components.cdvi.dis import DIS
from src.components.control.aggr_control import AggrControl
from src.components.control.bca_control import BCAControl
from src.components.control.mha_control import MHAControl
from src.components.decoder.decoder import Decoder
from src.components.dvinp import DVINP
from src.components.encoder.aggr_encoder import Aggr, AggrEncoder
from src.components.encoder.bca_encoder import BCAEncoder
from src.components.encoder.mha_encoder import MHAEncoder
from src.components.schedule.annealing_schedule import (
    AggrAnnealingSchedule,
    AnnealingSchedule,
    BCAAnnealingSchedule,
    MHAAnnealingSchedule,
)
from src.components.schedule.noise_schedule import (
    AggrNoiseSchedule,
    BCANoiseSchedule,
    MHANoiseSchedule,
    NoiseSchedule,
)
from src.components.schedule.step_size_schedule import StepSizeSchedule
from src.train.base_trainer import AbstractTrainer
from src.train.dvinp_trainer import BetterDVINPTrainer
from src.utils.datasets import MetaLearningDataset


class ContextualizationVariant(Enum):
    AGGR = "aggr"
    BCA = "bca"
    MHA = "mha"


def load_dvinp(
    cfg: DictConfig,
    device: torch.device,
    dir: str | None = None,
    load_decoder_only: bool = False,
    train_decoder: bool = True,
) -> Tuple[DVINP, AbstractTrainer, DataLoader]:

    torch.manual_seed(cfg.training.seed)
    random.seed(cfg.training.seed)
    np.random.seed(cfg.training.seed)

    g = torch.Generator()
    g.manual_seed(cfg.training.seed)

    benchmark: MetaLearningBenchmark = instantiate(cfg.benchmark)
    dataset = MetaLearningDataset(benchmark, cfg.training.max_context_size)

    train_set, val_set = random_split(
        dataset, [len(dataset) - cfg.training.num_val_tasks, cfg.training.num_val_tasks]
    )

    train_loader = DataLoader(
        dataset=train_set,
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

    test_loader = DataLoader(
        dataset=val_set,
        batch_size=1,
        shuffle=False,
        generator=g,
    )

    variant = ContextualizationVariant(cfg.common.variant)

    match variant:
        case ContextualizationVariant.AGGR:

            encoder = AggrEncoder(
                c_dim=cfg.common.c_dim,
                h_dim=cfg.common.h_dim,
                num_layers=cfg.common.num_layers,
                non_linearity=cfg.common.non_linearity,
                num_heads=cfg.common.self_attn_num_heads,
                aggregation=Aggr(cfg.common.aggregation),
                max_context_size=cfg.common.max_context_size,
            )

            control = AggrControl(
                h_dim=cfg.common.h_dim,
                z_dim=cfg.common.z_dim,
                num_steps=cfg.common.num_steps,
                num_layers=cfg.common.num_layers,
                non_linearity=cfg.common.non_linearity,
                use_score=cfg.common.use_score,
            )

            noise_schedule = AggrNoiseSchedule(
                z_dim=cfg.common.z_dim,
                h_dim=cfg.common.h_dim,
                non_linearity=cfg.common.non_linearity,
                num_steps=cfg.common.num_steps,
                device=device,
            )

            annealing_schedule = AggrAnnealingSchedule(
                h_dim=cfg.common.h_dim,
                non_linearity=cfg.common.non_linearity,
                num_steps=cfg.common.num_steps,
                device=device,
            )

        case ContextualizationVariant.BCA:

            encoder = BCAEncoder(
                c_dim=cfg.common.c_dim,
                h_dim=cfg.common.h_dim,
                num_layers=cfg.common.num_layers,
                non_linearity=cfg.common.non_linearity,
                num_heads=cfg.common.self_attn_num_heads,
            )

            control = BCAControl(
                h_dim=cfg.common.h_dim,
                z_dim=cfg.common.z_dim,
                num_steps=cfg.common.num_steps,
                num_layers=cfg.common.num_layers,
                non_linearity=cfg.common.non_linearity,
                use_score=cfg.common.use_score,
            )

            noise_schedule = BCANoiseSchedule(
                z_dim=cfg.common.z_dim,
                h_dim=cfg.common.h_dim,
                non_linearity=cfg.common.non_linearity,
                num_steps=cfg.common.num_steps,
                device=device,
            )

            annealing_schedule = BCAAnnealingSchedule(
                h_dim=cfg.common.h_dim,
                non_linearity=cfg.common.non_linearity,
                num_steps=cfg.common.num_steps,
                device=device,
            )

        case ContextualizationVariant.MHA:

            encoder = MHAEncoder(
                c_dim=cfg.common.c_dim,
                h_dim=cfg.common.h_dim,
                num_layers=cfg.common.num_layers,
                non_linearity=cfg.common.non_linearity,
                num_heads=cfg.common.self_attn_num_heads,
            )

            control = MHAControl(
                h_dim=cfg.common.h_dim,
                z_dim=cfg.common.z_dim,
                num_steps=cfg.common.num_steps,
                num_layers=cfg.common.num_layers,
                non_linearity=cfg.common.non_linearity,
                use_score=cfg.common.use_score,
                num_heads=cfg.common.cross_attn_num_heads,
            )

            noise_schedule = MHANoiseSchedule(
                z_dim=cfg.common.z_dim,
                h_dim=cfg.common.h_dim,
                non_linearity=cfg.common.non_linearity,
                num_steps=cfg.common.num_steps,
                device=device,
                num_heads=cfg.common.cross_attn_num_heads,
            )

            annealing_schedule = MHAAnnealingSchedule(
                h_dim=cfg.common.h_dim,
                non_linearity=cfg.common.non_linearity,
                num_steps=cfg.common.num_steps,
                device=device,
                num_heads=cfg.common.cross_attn_num_heads,
            )

    step_size_schedule = StepSizeSchedule(
        num_steps=cfg.common.num_steps,
        device=device,
    )

    if not cfg.common.contextual_schedules:

        noise_schedule = NoiseSchedule(
            z_dim=cfg.common.z_dim,
            num_steps=cfg.common.num_steps,
            device=device,
        )

        annealing_schedule = AnnealingSchedule(
            num_steps=cfg.common.num_steps,
            device=device,
        )

    cdvi = DIS(
        z_dim=cfg.common.z_dim,
        num_steps=cfg.common.num_steps,
        control=control,
        step_size_schedule=step_size_schedule,
        noise_schedule=noise_schedule,
        annealing_schedule=annealing_schedule,
        use_score=cfg.common.use_score,
        device=device,
    )

    decoder = Decoder(
        x_dim=cfg.common.x_dim,
        z_dim=cfg.common.z_dim,
        h_dim=cfg.common.h_dim,
        y_dim=cfg.common.y_dim,
        num_layers=cfg.common.num_layers,
        non_linearity=cfg.common.non_linearity,
    )

    dvinp = DVINP(
        encoder=encoder,
        cdvi=cdvi,
        decoder=decoder,
        contextual_target=None,
    ).to(device)

    params = (
        dvinp.parameters()
        if train_decoder
        else list(dvinp.encoder.parameters()) + list(dvinp.cdvi.parameters())
    )

    optimizer = AdamW(params=params, lr=cfg.training.learning_rate)

    trainer = BetterDVINPTrainer(
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

    if dir is not None:

        dvinp_path = f"{dir}/dvinp.pth"
        optim_path = f"{dir}/optim.pth"

        if os.path.exists(dvinp_path):
            dvinp_state_dict = torch.load(
                dvinp_path, map_location=torch.device("cpu"), weights_only=False
            )

            if load_decoder_only:
                dvinp.decoder.load_state_dict(
                    {
                        k.split("decoder.")[-1]: v
                        for k, v in dvinp_state_dict.items()
                        if "decoder" in k
                    }
                )
                print(f"loaded decoder from {dvinp_path}")
            else:
                dvinp.load_state_dict(dvinp_state_dict, strict=False)
                print(f"loaded dvinp from {dvinp_path}")

        if os.path.exists(optim_path):
            optim_state_dict = torch.load(
                optim_path, map_location=torch.device("cpu"), weights_only=False
            )

            if not load_decoder_only:
                trainer.optimizer.load_state_dict(optim_state_dict)
                print(f"loaded optim from {optim_path}")

    return dvinp, trainer, test_loader
