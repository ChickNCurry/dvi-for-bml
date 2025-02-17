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
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader, random_split

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
from src.components.schedule.cos_noise_schedule import (
    AggrCosineNoiseSchedule,
    BCACosineNoiseSchedule,
    CosineNoiseSchedule,
)
from src.components.schedule.noise_schedule import (
    AggrNoiseSchedule,
    BCANoiseSchedule,
    MHANoiseSchedule,
    NoiseSchedule,
)
from src.components.schedule.step_size_schedule import StepSizeSchedule
from src.train.base_trainer import AbstractTrainer
from src.train.dvinp_trainer import DVINPTrainer
from src.utils.datasets import MetaLearningDataset


class ModelVariant(Enum):
    MEAN = "mean"
    MAX = "max"
    BCA = "bca"
    MHA = "mha"


class NoiseVariant(Enum):
    FREE = "free"
    COS = "cos"


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

    model_variant = ModelVariant(cfg.model.model_variant)
    noise_variant = NoiseVariant(cfg.model.noise_variant)

    match model_variant:
        case ModelVariant.MEAN | ModelVariant.MAX:

            encoder = AggrEncoder(
                c_dim=cfg.model.c_dim,
                h_dim=cfg.model.h_dim,
                num_layers=cfg.model.num_layers,
                non_linearity=cfg.model.non_linearity,
                num_heads=cfg.model.self_attn_num_heads,
                aggregation=Aggr(ModelVariant(model_variant).value),
                max_context_size=cfg.model.max_context_size,
            )

            control = AggrControl(
                h_dim=cfg.model.h_dim,
                z_dim=cfg.model.z_dim,
                num_steps=cfg.model.num_steps,
                num_layers=cfg.model.num_layers,
                non_linearity=cfg.model.non_linearity,
                use_score=cfg.model.use_score,
            )

            annealing_schedule = AggrAnnealingSchedule(
                h_dim=cfg.model.h_dim,
                non_linearity=cfg.model.non_linearity,
                num_steps=cfg.model.num_steps,
                device=device,
            )

            match noise_variant:
                case NoiseVariant.FREE:

                    noise_schedule = AggrNoiseSchedule(
                        z_dim=cfg.model.z_dim,
                        h_dim=cfg.model.h_dim,
                        non_linearity=cfg.model.non_linearity,
                        num_steps=cfg.model.num_steps,
                        device=device,
                    )

                case NoiseVariant.COS:

                    noise_schedule = AggrCosineNoiseSchedule(
                        z_dim=cfg.model.z_dim,
                        h_dim=cfg.model.h_dim,
                        non_linearity=cfg.model.non_linearity,
                        num_steps=cfg.model.num_steps,
                    )

        case ModelVariant.BCA:

            encoder = BCAEncoder(
                c_dim=cfg.model.c_dim,
                h_dim=cfg.model.h_dim,
                num_layers=cfg.model.num_layers,
                non_linearity=cfg.model.non_linearity,
                num_heads=cfg.model.self_attn_num_heads,
            )

            control = BCAControl(
                h_dim=cfg.model.h_dim,
                z_dim=cfg.model.z_dim,
                num_steps=cfg.model.num_steps,
                num_layers=cfg.model.num_layers,
                non_linearity=cfg.model.non_linearity,
                use_score=cfg.model.use_score,
            )

            annealing_schedule = BCAAnnealingSchedule(
                h_dim=cfg.model.h_dim,
                non_linearity=cfg.model.non_linearity,
                num_steps=cfg.model.num_steps,
                device=device,
            )

            match noise_variant:
                case NoiseVariant.FREE:

                    noise_schedule = BCANoiseSchedule(
                        z_dim=cfg.model.z_dim,
                        h_dim=cfg.model.h_dim,
                        non_linearity=cfg.model.non_linearity,
                        num_steps=cfg.model.num_steps,
                        device=device,
                    )

                case NoiseVariant.COS:

                    noise_schedule = BCACosineNoiseSchedule(
                        z_dim=cfg.model.z_dim,
                        h_dim=cfg.model.h_dim,
                        non_linearity=cfg.model.non_linearity,
                        num_steps=cfg.model.num_steps,
                    )

        case ModelVariant.MHA:

            encoder = MHAEncoder(
                c_dim=cfg.model.c_dim,
                h_dim=cfg.model.h_dim,
                num_layers=cfg.model.num_layers,
                non_linearity=cfg.model.non_linearity,
                num_heads=cfg.model.self_attn_num_heads,
            )

            control = MHAControl(
                h_dim=cfg.model.h_dim,
                z_dim=cfg.model.z_dim,
                num_steps=cfg.model.num_steps,
                num_layers=cfg.model.num_layers,
                non_linearity=cfg.model.non_linearity,
                use_score=cfg.model.use_score,
                num_heads=cfg.model.cross_attn_num_heads,
            )

            annealing_schedule = MHAAnnealingSchedule(
                h_dim=cfg.model.h_dim,
                non_linearity=cfg.model.non_linearity,
                num_steps=cfg.model.num_steps,
                device=device,
                num_heads=cfg.model.cross_attn_num_heads,
            )

            match noise_variant:
                case NoiseVariant.FREE:

                    noise_schedule = MHANoiseSchedule(
                        z_dim=cfg.model.z_dim,
                        h_dim=cfg.model.h_dim,
                        non_linearity=cfg.model.non_linearity,
                        num_steps=cfg.model.num_steps,
                        device=device,
                    )

                case NoiseVariant.COS:

                    noise_schedule = MHANoiseSchedule(
                        z_dim=cfg.model.z_dim,
                        h_dim=cfg.model.h_dim,
                        non_linearity=cfg.model.non_linearity,
                        num_steps=cfg.model.num_steps,
                        device=device,
                        num_heads=cfg.model.cross_attn_num_heads,
                    )  # TODO

    if not cfg.model.contextual_schedules:

        annealing_schedule = AnnealingSchedule(
            num_steps=cfg.model.num_steps,
            device=device,
        )

        match noise_variant:
            case NoiseVariant.FREE:

                noise_schedule = NoiseSchedule(
                    z_dim=cfg.model.z_dim,
                    num_steps=cfg.model.num_steps,
                    device=device,
                )

            case NoiseVariant.COS:

                noise_schedule = CosineNoiseSchedule(
                    z_dim=cfg.model.z_dim,
                    num_steps=cfg.model.num_steps,
                    device=device,
                )

    step_size_schedule = StepSizeSchedule(
        num_steps=cfg.model.num_steps,
        device=device,
    )

    cdvi = DIS(
        z_dim=cfg.model.z_dim,
        num_steps=cfg.model.num_steps,
        control=control,
        step_size_schedule=step_size_schedule,
        noise_schedule=noise_schedule,
        annealing_schedule=annealing_schedule,
        use_score=cfg.model.use_score,
        device=device,
    )

    decoder = Decoder(
        x_dim=cfg.model.x_dim,
        z_dim=cfg.model.z_dim,
        h_dim=cfg.model.h_dim,
        y_dim=cfg.model.y_dim,
        num_layers=cfg.model.num_layers,
        non_linearity=cfg.model.non_linearity,
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

    # scheduler = CosineAnnealingWarmRestarts(
    #     optimizer=optimizer,
    #     T_0=5,
    #     T_mult=2,
    #     eta_min=cfg.training.learning_rate * 0.01,
    # )

    trainer = DVINPTrainer(
        device=device,
        dvinp=dvinp,
        dataset=dataset,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=None,  # scheduler,
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
