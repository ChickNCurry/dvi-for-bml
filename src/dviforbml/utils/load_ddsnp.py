import random
from enum import Enum
from typing import Tuple

import numpy as np
import torch
from hydra.utils import instantiate
from metalearning_benchmarks import MetaLearningBenchmark
from omegaconf import DictConfig
from torch.optim.adamw import AdamW


from torch.utils.data import DataLoader, random_split

from dviforbml.architectures.dvinp import DVINP
from dviforbml.components.cdvi.dds import DDS
from dviforbml.components.control.aggr_control import AggrControl
from dviforbml.components.control.bca_control import BCAControl
from dviforbml.components.control.mhca_control import MHCAControl
from dviforbml.components.decoder.decoder import Decoder
from dviforbml.components.encoder.aggr_encoder import Aggr, AggrEncoder
from dviforbml.components.encoder.bca_encoder import BCAEncoder
from dviforbml.components.encoder.mhca_encoder import MHCAEncoder

from dviforbml.training.ddsnp_trainer import DDSNPTrainerContext

from dviforbml.utils.datasets import MetaLearningDataset, Sinusoid1DFreq
from dviforbml.utils.helper import load_state_dicts_np


class ContextVariant(Enum):
    MEAN = "mean"
    MAX = "max"
    BCA = "bca"
    MHCA = "mhca"


def load_ddsnp(
    cfg: DictConfig,
    device: torch.device,
    dir: str | None = None,
    load_decoder_only: bool = False,
    train_decoder: bool = True,
    debugging: bool = False,
) -> Tuple[DVINP, DDSNPTrainerContext, DataLoader, DataLoader]:
    torch.manual_seed(cfg.training.seed)
    random.seed(cfg.training.seed)
    np.random.seed(cfg.training.seed)

    g = torch.Generator().manual_seed(cfg.training.seed)

    if (
        cfg.benchmark._target_
        == "metalearning_benchmarks.sinusoid1d_benchmark.Sinusoid1DFreq"
    ):
        benchmark = Sinusoid1DFreq(
            n_task=cfg.benchmark.n_task,
            n_datapoints_per_task=cfg.benchmark.n_datapoints_per_task,
            output_noise=cfg.benchmark.output_noise,
            seed_task=cfg.benchmark.seed_task,
            seed_x=cfg.benchmark.seed_x,
            seed_noise=cfg.benchmark.seed_noise,
        )
    else:
        benchmark: MetaLearningBenchmark = instantiate(cfg.benchmark)

    dataset = MetaLearningDataset(benchmark, cfg.training.max_context_size, g)

    train_set, val_set = (
        random_split(
            dataset,
            [len(dataset) - cfg.training.num_val_tasks, cfg.training.num_val_tasks],
        )
        if not debugging
        else (dataset, dataset)
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
        shuffle=True,
        generator=g,
    )

    test_loader = DataLoader(
        dataset=val_set,
        batch_size=1,
        shuffle=True,
        generator=g,
    )

    context_variant = ContextVariant(cfg.model.context_variant)

    match context_variant:
        case ContextVariant.MEAN | ContextVariant.MAX:
            encoder = AggrEncoder(
                c_dim=cfg.model.c_dim,
                h_dim=cfg.model.h_dim,
                z_dim=cfg.model.z_dim,
                num_layers=cfg.model.num_layers_enc,
                non_linearity=cfg.model.non_linearity,
                num_heads=cfg.model.self_attn_num_heads,
                num_blocks=cfg.model.num_blocks,
                max_context_size=cfg.model.max_context_size,
                aggregation=Aggr(ContextVariant(context_variant).value),
            )

            control = AggrControl(
                h_dim=cfg.model.h_dim,
                z_dim=cfg.model.z_dim,
                num_steps=cfg.model.num_steps,
                num_layers=cfg.model.num_layers_ctrl,
                non_linearity=cfg.model.non_linearity,
                max_context_size=cfg.model.max_context_size,
                use_score=False,
            )

        case ContextVariant.BCA:
            encoder = BCAEncoder(
                c_dim=cfg.model.c_dim,
                h_dim=cfg.model.h_dim,
                z_dim=cfg.model.z_dim,
                num_layers=cfg.model.num_layers_enc,
                non_linearity=cfg.model.non_linearity,
                num_heads=cfg.model.self_attn_num_heads,
                num_blocks=cfg.model.num_blocks,
                max_context_size=cfg.model.max_context_size,
                bca_dim=cfg.model.h_dim,
            )

            control = BCAControl(
                h_dim=cfg.model.h_dim,
                z_dim=cfg.model.z_dim,
                num_steps=cfg.model.num_steps,
                num_layers=cfg.model.num_layers_ctrl,
                non_linearity=cfg.model.non_linearity,
                max_context_size=cfg.model.max_context_size,
                use_score=False,
            )

        case ContextVariant.MHCA:
            encoder = MHCAEncoder(
                c_dim=cfg.model.c_dim,
                h_dim=cfg.model.h_dim,
                z_dim=cfg.model.z_dim,
                num_layers=cfg.model.num_layers_enc,
                non_linearity=cfg.model.non_linearity,
                num_heads=cfg.model.self_attn_num_heads,
                num_blocks=cfg.model.num_blocks,
                max_context_size=cfg.model.max_context_size,
            )

            control = MHCAControl(
                h_dim=cfg.model.h_dim,
                z_dim=cfg.model.z_dim,
                num_steps=cfg.model.num_steps,
                num_layers=cfg.model.num_layers_ctrl,
                non_linearity=cfg.model.non_linearity,
                max_context_size=cfg.model.max_context_size,
                use_score=False,
                num_heads=cfg.model.cross_attn_num_heads,
            )

    cdvi = DDS(
        z_dim=cfg.model.z_dim,
        num_steps=cfg.model.num_steps,
        control=control,
        meta_batch_size=cfg.training.batch_size,
        num_subtasks=cfg.training.num_subtasks,
        device=device,
    )

    decoder = Decoder(
        x_dim=cfg.model.x_dim,
        z_dim=cfg.model.z_dim,
        h_dim=cfg.model.h_dim_dec,  # 32
        y_dim=cfg.model.y_dim,
        num_layers=cfg.model.num_layers_dec,  # 3
        non_linearity=cfg.model.non_linearity,
    )

    model = DVINP(
        encoder=encoder,
        cdvi=cdvi,
        decoder=decoder,
    ).to(device)

    params = (
        model.parameters()
        if train_decoder
        else list(model.encoder.parameters()) + list(model.cdvi.parameters())
    )

    optimizer = AdamW(params=params, lr=cfg.training.learning_rate)

    trainer_params = {
        "model": model,
        "device": device,
        "dataset": dataset,
        "train_loader": train_loader,
        "val_loader": val_loader,
        "optimizer": optimizer,
        "scheduler": None,  # scheduler,
        "generator": g,
        "wandb_logging": cfg.wandb.logging,
        "num_subtasks": cfg.training.num_subtasks,
        "num_samples": cfg.training.num_samples,
        "val_grad_off": False,
    }

    trainer = DDSNPTrainerContext(**trainer_params)

    if dir is not None:
        model, trainer = load_state_dicts_np(dir, model, trainer, load_decoder_only)

    return model, trainer, test_loader, val_loader
