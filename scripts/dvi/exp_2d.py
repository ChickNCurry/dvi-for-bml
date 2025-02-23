from dataclasses import asdict, dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd  # type: ignore
import torch
from matplotlib import pyplot as plt
from torch.optim.adamw import AdamW
from torch.utils.data import DataLoader

import wandb
from src.architectures.dvi import DVI
from src.components.cdvi.cmcd import CMCD
from src.components.cdvi.dis import DIS
from src.components.control.aggr_control import AggrControl
from src.components.control.base_control import BaseControl
from src.components.control.bca_control import BCAControl
from src.components.control.mha_control import MHAControl
from src.components.encoder.aggr_encoder import Aggr, AggrEncoder
from src.components.encoder.base_encoder import BaseEncoder
from src.components.encoder.bca_encoder import BCAEncoder
from src.components.encoder.mha_encoder import MHAEncoder
from src.components.schedule.annealing_schedule import (
    AggrAnnealingSchedule,
    AnnealingSchedule,
    BCAAnnealingSchedule,
)
from src.components.schedule.base_schedule import BaseSchedule
from src.components.schedule.cos_noise_schedule import (
    AggrCosineNoiseSchedule,
    BCACosineNoiseSchedule,
    CosineNoiseSchedule,
)
from src.components.schedule.noise_schedule import (
    AggrNoiseSchedule,
    BCANoiseSchedule,
    NoiseSchedule,
)
from src.components.schedule.step_size_schedule import StepSizeSchedule
from src.evaluation.grid import (
    compute_jsd,
    create_grid,
    eval_dist_on_grid,
    eval_hist_on_grid,
)
from src.training.dvi_trainer import DVITrainer, DVITrainerContext
from src.utils.datasets import ContextSetDataset
from src.utils.distros import TaskPosteriorGMM
from src.utils.hash import get_object_hash, get_var_name


def run() -> None:

    device = torch.device("cpu")

    @dataclass
    class Config:
        use_score = False
        num_steps = 16
        c_dim = 2
        z_dim = 2
        h_dim = 32
        num_layers = 3
        non_linearity = "GELU"
        learning_rate = 3e-3
        size = 16
        batch_size = 16

    config = Config()

    dataset = ContextSetDataset(
        size=config.size,
        c_dim=config.c_dim,
        max_context_size=10,
        sampling_factor=4,
    )

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=config.batch_size,
        shuffle=True,
    )

    contextual_target = TaskPosteriorGMM

    # Encoders

    aggr_enc_mean = AggrEncoder(
        c_dim=config.c_dim,
        h_dim=config.h_dim,
        num_layers=config.num_layers,
        non_linearity=config.non_linearity,
        num_heads=None,
        aggregation=Aggr.MEAN,
        max_context_size=None,
    )

    torch.save(
        aggr_enc_mean.state_dict(),
        f"scripts/dvi/state_dicts/{get_object_hash(aggr_enc_mean)}.pth",
    )

    aggr_enc_mean_cse = AggrEncoder(
        c_dim=config.c_dim,
        h_dim=config.h_dim,
        num_layers=config.num_layers,
        non_linearity=config.non_linearity,
        num_heads=None,
        aggregation=Aggr.MEAN,
        max_context_size=dataset.max_context_size,
    )

    torch.save(
        aggr_enc_mean_cse.state_dict(),
        f"scripts/dvi/state_dicts/{get_object_hash(aggr_enc_mean_cse)}.pth",
    )

    aggr_enc_max = AggrEncoder(
        c_dim=config.c_dim,
        h_dim=config.h_dim,
        num_layers=config.num_layers,
        non_linearity=config.non_linearity,
        num_heads=None,
        aggregation=Aggr.MAX,
        max_context_size=None,
    )

    torch.save(
        aggr_enc_max.state_dict(),
        f"scripts/dvi/state_dicts/{get_object_hash(aggr_enc_max)}.pth",
    )

    aggr_enc_max_cse = AggrEncoder(
        c_dim=config.c_dim,
        h_dim=config.h_dim,
        num_layers=config.num_layers,
        non_linearity=config.non_linearity,
        num_heads=None,
        aggregation=Aggr.MAX,
        max_context_size=dataset.max_context_size,
    )

    torch.save(
        aggr_enc_max_cse.state_dict(),
        f"scripts/dvi/state_dicts/{get_object_hash(aggr_enc_max_cse)}.pth",
    )

    bca_encoder = BCAEncoder(
        c_dim=config.c_dim,
        h_dim=config.h_dim,
        z_dim=config.z_dim,
        num_layers=config.num_layers,
        non_linearity=config.non_linearity,
        num_heads=None,
    )

    torch.save(
        bca_encoder.state_dict(),
        f"scripts/dvi/state_dicts/{get_object_hash(bca_encoder)}.pth",
    )

    # Controls

    aggr_control = AggrControl(
        h_dim=config.h_dim,
        z_dim=config.z_dim,
        num_steps=config.num_steps,
        num_layers=config.num_layers,
        non_linearity=config.non_linearity,
        use_score=config.use_score,
    )

    torch.save(
        aggr_control.state_dict(),
        f"scripts/dvi/state_dicts/{get_object_hash(aggr_control)}.pth",
    )

    bca_control = BCAControl(
        h_dim=config.h_dim,
        z_dim=config.z_dim,
        num_steps=config.num_steps,
        num_layers=config.num_layers,
        non_linearity=config.non_linearity,
        use_score=config.use_score,
    )

    torch.save(
        bca_control.state_dict(),
        f"scripts/dvi/state_dicts/{get_object_hash(bca_control)}.pth",
    )

    # Noise Schedules

    noise_schedule = NoiseSchedule(
        z_dim=config.z_dim,
        num_steps=config.num_steps,
        device=device,
    )

    torch.save(
        noise_schedule.state_dict(),
        f"scripts/dvi/state_dicts/{get_object_hash(noise_schedule)}.pth",
    )

    cos_noise_schedule = CosineNoiseSchedule(
        z_dim=config.z_dim,
        num_steps=config.num_steps,
        device=device,
    )

    torch.save(
        cos_noise_schedule.state_dict(),
        f"scripts/dvi/state_dicts/{get_object_hash(cos_noise_schedule)}.pth",
    )

    aggr_noise_schedule = AggrNoiseSchedule(
        z_dim=config.z_dim,
        h_dim=config.h_dim,
        non_linearity=config.non_linearity,
        num_steps=config.num_steps,
        device=device,
    )

    torch.save(
        aggr_noise_schedule.state_dict(),
        f"scripts/dvi/state_dicts/{get_object_hash(aggr_noise_schedule)}.pth",
    )

    aggr_cos_noise_schedule = AggrCosineNoiseSchedule(
        z_dim=config.z_dim,
        h_dim=config.h_dim,
        non_linearity=config.non_linearity,
        num_steps=config.num_steps,
    )

    torch.save(
        aggr_cos_noise_schedule.state_dict(),
        f"scripts/dvi/state_dicts/{get_object_hash(aggr_cos_noise_schedule)}.pth",
    )

    bca_noise_schedule = BCANoiseSchedule(
        z_dim=config.z_dim,
        h_dim=config.h_dim,
        non_linearity=config.non_linearity,
        num_steps=config.num_steps,
        device=device,
    )

    torch.save(
        bca_noise_schedule.state_dict(),
        f"scripts/dvi/state_dicts/{get_object_hash(bca_noise_schedule)}.pth",
    )

    bca_cos_noise_schedule = BCACosineNoiseSchedule(
        z_dim=config.z_dim,
        h_dim=config.h_dim,
        non_linearity=config.non_linearity,
        num_steps=config.num_steps,
    )

    torch.save(
        bca_cos_noise_schedule.state_dict(),
        f"scripts/dvi/state_dicts/{get_object_hash(bca_cos_noise_schedule)}.pth",
    )

    # Annealing Schedules

    ann_schedule = AnnealingSchedule(num_steps=config.num_steps, device=device)

    torch.save(
        ann_schedule.state_dict(),
        f"scripts/dvi/state_dicts/{get_object_hash(ann_schedule)}.pth",
    )

    aggr_annealing_schedule = AggrAnnealingSchedule(
        h_dim=config.h_dim,
        non_linearity=config.non_linearity,
        num_steps=config.num_steps,
        device=device,
    )

    torch.save(
        aggr_annealing_schedule.state_dict(),
        f"scripts/dvi/state_dicts/{get_object_hash(aggr_annealing_schedule)}.pth",
    )

    bca_annealing_schedule = BCAAnnealingSchedule(
        h_dim=config.h_dim,
        non_linearity=config.non_linearity,
        num_steps=config.num_steps,
        device=device,
    )

    torch.save(
        bca_annealing_schedule.state_dict(),
        f"scripts/dvi/state_dicts/{get_object_hash(bca_annealing_schedule)}.pth",
    )

    # Step Size Schedules

    step_size_schedule = StepSizeSchedule(num_steps=config.num_steps, device=device)

    variations: List[Tuple[BaseControl, BaseSchedule, BaseSchedule, BaseEncoder]] = [
        (aggr_control, noise_schedule, ann_schedule, aggr_enc_mean),
        (aggr_control, noise_schedule, ann_schedule, aggr_enc_max),
        (aggr_control, noise_schedule, ann_schedule, aggr_enc_mean_cse),
        (aggr_control, noise_schedule, ann_schedule, aggr_enc_max_cse),
        (aggr_control, cos_noise_schedule, ann_schedule, aggr_enc_mean),
        (aggr_control, cos_noise_schedule, ann_schedule, aggr_enc_max),
        (aggr_control, cos_noise_schedule, ann_schedule, aggr_enc_mean_cse),
        (aggr_control, cos_noise_schedule, ann_schedule, aggr_enc_max_cse),
        (aggr_control, aggr_noise_schedule, ann_schedule, aggr_enc_mean),
        (aggr_control, aggr_noise_schedule, ann_schedule, aggr_enc_max),
        (aggr_control, aggr_noise_schedule, ann_schedule, aggr_enc_mean_cse),
        (aggr_control, aggr_noise_schedule, ann_schedule, aggr_enc_max_cse),
        (aggr_control, aggr_cos_noise_schedule, ann_schedule, aggr_enc_mean),
        (aggr_control, aggr_cos_noise_schedule, ann_schedule, aggr_enc_max),
        (aggr_control, aggr_cos_noise_schedule, ann_schedule, aggr_enc_mean_cse),
        (aggr_control, aggr_cos_noise_schedule, ann_schedule, aggr_enc_max_cse),
        (bca_control, noise_schedule, ann_schedule, bca_encoder),
        (bca_control, cos_noise_schedule, ann_schedule, bca_encoder),
        (bca_control, bca_noise_schedule, ann_schedule, bca_encoder),
        (bca_control, bca_cos_noise_schedule, ann_schedule, bca_encoder),
    ]

    for comps in variations:

        for c in comps:
            c.load_state_dict(
                torch.load(f"scripts/dvi/state_dicts/{get_object_hash(c)}.pth")
            )
            print(f"{c.__class__.__name__} loaded")

        id = "-".join([f"{obj.__class__.__name__}" for obj in comps])
        id += f"-use_score:{config.use_score}"

        if hasattr(comps[3], "aggregation"):
            id += f"-{comps[3].aggregation}"

        if hasattr(comps[3], "max_context_size"):
            id += f"-{comps[3].max_context_size}"

        cdvi = DIS(
            z_dim=config.z_dim,
            num_steps=config.num_steps,
            control=comps[0],
            step_size_schedule=step_size_schedule,
            noise_schedule=comps[1],
            annealing_schedule=comps[2],
            use_score=config.use_score,
            device=device,
        )

        # cdvi = CMCD(
        #     z_dim=config.z_dim,
        #     num_steps=config.num_steps,
        #     control=comps[0],
        #     step_size_schedule=step_size_schedule,
        #     noise_schedule=comps[1],
        #     annealing_schedule=comps[2],
        #     device=device,
        # )

        model = DVI(
            encoder=comps[3],
            cdvi=cdvi,
            contextual_target=contextual_target,
        )

        optimizer = AdamW(cdvi.parameters(), lr=config.learning_rate)

        trainer = DVITrainerContext(
            model=model,
            device=device,
            dataset=dataset,
            train_loader=dataloader,
            val_loader=dataloader,
            optimizer=optimizer,
            scheduler=None,
            generator=torch.Generator(),
            wandb_logging=False,
            num_subtasks=32,
            sample_size=32,
        )

        trainer.wandb_logging = True
        if trainer.wandb_logging:
            wandb.init(
                project="local-toytask-auto",
                name=id,
                group=f"use_score:{config.use_score}",
                config=asdict(config),
            )

        num_epochs = 5000

        trainer.train(num_epochs=num_epochs, max_clip_norm=None, alpha=None)

        for entry in [(1, 1), (-1, 1), (1, -1), (-1, -1)]:

            name = f"{id}-({str(entry)})"

            num_samples = 1600  # 8192
            bins = 40
            plot_range = [(-5.0, 5.0), (-5.0, 5.0)]
            nrows = dataset.max_context_size

            context = dataset.sampling_factor * torch.rand(
                (1, num_samples, nrows, config.c_dim), device=device
            )
            context[:, :, :, 0] = context[:, :, :, 0] * entry[0]
            context[:, :, :, 1] = context[:, :, :, 1] * entry[1]

            fig = plt.figure(figsize=(9, 3 * nrows), constrained_layout=True)
            subfigs = fig.subfigures(nrows=nrows, ncols=1)

            fig.suptitle(name)

            jsds = []

            for row, subfig in enumerate(subfigs):

                subfig.suptitle(f"context size: {row + 1}")
                ax = subfig.subplots(nrows=1, ncols=3, width_ratios=[1, 1, 1])

                context_size = row + 1
                sub_context = context[:, :, :context_size, :]

                assert model.contextual_target is not None

                target = model.contextual_target(sub_context, None)

                r = model.encoder(sub_context.to(device), None)
                _, z_samples = model.cdvi.run_both_processes(target, r, None)

                z_0_samples = z_samples[0].detach().cpu().numpy()
                z_T_samples = z_samples[-1].detach().cpu().numpy()
                z_trajectories = [
                    [z[0, i, :].detach().cpu().numpy() for z in z_samples]
                    for i in range(num_samples)
                ]
                z_target_samples = target.sample().detach().cpu().numpy()

                z_0_samples = z_0_samples.reshape(-1, z_0_samples.shape[-1])
                z_T_samples = z_T_samples.reshape(-1, z_T_samples.shape[-1])
                z_target_samples = z_target_samples.reshape(
                    -1, z_target_samples.shape[-1]
                )

                ax[0].hist2d(
                    z_0_samples[:, 0], z_0_samples[:, 1], bins=bins, range=plot_range
                )
                ax[0].set_title("prior $q_\phi(z_0)$")

                ax[1].hist2d(
                    z_T_samples[:, 0], z_T_samples[:, 1], bins=bins, range=plot_range
                )
                ax[1].set_title("marginal $q_\phi(z_T|z_{0:T-1},c)$")

                ax[2].hist2d(
                    z_target_samples[:, 0],
                    z_target_samples[:, 1],
                    bins=bins,
                    range=plot_range,
                )
                ax[2].set_title("target $p(z_T|c)$")

                for a in ax:
                    a.axis("off")

                num_cells = int(np.sqrt(context.shape[1]))
                grid = create_grid(plot_range, num_cells)

                dvi_vals = eval_hist_on_grid(z_T_samples, plot_range, num_cells)
                target_vals = eval_dist_on_grid(grid, target, device=device).squeeze(0)

                jsd = compute_jsd(dvi_vals, target_vals)

                jsds.append(jsd)

            df = pd.DataFrame({name: jsds}, index=[row + 1 for row in range(nrows)])
            df.to_csv(f"scripts/dvi/use_score_{config.use_score}/csv/{name}.csv")

            plt.savefig(f"scripts/dvi/use_score_{config.use_score}/png/{name}.png")

        wandb.finish()


if __name__ == "__main__":
    run()
