import hydra
from omegaconf import OmegaConf, DictConfig
import torch
import wandb
from src.context_datasets import MetaLearningDataset
from src.control_function import ControlFunction
from src.decoder import Decoder
import src.dvi_process
from src.encoder import SetEncoder
import metalearning_benchmarks  # type: ignore
from torch.utils.data import DataLoader
from src.train import train


@hydra.main(config_path="config", config_name="config")
def run(config: DictConfig) -> None:

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    benchmark = getattr(
        getattr(metalearning_benchmarks, config.benchmark.package),
        config.benchmark.name,
    )(
        n_task=config.benchmark.n_task,
        n_datapoints_per_task=config.benchmark.n_datapoints_per_task,
        output_noise=config.benchmark.output_noise,
        seed_task=config.benchmark.seed_task,
        seed_x=config.benchmark.seed_x,
        seed_noise=config.benchmark.seed_noise,
    )

    dataset = MetaLearningDataset(benchmark=benchmark)
    dataloader = DataLoader(
        dataset=dataset, batch_size=config.training.batch_size, shuffle=True
    )

    set_encoder = SetEncoder(
        c_dim=config.common.c_dim,
        h_dim=config.common.h_dim,
        num_layers=config.common.num_layers,
        non_linearity=config.common.non_linearity,
        is_attentive=config.set_encoder.is_attentive,
        is_aggregative=not config.dvi_process.is_cross_attentive
        or not config.decoder.is_cross_attentive,
        is_non_aggregative=config.dvi_process.is_cross_attentive
        or config.decoder.is_cross_attentive,
        use_context_size=config.set_encoder.use_context_size,
        aggregation=config.set_encoder.aggregation,
        max_context_size=dataset.max_context_size,
    ).to(device)

    control_function = ControlFunction(
        h_dim=config.common.h_dim,
        z_dim=config.common.z_dim,
        num_layers=config.common.num_layers,
        non_linearity=config.common.non_linearity,
        num_steps=config.dvi_process.num_steps,
        is_cross_attentive=config.dvi_process.is_cross_attentive,
    ).to(device)

    dvi_process = getattr(src.dvi_process, config.dvi_process.variant)(
        z_dim=config.common.z_dim,
        num_steps=config.dvi_process.num_steps,
        control_function=control_function,
        device=device,
    ).to(device)

    decoder = Decoder(
        x_dim=config.common.x_dim,
        z_dim=config.common.z_dim,
        h_dim=config.common.h_dim,
        y_dim=config.common.y_dim,
        num_layers=config.common.num_layers,
        non_linearity=config.common.non_linearity,
        has_lat_path=config.decoder.has_lat_path,
        has_det_path=config.decoder.has_det_path,
        is_cross_attentive=config.decoder.is_cross_attentive,
    ).to(device)

    if config.training.wandb_logging:
        wandb.init(project="dvi-bml", config=OmegaConf.to_container(config, resolve=True))  # type: ignore

    params = [
        {"params": dvi_process.parameters(), "lr": config.training.learning_rate},
        {"params": set_encoder.parameters(), "lr": config.training.learning_rate},
        {"params": decoder.parameters(), "lr": config.training.learning_rate},
    ]

    optimizer = torch.optim.Adam(params, lr=config.training.learning_rate)

    train(
        dvi_process=dvi_process,
        encoder=set_encoder,
        device=device,
        num_epochs=config.training.num_epochs,
        dataloader=dataloader,
        target_constructor=None,
        optimizer=optimizer,
        scheduler=None,
        wandb_logging=config.training.wandb_logging,
        decoder=decoder,
    )

    wandb.finish()


if __name__ == "__main__":
    run()
