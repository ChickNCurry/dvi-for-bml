import torch
import wandb
from hydra import compose, initialize

from dviforbml.evaluation.visualization.visualize_np import visualize_np
from dviforbml.evaluation.visualization.visualize_task import visualize_task
from dviforbml.utils.helper import download_run
from dviforbml.utils.load_np import load_np


def main() -> None:
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        try:
            import torch_directml

            device = torch_directml.device()
        except ImportError:
            device = torch.device("cpu")

    # project = "cluster-np"
    # name = "1-mean-cnp-data-None-0"
    # dir = download_run(project, name)

    dir = "../models/lnp"

    with initialize(version_base=None, config_path=dir):
        cfg = compose(config_name="cfg")

        model, trainer, test_loader, _ = load_np(
            cfg=cfg,
            device=device,
            dir=dir,
        )

    # visualize_task(test_loader)

    trainer.wandb_logging = False
    if trainer.wandb_logging:
        wandb.init(project="local-np-sine")

    train = False
    if train:
        num_epochs = 200
        trainer.train(
            num_epochs=num_epochs,
            max_clip_norm=cfg.training.max_clip_norm,
            alpha=cfg.training.alpha,
            validate=True,
        )
        # torch.save(dvinp.state_dict(), f"{dir}/dvinp.pth")

    max_context_size = 9
    num_samples = 100  # num_cells will be root

    visualize_np(
        model=model,
        device=device,
        dataloader=test_loader,
        num_samples=num_samples,
        max_context_size=max_context_size,
        show_sigma=False,
    )


if __name__ == "__main__":
    main()
