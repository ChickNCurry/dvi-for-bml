import warnings

import torch
import wandb
from hydra import compose, initialize

from dviforbml.evaluation.visualization.visualize_dvinp import visualize_dvinp
from dviforbml.evaluation.visualization.visualize_task import visualize_task
from dviforbml.utils.helper import download_run, get_name_dvinp
from dviforbml.utils.load_dvinp import load_dvinp


def main() -> None:
    # warnings.filterwarnings("ignore")

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        try:
            import torch_directml  # type: ignore

            device = torch_directml.device()
        except ImportError:
            device = torch.device("cpu")

    # project = "cluster-dvinp-linesine"
    # name = "32-None-bca-cos-dis-True-cntxt"
    # dir = download_run(project, name)

    dir = "models/dvinp"

    with initialize(version_base=None, config_path=f"../{dir}"):
        cfg = compose(config_name="cfg")

        model, trainer, test_loader, _ = load_dvinp(
            cfg=cfg, device=device, dir=dir, load_decoder_only=True, train_decoder=False
        )

    # visualize_task(test_loader)

    trainer.wandb_logging = False
    if trainer.wandb_logging:
        wandb.init(project="test", name=get_name_dvinp(cfg))

    train = False
    if train:
        # trainer.optimizer.param_groups[0]["lr"] = 0.1 * cfg.training.learning_rate
        # print(trainer.optimizer.param_groups[0]["lr"])

        num_epochs = 10
        trainer.train(
            num_epochs=num_epochs,
            max_clip_norm=cfg.training.max_clip_norm,
            alpha=cfg.training.alpha,
            validate=True,
        )

        # torch.save(model.state_dict(), f"{dir}/model.pth")

    max_context_size = 9
    num_samples = 900  # num_cells will be root
    ranges = [(-5, 5), (-5, 5)]

    visualize_dvinp(
        device=device,
        dvinp=model,
        dataloader=test_loader,
        num_samples=num_samples,
        max_context_size=max_context_size,
        ranges=ranges,
        save_dir=dir,
    )


if __name__ == "__main__":
    main()
