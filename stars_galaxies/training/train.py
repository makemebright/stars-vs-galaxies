import os

import git
import hydra
import matplotlib.pyplot as plt
import mlflow
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import Callback

from stars_galaxies.data.datamodule import GalaxyStarDataModule
from stars_galaxies.models.model import GalaxyStarClassifier

PLOTS_DIR = "plots"
os.makedirs(PLOTS_DIR, exist_ok=True)


class MLflowLoggerCallback(Callback):
    def __init__(self):
        super().__init__()
        self.history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "val_f1": [],
            "val_roc_auc": [],
        }

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        metrics = trainer.callback_metrics
        step = trainer.current_epoch
        for name in self.history.keys():
            value = metrics.get(name, float("nan"))
            if hasattr(value, "item"):
                value = value.item()
            self.history[name].append(value)
            mlflow.log_metric(name, value, step=step)


def main(cfg: DictConfig):
    torch.manual_seed(42)
    print("===== CONFIG =====")
    print(OmegaConf.to_yaml(cfg))

    # MLflow init
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment("stars_vs_galaxies")
    mlflow.start_run()
    mlflow.log_params(
        {
            "lr": cfg.model.lr,
            "batch_size": cfg.data.batch_size,
            "max_epochs": cfg.trainer.max_epochs,
        }
    )
    try:
        repo = git.Repo(search_parent_directories=True)
        mlflow.log_param("git_commit", repo.head.object.hexsha)
    except Exception:
        mlflow.log_param("git_commit", "unknown")

    # Data
    datamodule = GalaxyStarDataModule(
        data_dir=cfg.data.data_dir,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        image_size=cfg.data.image_size,
    )

    # Model
    model = GalaxyStarClassifier(lr=cfg.model.lr)

    # Trainer
    mlflow_callback = MLflowLoggerCallback()
    trainer = Trainer(
        max_epochs=cfg.trainer.max_epochs,
        accelerator="cpu" if not torch.cuda.is_available() else "gpu",
        devices=1,
        callbacks=[mlflow_callback],
        log_every_n_steps=10,
    )

    trainer.fit(model, datamodule=datamodule)

    # Checkpoint + ONNX
    os.makedirs("stars_galaxies/checkpoints", exist_ok=True)
    ckpt_path = "stars_galaxies/checkpoints/galaxy_star_model.ckpt"
    trainer.save_checkpoint(ckpt_path)
    print(f"Checkpoint сохранен: {ckpt_path}")

    dummy_input = torch.randn(1, 3, cfg.data.image_size, cfg.data.image_size)
    onnx_path = "stars_galaxies/checkpoints/galaxy_star_model.onnx"
    model.to_onnx(onnx_path, dummy_input)
    print(f"ONNX модель сохранена: {onnx_path}")

    # Графики
    def plot_metric(values, name):
        plt.figure()
        plt.plot(range(len(values)), values, marker="o")
        plt.xlabel("Epoch")
        plt.ylabel(name)
        plt.title(f"{name} per epoch")
        plt.grid(True)
        plt.savefig(os.path.join(PLOTS_DIR, f"{name}.png"))
        plt.close()

    for name, values in mlflow_callback.history.items():
        plot_metric(values, name)

    # Log artifacts
    for f in os.listdir(PLOTS_DIR):
        mlflow.log_artifact(os.path.join(PLOTS_DIR, f))

    mlflow.end_run()
    print(f"Графики сохранены в {PLOTS_DIR} и залогированы в MLflow.")


def run():
    """Вызывается из commands.py"""
    from hydra import compose, initialize

    with initialize(config_path="../configs"):
        cfg = compose(config_name="config")
        main(cfg)
