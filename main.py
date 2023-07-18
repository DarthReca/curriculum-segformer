import logging
from argparse import ArgumentParser

import comet_ml
import hydra
import networks
import pytorch_lightning as pl
import segmentation_transforms as transforms
import torch
from cityscape_datamodule import CityscapesDataModule
from lightning_lite import seed_everything
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    RichModelSummary,
    RichProgressBar,
    TQDMProgressBar,
)
from pytorch_lightning.loggers import CometLogger

log = logging.getLogger("debug")


@hydra.main(version_base=None, config_path="configs", config_name="default")
def hydra_main(cfg: DictConfig):
    # Deterministic settings
    seed_everything(42)
    # Datasets
    train_transforms = transforms.Compose(
        [
            transforms.PILToTensor(),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomCrop(size=1024),
        ]
    )
    test_transforms = transforms.Compose(
        [transforms.PILToTensor(), transforms.CenterCrop(size=1024)]
    )
    target_transforms = transforms.PILToTensor()
    datamodule = CityscapesDataModule(
        "data/cityscapes",
        target_type="semantic",
        num_workers=cfg.dataset.num_workers,
        batch_size=cfg.dataset.batch_size,
        train_transforms=train_transforms,
        val_transforms=test_transforms,
        test_transforms=test_transforms,
        target_transforms=target_transforms,
        sampling_milestones=cfg.dataset.sampling_milestones,
    )
    # Model
    model = networks.Segformer(
        cfg.model.decoder_config,
        cfg.model.encoder_config,
        curriculum=cfg.curriculum,
        selected_classes=cfg.dataset.selected_class,
        category_id=cfg.dataset.category_id,
        use_curriculum=cfg.curriculum.active,
        insertion_unit=cfg.curriculum.insertion_unit,
        lr_scheduler=cfg.model.lr_scheduler,
        lr=cfg.model.lr,
    )
    # DEBUG ZONE
    if cfg.debug:
        for _ in range(len(cfg.model.encoder_config.hidden_sizes)):
            model._apply_model_curriculum()
    # DEBUG ZONE END
    # Logging and Callbacks
    logger = CometLogger(
        api_key=cfg.api_key, save_dir="logs", project_name="model curriculum"
    )

    if cfg.curriculum.active:
        experiment_name = "Curriculum"
    else:
        experiment_name = "Standard"

    if cfg.model.load_pretrain:
        experiment_name += " Pretrained"
    else:
        experiment_name += " Random"

    if cfg.dataset.category_id:
        experiment_name += f" Category {cfg.dataset.category_id}"
    logger.experiment.set_name(experiment_name)

    callbacks = [
        RichModelSummary(),
        RichProgressBar(leave=True),
        ModelCheckpoint(
            dirpath=f"checkpoints/{logger.experiment.get_key()}",
            save_top_k=-1,
            every_n_epochs=50,
            save_last=True,
        ),
        LearningRateMonitor(),
    ]
    # Train
    trainer = pl.Trainer(
        deterministic=False,
        logger=logger,
        callbacks=callbacks,
        num_sanity_val_steps=1,
        **cfg.trainer,
    )
    if cfg.mode == "train":
        trainer.fit(model, datamodule=datamodule)
        logger.experiment.log_model("Best", trainer.checkpoint_callback.best_model_path)
    elif cfg.mode == "test":
        if cfg.curriculum.active:
            for _ in range(len(cfg.model.encoder_config.hidden_sizes)):
                model._apply_model_curriculum()
        trainer.test(model, datamodule=datamodule, ckpt_path=cfg.ckpt_path)
    logger.experiment.end()


if __name__ == "__main__":
    hydra_main()
