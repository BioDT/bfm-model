"""
Copyright (C) 2025 TNO, The Netherlands. All rights reserved.
"""

import hydra
import torch
import torch.distributed as dist
from hydra.core.hydra_config import HydraConfig
from lightning.pytorch import seed_everything
from omegaconf import OmegaConf

from bfm_model.bfm.dataloader_helpers import get_train_dataloader, get_val_dataloader
from bfm_model.bfm.model_helpers import (
    find_checkpoint_to_resume_from,
    get_mlflow_logger,
    get_trainer,
    post_training_get_last_checkpoint,
    setup_bfm_model,
    setup_checkpoint_callback,
    setup_fsdp,
)


@hydra.main(version_base=None, config_path="configs", config_name="train_config")
def main(cfg):
    # Setup config
    print(OmegaConf.to_yaml(cfg))
    # Set the internal precision for TensorCores (H100s)
    torch.set_float32_matmul_precision(cfg.training.precision_in)

    # Seed the experiment for numpy, torch and python.random.
    seed_everything(42, workers=True)

    output_dir = HydraConfig.get().runtime.output_dir
    print(f"Output directory: {output_dir}")

    # dataloaders
    train_dataloader = get_train_dataloader(cfg)
    val_dataloader = get_val_dataloader(cfg)

    experiment_name = "BFM-train"
    mlflow_logger = get_mlflow_logger(output_dir, experiment_name=experiment_name)
    # also log in the ./mlruns folder so that you can run mlflow server and see every run together
    mlflow_logger_current_folder = get_mlflow_logger(experiment_name=experiment_name)
    loggers = [l for l in [mlflow_logger, mlflow_logger_current_folder] if l]

    model = setup_bfm_model(cfg, mode="train")

    checkpoint_callback = setup_checkpoint_callback(cfg, output_dir)

    distr_strategy = setup_fsdp(cfg, model)

    trainer = get_trainer(cfg, mlflow_logger=loggers, distr_strategy=distr_strategy, callbacks=[checkpoint_callback])
    # Experimental
    # mlflow.set_tracking_uri(output_dir)
    # Auto log all MLflow entities
    # mlflow.pytorch.autolog()

    # with mlflow.start_run() as run:
    checkpoint_path = find_checkpoint_to_resume_from(cfg)

    # do the actual training
    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader, ckpt_path=checkpoint_path)

    # NOT NEEDED: we are already at the last
    # selected_ckpt = post_training_get_last_checkpoint(
    #     output_dir=output_dir, checkpoint_callback=checkpoint_callback, trainer=trainer
    # )

    # with FSDP, get some shape errors. Only do a test when not fsdp
    if cfg.training.strategy != "fsdp":
        trainer.test(model=model, dataloaders=val_dataloader)

    print("Finished testing successfully")
    trainer.print(torch.cuda.memory_summary())

    # Optional: clean up the distributed process group
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
