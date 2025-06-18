"""
Copyright (C) 2025 TNO, The Netherlands. All rights reserved.
"""

import os
from collections import defaultdict
from pathlib import Path

import hydra
import lightning as L
import torch
from hydra.core.hydra_config import HydraConfig
from lightning.pytorch import seed_everything
from omegaconf import DictConfig, OmegaConf

from bfm_model.bfm.dataloader_helpers import get_val_dataloader
from bfm_model.bfm.dataloader_monthly import (
    LargeClimateDataset,
    _convert,
    custom_collate,
)
from bfm_model.bfm.model import BFM
from bfm_model.bfm.model_helpers import get_mlflow_logger, get_trainer, setup_bfm_model
from bfm_model.bfm.test_lighting import BFM_lighting


@hydra.main(version_base=None, config_path="configs", config_name="train_config")
def main(cfg: DictConfig):
    """
    Test/inference script using a PyTorch Lightning module.

    Args:
        checkpoint_path (str): Path to the trained checkpoint (.ckpt).
        data_dir (str): Directory containing test data.
        batch_size (int): Batch size for test loader.
        num_workers (int): Number of workers for DataLoader.
        gpus (int): Number of GPUs to use (if 0, run on CPU).
        precision (int): Float precision (16 for half, 32 for single, etc.).
        accelerator (str): "gpu", "cpu", "tpu", etc.

    Returns:
        test_results (dict or list): Test metrics returned by trainer.test().
    """

    # Setup config
    print(OmegaConf.to_yaml(cfg))
    # Set the internal precision for TensorCores (H100s)
    torch.set_float32_matmul_precision(cfg.training.precision_in)

    # Seed the experiment for numpy, torch and python.random.
    seed_everything(42, workers=True)

    output_dir = HydraConfig.get().runtime.output_dir
    print(f"Output directory: {output_dir}")

    # dataloaders
    # also need test_dataset for scaling
    test_dataset = LargeClimateDataset(
        data_dir=cfg.data.test_data_path,
        scaling_settings=cfg.data.scaling,
        num_species=cfg.data.species_number,
        atmos_levels=cfg.data.atmos_levels,
        model_patch_size=cfg.model.patch_size,
    )
    # override batch_size
    test_dataloader = get_val_dataloader(cfg, batch_size_override=cfg.evaluation.batch_size)

    mlflow_logger = get_mlflow_logger(output_dir)
    # also log in the ./mlruns folder so that you can run mlflow server and see every run together
    mlflow_logger_current_folder = get_mlflow_logger()
    loggers = [l for l in [mlflow_logger, mlflow_logger_current_folder] if l]

    # trainer = L.Trainer(
    #     accelerator=cfg.training.accelerator,
    #     devices=cfg.training.devices,
    #     precision=cfg.training.precision,
    #     log_every_n_steps=cfg.training.log_steps,
    #     # limit_test_batches=1,
    #     limit_predict_batches=12,  # TODO Change this to select how many consecutive months you want to predict
    #     logger=loggers,
    #     enable_checkpointing=False,
    #     enable_progress_bar=True,
    # )
    trainer = get_trainer(
        cfg,
        mlflow_logger=loggers,
        callbacks=[],
    )

    bfm_model = setup_bfm_model(cfg, mode="test")

    checkpoint_path = cfg.evaluation.checkpoint_path
    # Load Model from Checkpoint
    print(f"Loading model from checkpoint: {checkpoint_path}")
    # Do the inference
    # test_results = trainer.test(model=bfm_model, ckpt_path=checkpoint_path, dataloaders=test_dataloader)
    predictions = trainer.predict(model=bfm_model, ckpt_path=checkpoint_path, dataloaders=test_dataloader)
    print("=== Test Results ===")
    SAVE_DIR = Path("pre-train_test_exports")
    SAVE_DIR.mkdir(exist_ok=True, parents=True)

    windows: defaultdict[int, dict] = defaultdict(dict)

    if trainer.global_rank == 0:
        for batch in predictions:
            for rec in batch:
                idx = rec["idx"]

                # scale tensors to original space before CPU conversion
                pred_scaled = test_dataset.scale_batch(rec["pred"], direction="original")
                gt_scaled = test_dataset.scale_batch(rec["gt"], direction="original")
                # For RAW unscaled
                # pred_scaled = rec["pred"]
                # gt_scaled = rec["gt"]

                windows[idx] = {
                    "pred": _convert(pred_scaled),
                    "gt": _convert(gt_scaled, move_cpu=True),
                }

        for idx, payload in windows.items():
            path = SAVE_DIR / f"window_{idx:05d}.pt"
            torch.save(payload, path)
            print(f"Saved {path}")


if __name__ == "__main__":
    main()
