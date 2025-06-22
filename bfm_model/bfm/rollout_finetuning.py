"""
Copyright (C) 2025 TNO, The Netherlands. All rights reserved.
"""

import os
from functools import partial
from pathlib import Path

import hydra
import lightning as L
import torch
import torch.distributed as dist
from hydra.core.hydra_config import HydraConfig
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.strategies import DDPStrategy, FSDPStrategy
from lightning.pytorch.utilities.model_summary import ModelSummary
from omegaconf import DictConfig, OmegaConf
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
from torch.utils.data import DataLoader

from bfm_model.bfm.dataloader_helpers import SequentialWindowDataset
from bfm_model.bfm.dataloader_monthly import LargeClimateDataset, custom_collate
from bfm_model.bfm.model_helpers import get_mlflow_logger, setup_bfm_model, setup_fsdp, get_trainer


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
    MODE = cfg.finetune.mode
    # Setup config
    print(OmegaConf.to_yaml(cfg))
    # Set the internal precision for TensorCores (H100s)
    torch.set_float32_matmul_precision(cfg.training.precision_in)

    # Seed the experiment for numpy, torch and python.random.
    seed_everything(42, workers=True)

    output_dir = HydraConfig.get().runtime.output_dir
    print(f"Output directory: {output_dir}")
    dataset = LargeClimateDataset(
        data_dir=cfg.data.data_path,
        scaling_settings=cfg.data.scaling,
        num_species=cfg.data.species_number,
        mode="finetune",
        atmos_levels=cfg.data.atmos_levels,
    )
    test_dataset = LargeClimateDataset(
        data_dir=cfg.data.test_data_path,
        scaling_settings=cfg.data.scaling,
        num_species=cfg.data.species_number,
        mode="finetune",
        atmos_levels=cfg.data.atmos_levels,
    )
    seq_dataset = SequentialWindowDataset(dataset, cfg.finetune.rollout_steps)
    seq_test_dataset = SequentialWindowDataset(test_dataset, cfg.finetune.rollout_steps)

    val_dataloader = DataLoader(
        seq_test_dataset,
        batch_size=cfg.finetune.batch_size,
        num_workers=cfg.training.workers,
        collate_fn=custom_collate,
        drop_last=False,
        shuffle=False,
    )

    train_dataloader = DataLoader(
        seq_dataset,
        shuffle=False,  # We need to keep the dates
        batch_size=cfg.finetune.batch_size,
        num_workers=cfg.training.workers,
        collate_fn=custom_collate,
        drop_last=False,
        pin_memory=True,
    )

    print(f"Setting up Daloaders with length train: {len(train_dataloader)} and test: {len(val_dataloader)}")

    mlflow_logger = get_mlflow_logger(output_dir)
    # also log in the ./mlruns folder so that you can run mlflow server and see every run together
    mlflow_logger_current_folder = get_mlflow_logger()
    loggers = [l for l in [mlflow_logger, mlflow_logger_current_folder] if l]

    checkpoint_path = cfg.finetune.checkpoint_path
    # Load Model from Checkpoint
    print(f"Loading model from checkpoint: {checkpoint_path}")
    model = setup_bfm_model(cfg, mode="rollout", checkpoint_path=checkpoint_path)

    model_summary = ModelSummary(model, max_depth=2)
    print(model_summary)

    checkpoint_callback = ModelCheckpoint(
        dirpath=f"{output_dir}/checkpoints",
        save_top_k=1,
        monitor="val_loss",  # `log('val_loss', value)` in the `LightningModule`
        mode="min",
        every_n_train_steps=cfg.finetune.checkpoint_every,
        filename="{epoch:02d}-{train_loss}",
        save_last=True,
    )

    print(f"Will be saving checkpoints at: {output_dir}/checkpoints")
    
    # ignored = [p for n,p in model.named_parameters() if "peft_*" in n]

    if cfg.training.strategy == "fsdp":
        distr_strategy = FSDPStrategy(
            sharding_strategy="FULL_SHARD",
            auto_wrap_policy=partial(size_based_auto_wrap_policy, min_num_params=1e6),
            # ignored_states=ignored,
            # activation_checkpointing_policy=activation_ckpt_policy,
        )
        print(f"Using {cfg.training.strategy} strategy: {distr_strategy}")
    elif cfg.training.strategy == "ddp":
        distr_strategy = DDPStrategy()
        print(f"Using {cfg.training.strategy} strategy: {distr_strategy}")
    else:
        distr_strategy = "auto"

    trainer = L.Trainer(
        max_epochs=cfg.finetune.epochs,
        accelerator=cfg.training.accelerator,
        devices=cfg.training.devices,
        precision=cfg.training.precision,
        strategy=distr_strategy,
        num_nodes=cfg.training.num_nodes,
        log_every_n_steps=cfg.training.log_steps,
        logger=loggers,  # Only the rank 0 process will have a logger
        limit_train_batches=10,  # Process 10 batches per epoch.
        limit_val_batches=1,
        limit_test_batches=2,
        limit_predict_batches=2,
        val_check_interval=cfg.finetune.eval_every,  # Run validation every n training batches.
        check_val_every_n_epoch=None,
        # limit_train_batches=1, # For debugging to see what happens at the end of epoch
        # check_val_every_n_epoch=None,  # Do eval every n epochs
        # val_check_interval=3, # Does not work in Distributed settings | Do eval every 10 training steps => 10 steps x 8 batch_size = Every 80 Batches
        callbacks=[checkpoint_callback, DeviceSummary(), FSDPShapeGuard()],
        # callbacks=[RolloutSaveCallback()],
        # plugins=[MyClusterEnvironment()],
    )

    # trainer = get_trainer(cfg, mlflow_logger=loggers, distr_strategy=distr_strategy, callbacks=[checkpoint_callback])

    if cfg.finetune.prediction:
        print(f"Will be doing {cfg.finetune.rollout_steps} - steps prediction and storing the results")
        all_outputs = trainer.predict(model, dataloaders=val_dataloader)

        # Flatten the nested lists:
        flat = [rec for batch_list in all_outputs for rec in batch_list]

        # Group by window index and save:
        SAVE_DIR = Path("rollout_exports")
        SAVE_DIR.mkdir(exist_ok=True, parents=True)

        by_window = {}
        for rec in flat:
            idx = rec["idx"]
            by_window.setdefault(idx, []).append(rec)

        for idx, recs in by_window.items():
            recs = sorted(recs, key=lambda r: r["step"])
            path = SAVE_DIR / f"window_{idx:05d}.pt"
            # each rec already has pred/gt = detached CPU Batches
            torch.save(recs, path)
            print(f"Saved {path} ({len(recs)} steps)")
    else:
        print(f"Starting {MODE} Finetune training from scratch for a horizon of {cfg.finetune.rollout_steps} ")
        trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

    if dist.is_initialized():
        dist.barrier()

    selected_ckpt = checkpoint_callback.best_model_path or cfg.finetune.checkpoint_path
    if not os.path.exists(selected_ckpt):
        raise FileNotFoundError(f"Checkpoint not found at {selected_ckpt}")

    if trainer.is_global_zero:
        print(f"[Rank 0] Using checkpoint: {selected_ckpt}")

    # broadcast checkpoint path
    if dist.is_initialized():
        ckpt_list = [selected_ckpt]
        dist.broadcast_object_list(ckpt_list, src=0)
        selected_ckpt = ckpt_list[0]


if __name__ == "__main__":
    main()
