"""
Copyright 2025 (C) TNO. Licensed under the MIT license.
"""

import os
from datetime import datetime
from functools import partial
from typing import List, Literal

import lightning as L
import torch
import torch.distributed as dist
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import MLFlowLogger
from lightning.pytorch.strategies import DDPStrategy, FSDPStrategy
from lightning.pytorch.utilities.model_summary import ModelSummary
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    apply_activation_checkpointing,
    checkpoint_wrapper,
)
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy

from bfm_model.bfm.model import BFM, BFMRollout
from bfm_model.mvit.mvit_model import MViT
from bfm_model.swin_transformer.core.swim_core_v2 import Swin3DTransformer


def activation_ckpt_policy(module):
    return isinstance(module, (Swin3DTransformer, MViT))


def get_mlflow_logger(output_dir: str | None = None, experiment_name: str = "BFM") -> MLFlowLogger | None:
    # Setup logger with rank-specific paths to avoid conflicts
    current_time = datetime.now()
    rank = int(os.environ.get("RANK", 0))
    print(f"Will be using rank {rank} for logging")
    # Single logger approach with rank-specific paths
    mlflow_logger = None
    # if "RANK" not in os.environ or os.environ["RANK"] == "0":
    if rank == 0 or rank == "0":
        if output_dir:
            # Use rank in experiment name to avoid conflicts
            mlflow_path = f"{output_dir}/logs/rank{rank}"
            mlflow_logger = MLFlowLogger(experiment_name=experiment_name, run_name=f"BFM_{current_time}", save_dir=mlflow_path)
        else:
            # this one will create in current directory ./mlruns
            mlflow_logger = MLFlowLogger(experiment_name=experiment_name, run_name=f"BFM_{current_time}")

        print(f"mlflow configured to log to {mlflow_logger.log_dir}")

    return mlflow_logger


def setup_bfm_model(cfg, mode: Literal["train", "test", "rollout"], checkpoint_path: str | None = None) -> BFM:
    swin_params = {}
    if cfg.model.backbone == "swin":
        selected_swin_config = cfg.model_swin_backbone[cfg.model.swin_backbone_size]
        swin_params = {
            "swin_encoder_depths": tuple(selected_swin_config.encoder_depths),
            "swin_encoder_num_heads": tuple(selected_swin_config.encoder_num_heads),
            "swin_decoder_depths": tuple(selected_swin_config.decoder_depths),
            "swin_decoder_num_heads": tuple(selected_swin_config.decoder_num_heads),
            "swin_window_size": tuple(selected_swin_config.window_size),
            "swin_mlp_ratio": selected_swin_config.mlp_ratio,
            "swin_qkv_bias": selected_swin_config.qkv_bias,
            "swin_drop_rate": selected_swin_config.drop_rate,
            "swin_attn_drop_rate": selected_swin_config.attn_drop_rate,
            "swin_drop_path_rate": selected_swin_config.drop_path_rate,
            "use_lora": selected_swin_config.use_lora,
        }

    if mode == "train":
        model = BFM(
            surface_vars=(cfg.model.surface_vars),
            edaphic_vars=(cfg.model.edaphic_vars),
            atmos_vars=(cfg.model.atmos_vars),
            climate_vars=(cfg.model.climate_vars),
            species_vars=(cfg.model.species_vars),
            vegetation_vars=(cfg.model.vegetation_vars),
            land_vars=(cfg.model.land_vars),
            agriculture_vars=(cfg.model.agriculture_vars),
            forest_vars=(cfg.model.forest_vars),
            redlist_vars=(cfg.model.redlist_vars),
            misc_vars=(cfg.model.misc_vars),
            atmos_levels=cfg.data.atmos_levels,
            species_num=cfg.data.species_number,
            H=cfg.model.H,
            W=cfg.model.W,
            num_latent_tokens=cfg.model.num_latent_tokens,
            backbone_type=cfg.model.backbone,
            patch_size=cfg.model.patch_size,
            embed_dim=cfg.model.embed_dim,
            num_heads=cfg.model.num_heads,
            head_dim=cfg.model.head_dim,
            depth=cfg.model.depth,
            learning_rate=cfg.training.lr,
            weight_decay=cfg.training.wd,
            batch_size=cfg.training.batch_size,
            td_learning=cfg.training.td_learning,
            land_mask_path=cfg.data.land_mask_path,
            use_mask=cfg.training.use_mask,
            partially_masked_groups=cfg.training.partially_masked_groups,
            **swin_params,
        )
        # BFM = torch.compile(model)
        apply_activation_checkpointing(model, checkpoint_wrapper_fn=checkpoint_wrapper, check_fn=activation_ckpt_policy)

    elif mode == "test":
        # force batch_size to 1 in test mode
        model = BFM(
            surface_vars=(cfg.model.surface_vars),
            edaphic_vars=(cfg.model.edaphic_vars),
            atmos_vars=(cfg.model.atmos_vars),
            climate_vars=(cfg.model.climate_vars),
            species_vars=(cfg.model.species_vars),
            vegetation_vars=(cfg.model.vegetation_vars),
            land_vars=(cfg.model.land_vars),
            agriculture_vars=(cfg.model.agriculture_vars),
            forest_vars=(cfg.model.forest_vars),
            redlist_vars=(cfg.model.redlist_vars),
            misc_vars=(cfg.model.misc_vars),
            atmos_levels=cfg.data.atmos_levels,
            species_num=cfg.data.species_number,
            H=cfg.model.H,
            W=cfg.model.W,
            num_latent_tokens=cfg.model.num_latent_tokens,
            backbone_type=cfg.model.backbone,
            patch_size=cfg.model.patch_size,
            embed_dim=cfg.model.embed_dim,
            num_heads=cfg.model.num_heads,
            head_dim=cfg.model.head_dim,
            depth=cfg.model.depth,
            batch_size=cfg.evaluation.batch_size,
            **swin_params,
        )
    elif mode == "rollout":
        if cfg.model.backbone == "swin":
            selected_swin_config = cfg.model_swin_backbone[cfg.model.swin_backbone_size]
            swin_params = {
                "swin_encoder_depths": tuple(selected_swin_config.encoder_depths),
                "swin_encoder_num_heads": tuple(selected_swin_config.encoder_num_heads),
                "swin_decoder_depths": tuple(selected_swin_config.decoder_depths),
                "swin_decoder_num_heads": tuple(selected_swin_config.decoder_num_heads),
                "swin_window_size": tuple(selected_swin_config.window_size),
                "swin_mlp_ratio": selected_swin_config.mlp_ratio,
                "swin_qkv_bias": selected_swin_config.qkv_bias,
                "swin_drop_rate": selected_swin_config.drop_rate,
                "swin_attn_drop_rate": selected_swin_config.attn_drop_rate,
                "swin_drop_path_rate": selected_swin_config.drop_path_rate,
            }
        assert checkpoint_path, "cannot do rollout without having an initial checkpoint"
        model = BFMRollout.load_from_checkpoint(
            map_location=torch.device("cpu"),  # need to go to CPU, FSDP will take care afterwards
            checkpoint_path=checkpoint_path,
            surface_vars=(cfg.model.surface_vars),
            edaphic_vars=(cfg.model.edaphic_vars),
            atmos_vars=(cfg.model.atmos_vars),
            climate_vars=(cfg.model.climate_vars),
            species_vars=(cfg.model.species_vars),
            vegetation_vars=(cfg.model.vegetation_vars),
            land_vars=(cfg.model.land_vars),
            agriculture_vars=(cfg.model.agriculture_vars),
            forest_vars=(cfg.model.forest_vars),
            redlist_vars=(cfg.model.redlist_vars),
            misc_vars=(cfg.model.misc_vars),
            atmos_levels=cfg.data.atmos_levels,
            species_num=cfg.data.species_number,
            H=cfg.model.H,
            W=cfg.model.W,
            num_latent_tokens=cfg.model.num_latent_tokens,
            backbone_type=cfg.model.backbone,
            patch_size=cfg.model.patch_size,
            embed_dim=cfg.model.embed_dim,
            num_heads=cfg.model.num_heads,
            head_dim=cfg.model.head_dim,
            depth=cfg.model.depth,
            learning_rate=cfg.finetune.lr,
            weight_decay=cfg.finetune.wd,
            batch_size=cfg.finetune.batch_size,
            td_learning=cfg.finetune.td_learning,
            # ground_truth_dataset=test_dataset,
            strict=False,  # False if loading from a pre-trained with PEFT checkpoint
            peft_r=cfg.finetune.rank,
            lora_alpha=cfg.finetune.lora_alpha,
            d_initial=cfg.finetune.d_initial,
            peft_dropout=cfg.finetune.peft_dropout,
            peft_steps=cfg.finetune.rollout_steps,
            peft_mode=cfg.finetune.peft_mode,
            use_lora=cfg.finetune.use_lora,
            use_vera=cfg.finetune.use_vera,
            rollout_steps=cfg.finetune.rollout_steps,
            finetune_mode=cfg.finetune.mode,
            # lora_steps=cfg.finetune.rollout_steps, # 1 month
            # lora_mode=cfg.finetune.lora_mode, # every step + layers #single
            **swin_params,
        )
        apply_activation_checkpointing(model, checkpoint_wrapper_fn=checkpoint_wrapper, check_fn=activation_ckpt_policy)
    else:
        raise ValueError(mode)

    model_summary = ModelSummary(model, max_depth=2)
    print(model_summary)
    print("Done \n Setting up the BFM model")
    return model


def setup_checkpoint_callback(cfg, output_dir: str) -> ModelCheckpoint:
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"{output_dir}/checkpoints",
        save_top_k=1,
        monitor="val_loss",  # `log('val_loss', value)` in the `LightningModule`
        mode="min",
        every_n_train_steps=cfg.training.checkpoint_every,
        filename="{epoch:02d}-{val_loss:.5f}",
        save_last=True,
    )
    print(f"Will be saving checkpoints at: {output_dir}/checkpoints")
    return checkpoint_callback


def setup_fsdp(cfg, model):
    latent_list = list(model.encoder._latent_parameter_list)
    if cfg.training.strategy == "fsdp":
        distr_strategy = FSDPStrategy(
            sharding_strategy="FULL_SHARD",
            auto_wrap_policy=partial(size_based_auto_wrap_policy, min_num_params=int(1e6)),
            ignored_states=latent_list,
            activation_checkpointing_policy=activation_ckpt_policy,
        )

    elif cfg.training.strategy == "ddp":
        distr_strategy = DDPStrategy()

    else:
        distr_strategy = "auto"

    print(f"Using {cfg.training.strategy} strategy: {distr_strategy}")
    return distr_strategy


def get_trainer(
    cfg,
    mlflow_logger: List[MLFlowLogger],
    distr_strategy: str | FSDPStrategy | DDPStrategy = "auto",
    callbacks: list = [],
) -> L.Trainer:
    trainer = L.Trainer(
        max_epochs=cfg.training.epochs,
        accelerator=cfg.training.accelerator,
        devices=cfg.training.devices,
        precision=cfg.training.precision,
        strategy=distr_strategy,
        num_nodes=cfg.training.num_nodes,
        log_every_n_steps=cfg.training.log_steps,
        logger=mlflow_logger,  # Only the rank 0 process will have a logger
        # limit_train_batches=3,  # Process 10 batches per epoch.
        # limit_val_batches=2,
        # limit_test_batches=10,
        # limit_predict_batches=12,
        val_check_interval=cfg.training.eval_every,  # Run validation every n training batches.
        check_val_every_n_epoch=None,
        # limit_train_batches=1, # For debugging to see what happens at the end of epoch
        # check_val_every_n_epoch=None,  # Do eval every n epochs
        # val_check_interval=3, # Does not work in Distributed settings | Do eval every 10 training steps => 10 steps x 8 batch_size = Every 80 Batches
        callbacks=callbacks,
        # plugins=[MyClusterEnvironment()],
    )
    return trainer


def find_checkpoint_to_resume_from(
    cfg,
) -> str | None:
    checkpoint_path = None
    if hasattr(cfg.training, "checkpoint_path") and cfg.training.checkpoint_path:
        # Check if the path is a directory
        if os.path.isdir(cfg.training.checkpoint_path):
            # Look for checkpoint files in the directory
            possible_checkpoints = []
            for root, _, files in os.walk(cfg.training.checkpoint_path):
                for file in files:
                    if file.endswith(".ckpt") and os.path.isfile(os.path.join(root, file)):
                        possible_checkpoints.append(os.path.join(root, file))
            if possible_checkpoints:
                # Sort by modification time (newest first)
                checkpoint_path = sorted(possible_checkpoints, key=os.path.getmtime, reverse=True)[0]
                print(f"Found most recent checkpoint from {len(possible_checkpoints)} possible: {checkpoint_path}")

            else:
                print(f"No checkpoint files found in {cfg.training.checkpoint_path}.")
        else:
            # Path is a specific file
            checkpoint_path = cfg.training.checkpoint_path
            print(f"Checkpoint path: {checkpoint_path}")
    else:
        print("No checkpoint path declared in config")

    return checkpoint_path


def post_training_get_last_checkpoint(output_dir: str, checkpoint_callback: ModelCheckpoint, trainer: L.Trainer):
    if dist.is_initialized():
        dist.barrier()

    selected_ckpt = checkpoint_callback.best_model_path or f"{output_dir}/checkpoints/last.ckpt"

    if not os.path.exists(selected_ckpt):
        raise FileNotFoundError(f"Checkpoint not found at {selected_ckpt}")

    if trainer.is_global_zero:
        print(f"[Rank 0] Using checkpoint: {selected_ckpt}")

    # broadcast checkpoint path
    if dist.is_initialized():
        ckpt_list = [selected_ckpt]
        dist.broadcast_object_list(ckpt_list, src=0)
        selected_ckpt = ckpt_list[0]

    return selected_ckpt
