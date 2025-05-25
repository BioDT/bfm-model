"""
Copyright (C) 2025 TNO, The Netherlands. All rights reserved.
"""
from datetime import datetime, timedelta
from typing import Literal, Tuple
from pathlib import Path
from collections import defaultdict

import hydra
import lightning as L
import torch
from hydra.core.hydra_config import HydraConfig
from lightning.pytorch import LightningModule, seed_everything
from lightning.pytorch.loggers import MLFlowLogger
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from bfm_model.bfm.dataloader_monthly import LargeClimateDataset, custom_collate, _convert
from bfm_model.bfm.train_lighting import BFM_lighting
from bfm_model.bfm.decoder import BFMDecoder
from bfm_model.bfm.encoder import BFMEncoder
from bfm_model.mvit.mvit_model import MViT
from bfm_model.swin_transformer.core.swim_core_v2 import Swin3DTransformer

class BFM_lighting(LightningModule):
    def __init__(
        self,
        surface_vars: tuple[str, ...],
        edaphic_vars: tuple[str, ...],
        atmos_vars: tuple[str, ...],
        climate_vars: tuple[str, ...],
        species_vars: tuple[str, ...],
        vegetation_vars: tuple[str, ...],
        land_vars: tuple[str, ...],
        agriculture_vars: tuple[str, ...],
        forest_vars: tuple[str, ...],
        redlist_vars: tuple[str, ...],
        misc_vars: tuple[str, ...],
        atmos_levels: list[int],
        species_num: int,
        H: int = 32,
        W: int = 64,
        num_latent_tokens: int = 8,
        backbone_type: Literal["swin", "mvit"] = "mvit",
        patch_size: int = 4,
        embed_dim: int = 1024,
        num_heads: int = 16,
        head_dim: int = 2,
        depth: int = 2,
        learning_rate: float = 5e-4,
        weight_decay: float = 5e-6,
        batch_size: int = 1,
        warmup_steps: int = 1000,
        total_steps: int = 20000,
        td_learning: bool = True,
        lead_time: int = 2,
        swin_encoder_depths: Tuple[int, ...] = (2,2,2),
        swin_encoder_num_heads: Tuple[int, ...] = (8,16,32),
        swin_decoder_depths: Tuple[int, ...] = (2,2,2),
        swin_decoder_num_heads: Tuple[int, ...] = (32,16,8),
        swin_window_size: Tuple[int, ...] = (1,4,5),
        swin_mlp_ratio: float = 4.0,
        swin_qkv_bias: bool = True,
        swin_drop_rate: float = 0.0,
        swin_attn_drop_rate: float = 0.0,
        swin_drop_path_rate: float = 0.1,
        swin_use_lora: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.H = H
        self.W = W

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.td_learning = td_learning
        self.lead_time = lead_time

        self.swin_encoder_depths = swin_encoder_depths
        self.swin_encoder_num_heads = swin_encoder_num_heads
        self.swin_decoder_depths = swin_decoder_depths
        self.swin_decoder_num_heads = swin_decoder_num_heads
        self.swin_window_size = swin_window_size
        self.swin_mlp_ratio = swin_mlp_ratio
        self.swin_qkv_bias = swin_qkv_bias
        self.swin_drop_rate = swin_drop_rate
        self.swin_attn_drop_rate = swin_attn_drop_rate
        self.swin_drop_path_rate = swin_drop_path_rate
        self.swin_use_lora = swin_use_lora

        self.encoder = BFMEncoder(
            surface_vars=surface_vars,
            edaphic_vars=edaphic_vars,
            atmos_vars=atmos_vars,
            climate_vars=climate_vars,
            species_vars=species_vars,
            vegetation_vars=vegetation_vars,
            land_vars=land_vars,
            agriculture_vars=agriculture_vars,
            forest_vars=forest_vars,
            redlist_vars=redlist_vars,
            misc_vars=misc_vars,
            atmos_levels=atmos_levels,
            species_num=species_num,
            H=H,
            W=W,
            patch_size=patch_size,
            embed_dim=embed_dim,
            num_heads=num_heads,
            head_dim=head_dim,
            depth=depth,
            **kwargs,
        )

        patch_shape = (num_latent_tokens, H // self.encoder.patch_size, W // self.encoder.patch_size)

        if backbone_type == "swin":
            self.backbone = Swin3DTransformer(
                embed_dim=embed_dim,
                encoder_depths=self.swin_encoder_depths,
                encoder_num_heads=self.swin_encoder_num_heads,
                decoder_depths=self.swin_decoder_depths,
                decoder_num_heads=self.swin_decoder_num_heads,
                window_size=self.swin_window_size,
                mlp_ratio=self.swin_mlp_ratio,
                qkv_bias=self.swin_qkv_bias,
                drop_rate=self.swin_drop_rate,
                attn_drop_rate=self.swin_attn_drop_rate,
                drop_path_rate=self.swin_drop_path_rate,
                use_lora=self.swin_use_lora,
            )
        elif backbone_type == "mvit":
            self.backbone = MViT(
                patch_shape=patch_shape,
                embed_dim=embed_dim,
                depth=4,
                num_heads=1,
                mlp_ratio=4.0,
                qkv_bias=True,
                path_drop_rate=0.1,
                attn_mode="conv",
                pool_first=False,
                rel_pos=False,
                zero_init_rel=True,
                res_pool=True,
                dim_mul_attn=False,
                dim_scales=[(i, 1.0) for i in range(4)],  # No dimension change
                head_scales=[(1, 2.0), (2, 2.0)],  # Keep head scaling for attention
                pool_kernel=[1, 1, 1],
                kv_stride=[1, 1, 1],
                q_stride=[(0, [1, 1, 1]), (1, [1, 1, 1]), (2, [1, 1, 1])],
            )
        else:
            raise ValueError(f"Unknown backbone type: {backbone_type}")

        self.backbone_type = backbone_type
        self.decoder = BFMDecoder(
            surface_vars=surface_vars,
            edaphic_vars=edaphic_vars,
            atmos_vars=atmos_vars,
            climate_vars=climate_vars,
            species_vars=species_vars,
            vegetation_vars=vegetation_vars,
            land_vars=land_vars,
            agriculture_vars=agriculture_vars,
            forest_vars=forest_vars,
            redlist_vars=redlist_vars,
            misc_vars=misc_vars,
            atmos_levels=atmos_levels,
            species_num=species_num,
            H=H,
            W=W,
            patch_size=patch_size,
            embed_dim=embed_dim,
            num_heads=num_heads,
            head_dim=head_dim,
            depth=depth,
            **kwargs,
        )

    def forward(self, batch, lead_time=2, batch_size: int = 1):
        encoded = self.encoder(batch, lead_time, batch_size)
        num_patches_h = self.H // self.encoder.patch_size
        num_patches_w = self.W // self.encoder.patch_size
        total_patches = num_patches_h * num_patches_w  # noqa
        depth = encoded.shape[1] // (num_patches_h * num_patches_w)
        patch_shape = (
            depth,  # depth dimension matches sequence length / (H*W)
            num_patches_h,  # height in patches
            num_patches_w,  # width in patches
        )

        if self.backbone_type == "mvit":
            encoded = encoded.view(encoded.size(0), -1, self.encoder.embed_dim)
            print(f"Reshaped encoded for MViT: {encoded.shape}")
        backbone_output = self.backbone(encoded, lead_time=lead_time, rollout_step=0, patch_shape=patch_shape)
        output = self.decoder(backbone_output, batch, lead_time)
        return output

    def predict_step(self, batch, batch_idx):
        records = []
        x, y = batch
        output = self(x, self.lead_time, batch_size=self.batch_size)
        # pred_cpu = detach_output_dict(output) # helper does detach.clone().cpu()
        # gt_cpu   = detach_batch(y) # The first timestep is the ground truth
        records.append({
            "idx": batch_idx,
            "pred": output,
            "gt": y,
        })
        return records


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

    # Load the Test Dataset
    print("Setting up Dataloader ...")
    test_dataset = LargeClimateDataset(
        data_dir=cfg.evaluation.test_data, scaling_settings=cfg.data.scaling, 
        num_species=cfg.data.species_number, atmos_levels=cfg.data.atmos_levels)  # Adapt
    print("Reading test data from :", cfg.evaluation.test_data)
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=cfg.evaluation.batch_size,
        num_workers=cfg.training.workers,
        collate_fn=custom_collate,
        drop_last=True,
        shuffle=False,
    )

    output_dir = HydraConfig.get().runtime.output_dir

    # Setup logger
    current_time = datetime.now()
    # log the metrics in the hydra folder (easier to find)
    mlf_logger_in_hydra_folder = MLFlowLogger(
        experiment_name="BFM_logs", run_name=f"BFM_{current_time}", save_dir=f"{output_dir}/logs"
    )
    # also log in the .mlruns folder so that you can run mlflow server and see every run together
    # tracking_uri = f"http://0.0.0.0:{cfg.mlflow.port}"
    mlf_logger_in_current_folder = MLFlowLogger(experiment_name="BFM_logs", run_name=f"BFM_{current_time}")

    trainer = L.Trainer(
        accelerator=cfg.training.accelerator,
        devices=cfg.training.devices,
        precision=cfg.training.precision,
        log_every_n_steps=cfg.training.log_steps,
        # limit_test_batches=1,
        limit_predict_batches=12, #TODO Change this to select how many consecutive months you want to predict
        logger=[mlf_logger_in_hydra_folder, mlf_logger_in_current_folder],
        enable_checkpointing=False,
        enable_progress_bar=True,
    )

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
            "swin_use_lora": selected_swin_config.use_lora,
        }

    bfm_model = BFM_lighting(
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

    for batch in predictions:
        for rec in batch:
            idx = rec["idx"]

            # scale tensors to original space before CPU conversion
            pred_scaled = test_dataset.scale_batch(rec["pred"], direction="original")
            gt_scaled = test_dataset.scale_batch(rec["gt"], direction="original")

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
