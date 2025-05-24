"""
Copyright (C) 2025 TNO, The Netherlands. All rights reserved.
"""

import math
import os
import pickle
from datetime import datetime
from functools import partial
from typing import Literal

import hydra
import lightning as L
import torch
import torch.distributed as dist
from hydra.core.hydra_config import HydraConfig
from lightning.pytorch import LightningModule, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import MLFlowLogger
from lightning.pytorch.plugins.environments import ClusterEnvironment
from lightning.pytorch.strategies import DDPStrategy, FSDPStrategy
from lightning.pytorch.utilities.model_summary import ModelSummary
from omegaconf import OmegaConf
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    apply_activation_checkpointing,
    checkpoint_wrapper,
)
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
from torch.utils.data import DataLoader

from bfm_model.bfm.dataloader_monthly import LargeClimateDataset, custom_collate
from bfm_model.bfm.decoder import BFMDecoder
from bfm_model.bfm.encoder import BFMEncoder
from bfm_model.mvit.mvit_model import MViT
from bfm_model.swin_transformer.core.swim_core_v2 import Swin3DTransformer


def activation_ckpt_policy(module):
    return isinstance(module, (Swin3DTransformer, MViT))


class MyClusterEnvironment(ClusterEnvironment):
    @property
    def creates_processes_externally(self) -> bool:
        """Return True if the cluster is managed (you don't launch processes yourself)"""
        return True

    def world_size(self) -> int:
        return int(os.environ["WORLD_SIZE"])

    def global_rank(self) -> int:
        return int(os.environ["RANK"])

    def local_rank(self) -> int:
        return int(os.environ["LOCAL_RANK"])

    def node_rank(self) -> int:
        return int(os.environ["NODE_RANK"])

    def main_address(self) -> str:
        return os.environ["MASTER_ADDRESS"]

    def main_port(self) -> int:
        return int(os.environ["MASTER_PORT"])

class BFM_lighting(LightningModule):
    """
    Biodiversity Foundation Model.

    This model combines encoder, backbone and decoder components to process climate and biodiversity-related variables.

    Args:
        surface_vars (tuple[str, ...]): Names of surface-level variables
        single_vars (tuple[str, ...]): Names of single-level variables
        atmos_vars (tuple[str, ...]): Names of atmospheric variables
        species_vars (tuple[str, ...]): Names of species-related variables
        species_distr_vars (tuple[str, ...]): Names of species distributions-related variables
        land_vars (tuple[str, ...]): Names of land-related variables
        agriculture_vars (tuple[str, ...]): Names of agriculture-related variables
        forest_vars (tuple[str, ...]): Names of forest-related variables
        atmos_levels (list[int]): Pressure levels for atmospheric variables
        species_num (int): Number of species distribution to account for
        H (int, optional): Height of output grid. Defaults to 32.
        W (int, optional): Width of output grid. Defaults to 64.
        num_latent_tokens (int, optional): Number of latent tokens. Defaults to 8.
        backbone_type (Literal["swin", "mvit"], optional): Type of backbone architecture. Defaults to "mvit".
        patch_size (int, optional): Size of spatial patches. Defaults to 4.
        embed_dim (int, optional): Embedding dimension. Defaults to 1024.
        num_heads (int, optional): Number of attention heads. Defaults to 16.
        head_dim (int, optional): Dimension of each attention head. Defaults to 64.
        depth (int, optional): Number of transformer layers. Defaults to 2.
        **kwargs: Additional arguments passed to encoder and decoder

    Attributes:
        encoder (BFMEncoder): Encoder component
        backbone (nn.Module): Backbone network (Swin or MViT)
        decoder (BFMDecoder): Decoder component
        backbone_type (str): Type of backbone being used
    """

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
        land_mask_path: str = "",
        use_mask: bool = True,
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
        self.use_mask = use_mask

        # load land-sea mask
        try:
            with open(land_mask_path, 'rb') as f:
                land_sea_mask_numpy = pickle.load(f)
            # the loaded mask is a numpy array with 1 for land, 0 for sea and has shape (H, W) matching self.H, self.W after training_config values are used
            land_sea_mask_tensor = torch.from_numpy(land_sea_mask_numpy.astype(float)).float()
            if land_sea_mask_tensor.shape != (self.H, self.W):
                print(f"Land mask shape {land_sea_mask_tensor.shape} does not match H,W parameters ({self.H},{self.W})")
            self.register_buffer('land_sea_mask', land_sea_mask_tensor, persistent=False)
        except FileNotFoundError:
            print(f"Land-sea mask file not found at {land_mask_path}. Loss will be calculated over all pixels.")
            self.register_buffer('land_sea_mask', None, persistent=False)

        self.variable_weights = {
            "surface_variables": {
                "t2m": 0.1,
                "msl": 0.1,
                "slt": 0.1,
                "z": 0.1,
                "u10": 0.1,
                "v10": 0.1,
                "lsm": 0.1,
                },
            "edaphic_variables": {"swvl1": 0.1, "swvl2": 0.1, "stl1": 0.1, "stl2":0.1,},
            "atmospheric_variables": {"z": 0.1, "t": 0.1, "u": 0.1, "v": 0.1, "q": 0.1},
            "climate_variables": {"smlt": 0.1, "tp": 0.1, "csfr": 0.1, "avg_sdswrf": 0.1,
                                "avg_snswrf": 0.1, "avg_snlwrf": 0.1, "avg_tprate": 0.1,
                                "avg_sdswrfcs": 0.1, "sd": 0.1, "t2m": 0.1, "d2m": 0.1},
            "land_variables": {"Land": 0.1},
            "agriculture_variables": {"Agriculture": 0.1, "Arable": 0.1, "Cropland": 0.1},
            "forest_variables": {"Forest": 0.1},
            "redlist_variables": {"RLI": 0.1},
            "misc_variables": {"avg_slhtf": 0.1, "avg_pevr": 0.1},
            "species_variables": 5.0
        }

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
                encoder_depths=(2, 2),
                encoder_num_heads=(8, 16),
                decoder_depths=(2, 2),
                decoder_num_heads=(32, 16),
                window_size=(1, 1, 1), # ()
                mlp_ratio=4.0,
                qkv_bias=True,
                drop_rate=0.0,
                attn_drop_rate=0.0,
                drop_path_rate=0.1,
                use_lora=False,
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

    def forward(self, batch, lead_time: int = 2, batch_size: int = 1):
        """
        Forward pass of the model.

        Args:
            batch: Batch object containing input variables and metadata
            lead_time (int): Time difference in months between input and target

        Returns:
            dict: Dictionary containing decoded outputs for each variable category

        """
        # print(f"BFM batch size: {batch_size}")
        encoded = self.encoder(batch, lead_time, batch_size)
        # print("Encoded shape", encoded.shape)

        # calculate number of patches in 2D
        num_patches_h = self.H // self.encoder.patch_size
        num_patches_w = self.W // self.encoder.patch_size
        total_patches = num_patches_h * num_patches_w  # noqa

        # calculate depth to match the sequence length
        depth = encoded.shape[1] // (num_patches_h * num_patches_w)
        # print(f"BFM depth: {depth} | patch_size {self.encoder.patch_shape} | encoder shape {encoded.shape}")
        patch_shape = (
            depth,  # depth dimension matches sequence length / (H*W)
            num_patches_h,  # height in patches
            num_patches_w,  # width in patches
        )

        if self.backbone_type == "mvit":
            encoded = encoded.view(encoded.size(0), -1, self.encoder.embed_dim)
            print(f"Reshaped encoded for MViT: {encoded.shape}")

        backbone_output = self.backbone(encoded, lead_time=lead_time, rollout_step=0, patch_shape=patch_shape)
        # print("Backbone output", backbone_output.shape)
        # decode
        output = self.decoder(backbone_output, batch, lead_time)
        # print("Decoded output:", output)
        return output

    def validation_step(self, batch, batch_idx):
        lead_time = 2  # fixed lead time (months) for pre-training
        x, y = batch
        output = self(x, lead_time, batch_size=self.batch_size)
        print("Validation time!")
        loss = self.compute_loss(output, y)
        self.log(
            "val_loss", loss, batch_size=self.batch_size, sync_dist=True
        )  # on_step=False, on_epoch=True, prog_bar=True, sync_dist=True
        return loss

    def training_step(self, batch, batch_idx):
        lead_time = 2  # fixed lead time (months) for pre-training
        x, y = batch
        output = self(x, lead_time, batch_size=self.batch_size)
        loss = self.compute_loss(output, y)
        self.log("train_loss", loss, batch_size=self.batch_size, sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx):
        lead_time = 2   # fixed lead time (months) for pre-training
        x, y = batch
        output = self(x, lead_time, batch_size=self.batch_size)
        print("Test time")
        loss = self.compute_loss(output, y)
        self.log("test_loss", loss, batch_size=self.batch_size, sync_dist=True)
        return loss

    def predict_step(self, batch, batch_idx):
        x, y = batch
        lead_time = 2  # fixed lead time (months) for pre-training
        output = self(x, lead_time, batch_size=self.batch_size)
        return output

    def on_after_backward(self):
        """
        Checker for not learnable parameters -> Should output nothing!
        """
        for name, param in self.named_parameters():
            if param.grad is None:
                print(name)

    def compute_loss(self, output, batch):
        """
        Some helpers:
        1) https://link.springer.com/article/10.1007/s13253-025-00676-8 
        2) https://www.sciencedirect.com/science/article/pii/S1574954124001651
        3) https://www.sciencedirect.com/science/article/pii/S1470160X22009608
        """

        groups = [
            "surface_variables",
            "edaphic_variables",
            "atmospheric_variables",
            "climate_variables",
            "species_variables",
            "vegetation_variables",
            "land_variables",
            "agriculture_variables",
            "forest_variables",
            "redlist_variables",
            "misc_variables",
        ]

        if self.land_sea_mask is not None and self.use_mask:
            print("Computing loss with land-sea mask.")
            total_masked_loss = 0.0
            count = 0
            current_land_mask = None
            try:
                sample_output_group_key = next(iter(output))
                sample_output_var_key = next(iter(output[sample_output_group_key]))
                target_device = output[sample_output_group_key][sample_output_var_key].device
                current_land_mask = self.land_sea_mask.to(target_device)
            except Exception as e:
                return self._compute_unmasked_loss(output, batch, groups)

            for group_name in groups:
            # If group doesn't exist in output or batch, skip
                if group_name not in output or group_name not in batch._asdict():
                    continue

                pred_dict = output[group_name]
                true_dict = getattr(batch, group_name)
                group_masked_loss = 0.0
                var_count = 0

                for var_name, pred_tensor in pred_dict.items():
                    if var_name not in true_dict:
                        print(f"compute_loss: variable {var_name} in group {group_name} not in true_dict, skipping for masked loss")
                        continue
                    gt_tensor = true_dict[var_name]
                    abs_error_map = torch.tensor(0.0, device=pred_tensor.device)

                    if self.td_learning:
                        time0 = gt_tensor[:, 0]
                        time1 = gt_tensor[:, 1]
                        true_diff = time1 - time0
                        pred_diff = pred_tensor - time0
                        abs_error_map = torch.abs(pred_diff - true_diff)
                    else:
                        time1 = gt_tensor[:, 1]
                        abs_error_map = torch.abs(pred_tensor - time1)
                    
                    # Calculate masked loss directly
                    masked_loss_var = torch.mean(abs_error_map) # Default if mask cannot be applied
                    if current_land_mask is not None and abs_error_map.ndim >=2 and \
                       abs_error_map.shape[-2:] == current_land_mask.shape:
                        
                        broadcastable_mask = current_land_mask
                        if abs_error_map.ndim == 3: 
                            broadcastable_mask = current_land_mask.unsqueeze(0) 
                        elif abs_error_map.ndim == 4: 
                            broadcastable_mask = current_land_mask.unsqueeze(0).unsqueeze(0) 
                        # No explicit else for ndim < 3 as masked_loss_var is already defaulted
                        
                        if abs_error_map.ndim == 3 or abs_error_map.ndim == 4:
                            masked_error_sum = torch.sum(abs_error_map * broadcastable_mask)
                            num_elements_for_mean = torch.sum(broadcastable_mask.expand_as(abs_error_map))

                            if num_elements_for_mean > 0:
                                masked_loss_var = masked_error_sum / num_elements_for_mean
                            else:
                                masked_loss_var = torch.tensor(0.0, device=pred_tensor.device) 
                                print(f"compute_loss: variable {var_name} had 0 land pixels for masked loss calculation.")
                    elif current_land_mask is not None:  # mask exists but not applicable due to shape mismatch
                        print(f"compute_loss: land_sea_mask not applicable for {var_name} (map shape {abs_error_map.shape[-2:]} vs mask shape {current_land_mask.shape}), using unmasked mean for this var.")

                    self.log(f"{var_name} land_masked raw loss", masked_loss_var, batch_size=gt_tensor.size(0))
                    
                    group_weights = self.variable_weights.get(group_name, {})
                    w = group_weights.get(var_name, 1.0) if isinstance(group_weights, dict) else group_weights
                    group_masked_loss += w * masked_loss_var
                    var_count += 1

                if var_count > 0:
                    group_masked_loss /= var_count  
                    total_masked_loss += group_masked_loss
                    count += 1

            final_total_masked_loss = total_masked_loss / count if count > 0 else torch.tensor(0.0, device=output[groups[0]][next(iter(output[groups[0]]))].device) 
            return final_total_masked_loss
        else:
            return self._compute_unmasked_loss(output, batch, groups)

    def _compute_unmasked_loss(self, output, batch, groups):
        total_loss = 0.0
        count = 0
        for group_name in groups:
            if group_name not in output or group_name not in batch._asdict():
                continue

            pred_dict = output[group_name]
            true_dict = getattr(batch, group_name)
            group_loss = 0.0
            var_count = 0

            for var_name, pred_tensor in pred_dict.items():
                # If var_name not in the ground truth dict, skip
                if var_name not in true_dict:
                    print(f"{var_name} not in true_dict")
                    continue
                gt_tensor = true_dict[var_name]

                if self.td_learning:
                    time0 = gt_tensor[:, 0]
                    time1 = gt_tensor[:, 1]

                    true_diff = time1 - time0
                    pred_diff = pred_tensor - time0

                    loss_var = torch.mean(torch.abs(pred_diff - true_diff))
                else:
                    time1 = gt_tensor[:, 0]
                    loss_var = torch.mean(torch.abs(pred_tensor - time1))

                # Determine the weight for this variable.
                group_weights = self.variable_weights.get(group_name, {})
                w = group_weights.get(var_name, 1.0) if isinstance(group_weights, dict) else group_weights
                # Log each variable's raw loss
                self.log(f"{var_name} raw loss (unmasked)", loss_var, batch_size=gt_tensor.size(0))
                group_loss += w * loss_var
                var_count += 1

            if var_count > 0:
                group_loss /= var_count  # average within group
                total_loss += group_loss
                count += 1
        
        final_total_loss = total_loss / count if count > 0 else torch.tensor(0.0, device=output[groups[0]][next(iter(output[groups[0]]))].device)
        return final_total_loss

    def lr_lambda(self, current_step):
        if current_step < self.warmup_steps:
            # Linear warmup from 0 to 1.
            return float(current_step) / float(max(1, self.warmup_steps))
        else:
            # After warmup, cosine decay from 1.0 to 0.1 over the remaining steps.
            progress = float(current_step - self.warmup_steps) / float(max(1, self.total_steps - self.warmup_steps))
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
            # Scale so that at the end (progress=1), the multiplier is 0.1.
            return 0.9 * cosine_decay + 0.1

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=12000, eta_min=self.learning_rate / 10)
        # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=self.lr_lambda)

        return [optimizer], [scheduler]


class OptimizerParamsLogger(L.Callback):
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        # Ensure the logger is an MLFlowLogger instance.
        if not isinstance(trainer.logger, MLFlowLogger):
            return

        mlflow_logger = trainer.logger
        run_id = mlflow_logger.run_id
        current_step = trainer.global_step

        # Loop over each optimizer.
        for opt_idx, optimizer in enumerate(trainer.optimizers):
            for pg_idx, param_group in enumerate(optimizer.param_groups):
                lr = param_group.get("lr")
                wd = param_group.get("weight_decay")
                # Log the current learning rate and weight decay as metrics.
                mlflow_logger.experiment.log_metric(run_id, f"optimizer_{opt_idx}_group_{pg_idx}_lr", lr, step=current_step)
                mlflow_logger.experiment.log_metric(
                    run_id, f"optimizer_{opt_idx}_group_{pg_idx}_weight_decay", wd, step=current_step
                )


class FSDPEvalCallback(L.Callback):
    def __init__(self, eval_interval=10):
        self.eval_interval = eval_interval

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        # Only on global rank zero
        if not trainer.is_global_zero:
            return

        if trainer.global_step % self.eval_interval == 0:
            # Use best checkpoint if available, else fall back to last.
            best_ckpt = trainer.checkpoint_callback.best_model_path
            ckpt_to_use = best_ckpt if best_ckpt else "last"
            print(f"Step {trainer.global_step}: Validating using checkpoint: {ckpt_to_use}")
            trainer.validate(ckpt_path=ckpt_to_use)


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
    dataset = LargeClimateDataset(
        data_dir=cfg.data.data_path,
        scaling_settings=cfg.data.scaling,
        num_species=cfg.data.species_number,
        atmos_levels=cfg.data.atmos_levels,
        model_patch_size=cfg.model.patch_size
    )
    test_dataset = LargeClimateDataset(
        data_dir=cfg.data.test_data_path,
        scaling_settings=cfg.data.scaling,
        num_species=cfg.data.species_number,
        atmos_levels=cfg.data.atmos_levels,
        model_patch_size=cfg.model.patch_size
    )

    val_dataloader = DataLoader(
        test_dataset,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.training.workers,
        collate_fn=custom_collate,
        drop_last=True,
        shuffle=False,
    )

    train_dataloader = DataLoader(
        dataset,
        shuffle=True,  # keep shuffle=True here
        batch_size=cfg.training.batch_size,
        num_workers=cfg.training.workers,
        collate_fn=custom_collate,
        drop_last=True,
        pin_memory=True,
    )
    print(f"Setting up Daloaders with length train: {len(train_dataloader)} and test: {len(val_dataloader)}")
    # Setup logger with rank-specific paths to avoid conflicts
    current_time = datetime.now()
    rank = int(os.environ.get("RANK", "0"))
    print(f"Will be using rank {rank} for logging")
    # Single logger approach with rank-specific paths
    mlf_logger = None
    # if "RANK" not in os.environ or os.environ["RANK"] == "0":
    if rank == 0 or rank == "0":
        # Use rank in experiment name to avoid conflicts
        mlf_logger = MLFlowLogger(
            experiment_name=f"BFM_logs_r{rank}", run_name=f"BFM_{current_time}", save_dir=f"{output_dir}/logs/rank{rank}"
        )

    print("Done \n Setting up the BFM")
    BFM = BFM_lighting(
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
    )

    apply_activation_checkpointing(BFM, checkpoint_wrapper_fn=checkpoint_wrapper, check_fn=activation_ckpt_policy)

    model_summary = ModelSummary(BFM, max_depth=2)
    print(model_summary)

    checkpoint_callback = ModelCheckpoint(
        dirpath=f"{output_dir}/checkpoints",
        save_top_k=3,
        monitor="val_loss",  # `log('val_loss', value)` in the `LightningModule`
        mode="min",
        every_n_train_steps=cfg.training.checkpoint_every,
        filename="{epoch:02d}-{val_loss:.5f}",
        save_last=True,
    )

    print(f"Will be saving checkpoints at: {output_dir}/checkpoints")

    if cfg.training.strategy == "fsdp":
        distr_strategy = FSDPStrategy(
            sharding_strategy="FULL_SHARD",
            auto_wrap_policy=partial(size_based_auto_wrap_policy, min_num_params=1e6),
            # activation_checkpointing_policy=activation_ckpt_policy,
        )

    elif cfg.training.strategy == "ddp":
        distr_strategy = DDPStrategy()

    print(f"Using {cfg.training.strategy} strategy: {distr_strategy}")

    trainer = L.Trainer(
        max_epochs=cfg.training.epochs,
        accelerator=cfg.training.accelerator,
        devices=cfg.training.devices,
        precision=cfg.training.precision,
        # strategy=distr_strategy,
        num_nodes=cfg.training.num_nodes,
        log_every_n_steps=cfg.training.log_steps,
        logger=mlf_logger,  # Only the rank 0 process will have a logger
        # limit_train_batches=2,  # Process 10 batches per epoch.
        # limit_val_batches=2,
        # limit_test_batches=10,
        val_check_interval=cfg.training.eval_every,  # Run validation every n training batches.
        check_val_every_n_epoch=None,
        # limit_train_batches=1, # For debugging to see what happens at the end of epoch
        # check_val_every_n_epoch=None,  # Do eval every n epochs
        # val_check_interval=3, # Does not work in Distributed settings | Do eval every 10 training steps => 10 steps x 8 batch_size = Every 80 Batches
        callbacks=[checkpoint_callback],
        # plugins=[MyClusterEnvironment()],
    )
    # Experimental
    # mlflow.set_tracking_uri(output_dir)
    # Auto log all MLflow entities
    # mlflow.pytorch.autolog()

    # with mlflow.start_run() as run:
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
                checkpoint_to_load = sorted(possible_checkpoints, key=os.path.getmtime, reverse=True)[0]
                print(f"Found checkpoint: {checkpoint_to_load}")
                trainer.fit(BFM, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader, ckpt_path=checkpoint_to_load)
            else:
                print(f"No checkpoint files found in {cfg.training.checkpoint_path}. Starting training from scratch.")
                trainer.fit(BFM, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
        else:
            # Path is a specific file
            print(f"Loading and resuming training from {cfg.training.checkpoint_path}")
            trainer.fit(
                BFM, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader, ckpt_path=cfg.training.checkpoint_path
            )
    else:
        print("Start training from scratch")
        trainer.fit(BFM, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

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

    trainer.test(model=BFM, ckpt_path=selected_ckpt, dataloaders=val_dataloader)

    print("Finished testing successfully")
    trainer.print(torch.cuda.memory_summary())

    # Optional: clean up the distributed process group
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
