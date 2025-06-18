"""
Copyright (C) 2025 TNO, The Netherlands. All rights reserved.

BFM (Biodiversity Foundation Model) Main Module.

This module contains the main BFM architecture, combining encoder, backbone and decoder components
to process climate and biodiversity-related variables.

The model uses either a Swin or MViT backbone architecture to process encoded representations
before decoding back to the original variable space.

Key Components:
    - Variable preprocessing and cropping
    - Encoder for initial representation learning
    - Backbone (Swin or MViT) for temporal-spatial processing
    - Decoder for reconstructing variables
    - Multi-category variable handling (surface, atmospheric, species, etc.)
"""

import math
import os
import pickle
from datetime import datetime
from functools import partial
from typing import Literal, Tuple

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


class BFM(LightningModule):
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
        use_mask: str = "no",
        partially_masked_groups: list[str] = ["species_variables"],
        lead_time: int = 2,
        swin_encoder_depths: Tuple[int, ...] = (2, 2, 2),
        swin_encoder_num_heads: Tuple[int, ...] = (8, 16, 32),
        swin_decoder_depths: Tuple[int, ...] = (2, 2, 2),
        swin_decoder_num_heads: Tuple[int, ...] = (32, 16, 8),
        swin_window_size: Tuple[int, ...] = (1, 4, 5),
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
        self.use_mask = use_mask
        self.partially_masked_groups = partially_masked_groups

        # load land-sea mask
        try:
            with open(land_mask_path, "rb") as f:
                land_sea_mask_numpy = pickle.load(f)
            # the loaded mask is a numpy array with 1 for land, 0 for sea and has shape (H, W) matching self.H, self.W after training_config values are used
            land_sea_mask_tensor = torch.from_numpy(land_sea_mask_numpy.astype(float)).float()
            if land_sea_mask_tensor.shape != (self.H, self.W):
                print(f"Land mask shape {land_sea_mask_tensor.shape} does not match H,W parameters ({self.H},{self.W})")
            self.register_buffer("land_sea_mask", land_sea_mask_tensor, persistent=False)
        except FileNotFoundError:
            print(f"Land-sea mask file not found at {land_mask_path}. Loss will be calculated over all pixels.")
            self.register_buffer("land_sea_mask", None, persistent=False)

        self.variable_weights = {
            "surface_variables": {
                "t2m": 2.5,
                "msl": 1.5,
                "slt": 0.8,
                "z": 1.0,
                "u10": 0.77,
                "v10": 0.66,
                "lsm": 1.2,
            },
            "edaphic_variables": {
                "swvl1": 1.1,
                "swvl2": 0.9,
                "stl1": 0.7,
                "stl2": 0.6,
            },
            "atmospheric_variables": {"z": 2.8, "t": 1.7, "u": 0.87, "v": 0.6, "q": 0.78},
            "climate_variables": {
                "smlt": 1.0,
                "tp": 2.2,
                "csfr": 0.6,
                "avg_sdswrf": 0.9,
                "avg_snswrf": 0.7,
                "avg_snlwrf": 0.5,
                "avg_tprate": 2.0,
                "avg_sdswrfcs": 0.5,
                "sd": 0.9,
                "t2m": 2.5,
                "d2m": 1.3,
            },
            "vegetation_variables": {"NDVI": 0.8},
            "land_variables": {"Land": 0.6},
            "agriculture_variables": {"Agriculture": 0.4, "Arable": 0.3, "Cropland": 0.4},
            "forest_variables": {"Forest": 1.2},
            "redlist_variables": {"RLI": 1.3},
            "misc_variables": {"avg_slhtf": 1.2, "avg_pevr": 1.0},
            "species_variables": 10.0,
        }

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

    def forward(self, batch, lead_time: int | None, batch_size: int = 1):
        """
        Forward pass of the model.

        Args:
            batch: Batch object containing input variables and metadata
            lead_time (int): Time difference in months between input and target (default=self.lead_time)

        Returns:
            dict: Dictionary containing decoded outputs for each variable category

        """
        if not lead_time:
            lead_time = self.lead_time
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
        x, y = batch
        output = self(x, self.lead_time, batch_size=self.batch_size)
        print("Validation time!")
        loss = self.compute_loss(output, y)
        self.log(
            "val_loss", loss, batch_size=self.batch_size, sync_dist=True
        )  # on_step=False, on_epoch=True, prog_bar=True, sync_dist=True
        return loss

    def training_step(self, batch, batch_idx):
        x, y = batch
        output = self(x, self.lead_time, batch_size=self.batch_size)
        loss = self.compute_loss(output, y)
        self.log("train_loss", loss, batch_size=self.batch_size, sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        output = self(x, self.lead_time, batch_size=self.batch_size)
        print("Test time")
        loss = self.compute_loss(output, y)
        self.log("test_loss", loss, batch_size=self.batch_size, sync_dist=True)
        return loss

    def predict_step(self, batch, batch_idx):
        records = []
        x, y = batch
        output = self(x, self.lead_time, batch_size=self.batch_size)
        # pred_cpu = detach_output_dict(output) # helper does detach.clone().cpu()
        # gt_cpu   = detach_batch(y) # The first timestep is the ground truth
        records.append(
            {
                "idx": batch_idx,
                "pred": output,
                "gt": y,
            }
        )
        return records

    # def on_after_backward(self):
    #     """
    #     Checker for not learnable parameters -> Should output nothing!
    #     Does NOT work with FSDP as it flatts the params and makes proxies
    #     Use it with single GPU to check the params & grads
    #     """
    #     for name, param in self.named_parameters():
    #         if param.grad is None:
    #             print(name)

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

        total_loss = 0.0
        count = 0
        current_land_mask = None

        mask_available_and_needed = self.land_sea_mask is not None and self.use_mask != "no"
        if mask_available_and_needed:
            try:
                sample_output_group_key = next(iter(output))
                sample_output_var_key = next(iter(output[sample_output_group_key]))
                target_device = output[sample_output_group_key][sample_output_var_key].device
                current_land_mask = self.land_sea_mask.to(target_device)
            except Exception as e:
                print(f"Error preparing land_sea_mask: {e}. Proceeding without mask.")
                mask_available_and_needed = False  # welp, no mask if error

        for group_name in groups:
            if group_name not in output or group_name not in batch._asdict():
                continue

            pred_dict = output[group_name]
            true_dict = getattr(batch, group_name)
            group_loss = 0.0
            var_count = 0

            apply_mask_to_group = mask_available_and_needed and (
                self.use_mask == "fully" or (self.use_mask == "partially" and (group_name in self.partially_masked_groups))
            )

            for var_name, pred_tensor in pred_dict.items():
                if var_name not in true_dict:
                    continue
                gt_tensor = true_dict[var_name]

                # Determine target tensor based on td_learning
                target_tensor = gt_tensor[:, 1]
                prediction_for_loss = pred_tensor
                if self.td_learning:
                    target_tensor = gt_tensor[:, 1] - gt_tensor[:, 0]
                    prediction_for_loss = pred_tensor - gt_tensor[:, 0]

                abs_error_map = torch.abs(prediction_for_loss - target_tensor)

                loss_var = torch.tensor(0.0, device=pred_tensor.device)
                use_masked_loss_for_var = (
                    apply_mask_to_group
                    and current_land_mask is not None
                    and abs_error_map.ndim >= 2
                    and abs_error_map.shape[-2:] == current_land_mask.shape
                )

                if use_masked_loss_for_var:
                    broadcastable_mask = current_land_mask
                    if abs_error_map.ndim == 3:
                        broadcastable_mask = current_land_mask.unsqueeze(0)
                    elif abs_error_map.ndim == 4:
                        broadcastable_mask = current_land_mask.unsqueeze(0).unsqueeze(0)

                    masked_error_sum = torch.sum(abs_error_map * broadcastable_mask)
                    num_elements_for_mean = torch.sum(broadcastable_mask.expand_as(abs_error_map))
                    loss_var = (
                        masked_error_sum / num_elements_for_mean
                        if num_elements_for_mean > 0
                        else torch.tensor(0.0, device=pred_tensor.device)
                    )
                else:
                    loss_var = torch.mean(abs_error_map)

                self.log(f"{group_name}_{var_name}_loss", loss_var, batch_size=gt_tensor.size(0))

                group_weights = self.variable_weights.get(group_name, {})
                w = group_weights.get(var_name, 1.0) if isinstance(group_weights, dict) else group_weights
                group_loss += w * loss_var
                var_count += 1

            if var_count > 0:
                group_loss /= var_count  # average within group
                total_loss += group_loss
                count += 1

        final_total_loss = (
            total_loss / count
            if count > 0
            else torch.tensor(
                0.0,
                device=output[next(iter(output))][next(iter(output[next(iter(output))]))].device if output else torch.tensor(0.0),
            )
        )
        print(f"Total loss {final_total_loss}")
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
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2000, eta_min=self.learning_rate / 10)
        # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=self.lr_lambda)

        return [optimizer], [scheduler]
