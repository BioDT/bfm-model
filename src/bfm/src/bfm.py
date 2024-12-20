"""
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

Example usage:
    model = BFM(
        surface_vars=('temperature', 'pressure'),
        single_vars=('humidity',),
        atmos_vars=('wind_u', 'wind_v'),
        species_vars=('species_1', 'species_2'),
        land_vars=('soil_moisture',),
        agriculture_vars=('crop_yield',),
        forest_vars=('tree_coverage',),
        atmos_levels=[1000, 850, 700],
        H=32,
        W=64,
        backbone_type='mvit'
    )
    output = model(batch, lead_time)
"""

from collections import namedtuple
from datetime import datetime, timedelta
from typing import Literal
import os

import torch
import torch.nn as nn

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
import torch.distributed as dist
import torch.multiprocessing as mp

from src.bfm.src.dataset_basics import load_batches
from src.bfm.src.decoder import BFMDecoder
from src.bfm.src.encoder import BFMEncoder
from src.mvit.mvit_model import MViT
from src.swin_transformer.core.swim_core_v2 import Swin3DTransformer

DEVICE = "cuda:1"

def crop_variables(variables, new_H, new_W):
    """
    Crop and clean variables to specified dimensions, handling NaN and Inf values.

    Args:
        variables (dict): Dictionary of variable tensors to process
        new_H (int): Target height dimension
        new_W (int): Target width dimension

    Returns:
        dict: Processed variables with cleaned and cropped tensors
    """
    processed_vars = {}
    for k, v in variables.items():
        # crop dimensions
        cropped = v[..., :new_H, :new_W]

        # infinities first
        inf_mask = torch.isinf(cropped)
        inf_count = inf_mask.sum().item()
        if inf_count > 0:
            print(f"\nHandling Inf values in {k}:")
            print(f"Inf count: {inf_count}")
            valid_values = cropped[~inf_mask & ~torch.isnan(cropped)]
            if len(valid_values) > 0:
                max_val = valid_values.max().item()
                min_val = valid_values.min().item()
                cropped = torch.clip(cropped, min_val, max_val)
            else:
                cropped = torch.clip(cropped, -1e6, 1e6)

        # handle NaNs
        nan_mask = torch.isnan(cropped)
        nan_count = nan_mask.sum().item()
        if nan_count > 0:
            print(f"\nHandling NaN values in {k}:")
            print(f"Shape: {cropped.shape}")
            print(f"Total NaN count: {nan_count}")
            valid_values = cropped[~nan_mask & ~torch.isinf(cropped)]
            if len(valid_values) > 0:
                mean_val = valid_values.mean().item()
                std_val = valid_values.std().item()
                # use mean +- 2*std as clipping bounds
                clip_min = mean_val - 2 * std_val
                clip_max = mean_val + 2 * std_val
                # replace NaNs with clipped mean
                cropped = torch.nan_to_num(cropped, nan=mean_val)
                cropped = torch.clip(cropped, clip_min, clip_max)
            else:
                cropped = torch.nan_to_num(cropped, nan=0.0)
                cropped = torch.clip(cropped, -1.0, 1.0)

        processed_vars[k] = cropped
    return processed_vars


class BFM(nn.Module):
    """
    Biodiversity Foundation Model.

    This model combines encoder, backbone and decoder components to process climate and biodiversity-related variables.

    Args:
        surface_vars (tuple[str, ...]): Names of surface-level variables
        single_vars (tuple[str, ...]): Names of single-level variables
        atmos_vars (tuple[str, ...]): Names of atmospheric variables
        species_vars (tuple[str, ...]): Names of species-related variables
        land_vars (tuple[str, ...]): Names of land-related variables
        agriculture_vars (tuple[str, ...]): Names of agriculture-related variables
        forest_vars (tuple[str, ...]): Names of forest-related variables
        atmos_levels (list[int]): Pressure levels for atmospheric variables
        H (int, optional): Height of output grid. Defaults to 32.
        W (int, optional): Width of output grid. Defaults to 64.
        embed_dim (int, optional): Embedding dimension. Defaults to 1024.
        num_latent_tokens (int, optional): Number of latent tokens. Defaults to 8.
        backbone_type (Literal["swin", "mvit"], optional): Type of backbone architecture. Defaults to "mvit".
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
        single_vars: tuple[str, ...],
        atmos_vars: tuple[str, ...],
        species_vars: tuple[str, ...],
        land_vars: tuple[str, ...],
        agriculture_vars: tuple[str, ...],
        forest_vars: tuple[str, ...],
        atmos_levels: list[int],
        H: int = 32,
        W: int = 64,
        embed_dim: int = 1024,
        num_latent_tokens: int = 8,
        backbone_type: Literal["swin", "mvit"] = "mvit",
        patch_size: int = 4,
        **kwargs,
    ):
        super().__init__()
        self.H = H
        self.W = W

        self.encoder = BFMEncoder(
            surface_vars=surface_vars,
            single_vars=single_vars,
            atmos_vars=atmos_vars,
            species_vars=species_vars,
            land_vars=land_vars,
            agriculture_vars=agriculture_vars,
            forest_vars=forest_vars,
            atmos_levels=atmos_levels,
            # patch_size=kwargs.get("patch_size", 4),
            patch_size=patch_size,
            embed_dim=embed_dim,
            H=H,
            W=W,
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
                window_size=(1, 1, 2),
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
            single_vars=single_vars,
            atmos_vars=atmos_vars,
            species_vars=species_vars,
            land_vars=land_vars,
            agriculture_vars=agriculture_vars,
            forest_vars=forest_vars,
            atmos_levels=atmos_levels,
            H=H,
            W=W,
            embed_dim=embed_dim,
            **kwargs,
        )

    def forward(self, batch, lead_time):
        """
        Forward pass of the model.

        Args:
            batch: Batch object containing input variables and metadata
            lead_time (timedelta): Time difference between input and target

        Returns:
            dict: Dictionary containing decoded outputs for each variable category
        """
        encoded = self.encoder(batch, lead_time)

        # calculate number of patches in 2D
        num_patches_h = self.H // self.encoder.patch_size
        num_patches_w = self.W // self.encoder.patch_size
        total_patches = num_patches_h * num_patches_w  # noqa

        # calculate depth to match the sequence length
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

        # decode
        output = self.decoder(backbone_output, batch, lead_time)

        return output

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()


def main():
    """Main function for testing the BFM implementation."""
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    print("\nLoading batches...")
    batches = load_batches("data/", device=device)
    print(f"Loaded {len(batches)} batches")

    # some info about the first batch
    batch = batches[0]
    print("\nFirst batch variable counts:")
    print(f"Surface variables: {len(batch.surface_variables)} vars")
    print(f"Single variables: {len(batch.single_variables)} vars")
    print(f"Atmospheric variables: {len(batch.atmospheric_variables)} vars × {len(batch.batch_metadata.pressure_levels)} levels")
    print(f"Species variables: {len(batch.species_extinction_variables)} vars")
    print(f"Land variables: {len(batch.land_variables)} vars")
    print(f"Agriculture variables: {len(batch.agriculture_variables)} vars")
    print(f"Forest variables: {len(batch.forest_variables)} vars")

    # crop dimensions to be divisible by patch size (or just set patch size to 1)
    patch_size = 4
    H, W = batch.batch_metadata.latitudes, batch.batch_metadata.longitudes
    new_H = (H // patch_size) * patch_size
    new_W = (W // patch_size) * patch_size

    print(f"\nOriginal spatial dimensions: {H}×{W}")
    print(f"Cropped spatial dimensions: {new_H}×{new_W}")

    print("\nProcessing batches...")
    for i, batch in enumerate(batches):
        print(f"\nProcessing batch {i+1}/{len(batches)}")
        batch.surface_variables = crop_variables(batch.surface_variables, new_H, new_W)
        batch.single_variables = crop_variables(batch.single_variables, new_H, new_W)
        batch.atmospheric_variables = crop_variables(batch.atmospheric_variables, new_H, new_W)
        batch.species_extinction_variables = crop_variables(batch.species_extinction_variables, new_H, new_W)
        batch.land_variables = crop_variables(batch.land_variables, new_H, new_W)
        batch.agriculture_variables = crop_variables(batch.agriculture_variables, new_H, new_W)
        batch.forest_variables = crop_variables(batch.forest_variables, new_H, new_W)

        # crop metadata dimensions
        batch.batch_metadata.latitudes = batch.batch_metadata.latitudes[:new_H]
        batch.batch_metadata.longitudes = batch.batch_metadata.longitudes[:new_W]

    print("\nSpatial dimensions after cropping:")
    print(f"Grid size: {new_H}×{new_W}")
    print(f"Number of patches: {new_H//patch_size}×{new_W//patch_size}")

    # init model
    model = BFM(
        surface_vars=tuple(batch.surface_variables.keys()),
        single_vars=tuple(batch.single_variables.keys()),
        atmos_vars=tuple(batch.atmospheric_variables.keys()),
        species_vars=tuple(batch.species_extinction_variables.keys()),
        land_vars=tuple(batch.land_variables.keys()),
        agriculture_vars=tuple(batch.agriculture_variables.keys()),
        forest_vars=tuple(batch.forest_variables.keys()),
        atmos_levels=batch.batch_metadata.pressure_levels,
        H=new_H,
        W=new_W,
        backbone_type="mvit",  # or "swin"
    )

    # RANK = 1
    # torch.cuda.set_device(RANK)

    # model = model.to(RANK)
    # model = FSDP(model)
    model.eval()
    # pass each batch through the model
    for i, batch in enumerate(batches):
        print(f"\nProcessing batch {i+1}/{len(batches)}")
        t1, t2 = batch.batch_metadata.timestamp
        lead_time = t2 - t1

        try:
            with torch.no_grad():
                output = model(batch, lead_time)  # noqa

        except Exception as e:
            print(f"\nError processing batch {i+1}:")
            print(f"Error type: {type(e).__name__}")
            print(f"Error message: {str(e)}")
            raise


if __name__ == "__main__":
    main()
