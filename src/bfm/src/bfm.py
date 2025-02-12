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
from src.bfm.src.dataloder import crop_variables
from src.bfm.src.decoder import BFMDecoder
from src.bfm.src.encoder import BFMEncoder
from src.mvit.mvit_model import MViT
from src.swin_transformer.core.swim_core_v2 import Swin3DTransformer

DEVICE = "cuda:0"

class BFM(nn.Module):
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
        species_distr_vars: tuple[str, ...],
        land_vars: tuple[str, ...],
        agriculture_vars: tuple[str, ...],
        forest_vars: tuple[str, ...],
        atmos_levels: list[int],
        species_num: int,
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
            species_distr_vars=species_distr_vars,
            land_vars=land_vars,
            agriculture_vars=agriculture_vars,
            forest_vars=forest_vars,
            atmos_levels=atmos_levels,
            species_num=species_num,
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
            species_distr_vars=species_distr_vars,
            land_vars=land_vars,
            agriculture_vars=agriculture_vars,
            forest_vars=forest_vars,
            atmos_levels=atmos_levels,
            species_num=species_num,
            H=H,
            W=W,
            embed_dim=embed_dim,
            **kwargs,
        )

    def forward(self, batch, lead_time, batch_size):
        """
        Forward pass of the model.

        Args:
            batch: Batch object containing input variables and metadata
            lead_time (timedelta): Time difference between input and target

        Returns:
            dict: Dictionary containing decoded outputs for each variable category

        """
        print(f"BFM batch size: {batch_size}")
        ### V1 
        encoded = self.encoder(batch, lead_time, batch_size)
        print("Encoded shape", encoded.shape)

        # calculate number of patches in 2D
        num_patches_h = self.H // self.encoder.patch_size
        num_patches_w = self.W // self.encoder.patch_size
        total_patches = num_patches_h * num_patches_w  # noqa

        # calculate depth to match the sequence length
        depth = encoded.shape[1] // (num_patches_h * num_patches_w)
        print(f"BFM depth: {depth} | patch_size {self.encoder.patch_shape} | encoder shape {encoded.shape}")
        patch_shape = (
            depth,  # depth dimension matches sequence length / (H*W)
            num_patches_h,  # height in patches
            num_patches_w,  # width in patches
        )

        if self.backbone_type == "mvit":
            encoded = encoded.view(encoded.size(0), -1, self.encoder.embed_dim)
            print(f"Reshaped encoded for MViT: {encoded.shape}")

        backbone_output = self.backbone(encoded, lead_time=lead_time, rollout_step=0, patch_shape=patch_shape)
        print("Backbone output", backbone_output.shape)
        # decode
        output = self.decoder(backbone_output, batch, lead_time)
        # print("Decoded output:", output)
        return output


        # 1) Encode input: shape [B_local, L_total, E]

        ### V2 
        # encoded = self.encoder(batch, lead_time)
        # B_local, L_total, E = encoded.shape

        # # 2) Calculate number of patches in 2D
        # num_patches_h = self.H // self.encoder.patch_size
        # num_patches_w = self.W // self.encoder.patch_size
        # patches_per_sample = num_patches_h * num_patches_w

        # # ----------------------------------------------------------------
        # # 3) Fix: Derive "depth" from tokens-per-sample
        # #    We assume each sample is split into depth * patches_per_sample tokens.
        # # ----------------------------------------------------------------
        # if L_total % B_local != 0:
        #     raise ValueError(
        #         f"Encoded tokens ({L_total}) not divisible by local batch size ({B_local}). "
        #         "Check your encoder or batch size setup."
        #     )

        # tokens_per_sample = L_total // B_local
        # if tokens_per_sample % patches_per_sample != 0:
        #     raise ValueError(
        #         f"tokens_per_sample ({tokens_per_sample}) not divisible by patch count ({patches_per_sample}). "
        #         "Check your patch size or encoder output."
        #     )

        # depth = tokens_per_sample // patches_per_sample

        # print(
        #     f"BFM depth: {depth} | patch_size {self.encoder.patch_shape} | "
        #     f"encoder shape {encoded.shape}"
        # )
        # patch_shape = (depth, num_patches_h, num_patches_w)
        
        #########################################
        ####### V3
               # ---------------------------------------------------
        # Step 1: local encoding
        # ---------------------------------------------------
        # encoded_local = self.encoder(batch, lead_time)  # [B_local, L_local, E]
        # B_local, L_local, E = encoded_local.shape
        # rank = dist.get_rank() if dist.is_initialized() else 0

        # # ---------------------------------------------------
        # # Step 2: gather all local encodings => global
        # # ---------------------------------------------------
        # global_encoded = self._all_gather_encoded_simple(encoded_local)
        # B_global, L_global, E = global_encoded.shape
        # print(f"[Rank {rank}] global_encoded shape={global_encoded.shape}")

        # # ---------------------------------------------------
        # # Step 3: compute patch shape from the global dims
        # # ---------------------------------------------------
        # # Suppose each sample has an image of shape (H, W). 
        # # We do a 2D patching: h = H//patch_size, w = W//patch_size
        # num_patches_h = self.H // self.encoder.patch_size
        # num_patches_w = self.W // self.encoder.patch_size
        # patches_per_sample = num_patches_h * num_patches_w

        # # tokens_per_sample = L_global // B_global
        # # If we assume each sample is identical in size, do:
        # if (L_global % B_global) != 0:
        #     raise ValueError(
        #         f"L_global={L_global} not divisible by B_global={B_global}. "
        #         "Check if your data is truly one-sample or multiple-sample."
        #     )
        # tokens_per_sample = L_global // B_global

        # # Now, depth is how many tokens remain after accounting for (num_patches_h * num_patches_w)
        # if (tokens_per_sample % patches_per_sample) != 0:
        #     raise ValueError(
        #         f"tokens_per_sample={tokens_per_sample} not divisible by "
        #         f"patches_per_sample={patches_per_sample}. Adjust patch size or data."
        #     )
        # depth = tokens_per_sample // patches_per_sample

        # patch_shape = (depth, num_patches_h, num_patches_w)
        # print(f"[Rank {rank}] Computed patch_shape={patch_shape}, depth={depth}")

        # #TODO Test this 
        # if self.backbone_type == "mvit":
        #     encoded = encoded.view(encoded.size(0), -1, self.encoder.embed_dim)
        #     print(f"Reshaped encoded for MViT: {encoded.shape}")

        # backbone_output = self.backbone(global_encoded, lead_time=lead_time, rollout_step=0, patch_shape=patch_shape)

        # # decode
        # output = self.decoder(backbone_output, batch, lead_time)


        # ### V4
        #     # 1) Locally encode: shape [B_local, L_local, E]
        # encoded_local = self.encoder(batch, lead_time)

        # # 2) If you have a single sample split by tokens, gather along dim=1
        # global_encoded = self._all_gather_tokens(encoded_local)
        # B_local, L_global, E = global_encoded.shape
        # # => B_local is the same as before, L_global = world_size * L_local

        # # 3) If you truly have only 1 local sample per rank => B_local=1
        # #    Then total "batch" is still 1. 
        # #    So you can do: B_global = B_local (which might be 1)
        # #    tokens_per_sample = L_global // B_global = L_global
        # B_global = B_local  # e.g. 1
        # tokens_per_sample = L_global // B_global  # e.g. L_global

        # # 4) Now tokens_per_sample should be 27360 if 2 ranks each had 13680 tokens
        # #    Then patches_per_sample = (H//patch_size) * (W//patch_size) = 3040
        # #    depth = tokens_per_sample // patches_per_sample = 27360 // 3040 = 9
        # patches_per_sample = (self.H // self.encoder.patch_size) * (self.W // self.encoder.patch_size)
        # if tokens_per_sample % patches_per_sample != 0:
        #     raise ValueError(
        #         f"tokens_per_sample={tokens_per_sample} not divisible by patches_per_sample={patches_per_sample}"
        #     )
        # depth = tokens_per_sample // patches_per_sample

        # # 5) Patch shape
        # patch_shape = (depth, self.H // self.encoder.patch_size, self.W // self.encoder.patch_size)

        # # 6) Pass to backbone
        # backbone_output = self.backbone(global_encoded, lead_time=lead_time, rollout_step=0, patch_shape=patch_shape)
        # output = self.decoder(backbone_output, batch, lead_time)
        # return output

    def _all_gather_tokens(self, encoded_local: torch.Tensor) -> torch.Tensor:
        """
        Gathers partial tokens from each rank along the tokens dimension (dim=1).
        This is for the case where a single sample is horizontally (or by tokens) split
        across ranks, so that each rank has [B_local, partial_L, E].
        
        After all_gather, we cat on dim=1 -> [B_local, world_size*partial_L, E].
        """
        if not dist.is_initialized():
            return encoded_local  # single process fallback

        # 1) all_gather list
        world_size = dist.get_world_size()
        gather_list = [torch.zeros_like(encoded_local) for _ in range(world_size)]
        dist.all_gather(gather_list, encoded_local)  
        # gather_list[r] now has shape [B_local, L_local, E] from rank r

        # 2) cat along dim=1 (token dimension)
        global_encoded = torch.cat(gather_list, dim=1)
        return global_encoded


    def _all_gather_encoded_simple(self, encoded_local: torch.Tensor) -> torch.Tensor:
        """
        Gathers each rank's [B_local, L_local, E] into a single global tensor
        of shape [B_global, L_local, E], where B_global = sum of all B_local.
        This uses torch.distributed.all_gather(), so each rank ends up with
        the same final tensor.

        IMPORTANT:
        - Assumes all ranks have the same (L_local, E) shapes, just different B_local
            (i.e., if you are sharding by batch).
        - If your data is sharded differently (e.g. by tokens dimension), adapt accordingly.
        """
        if not dist.is_initialized():
            # Single-process fallback
            return encoded_local

        world_size = dist.get_world_size()

        # 1) All ranks create a list of placeholders for the gather
        #    Each element in 'all_list' will have the same shape as 'encoded_local'.
        all_list = [torch.zeros_like(encoded_local) for _ in range(world_size)]

        # 2) all_gather => after this call, each rank's all_list[i] will contain rank i's local tensor
        dist.all_gather(all_list, encoded_local)

        # 3) Concatenate along the batch dimension (dim=0), forming [B_global, L_local, E]
        # We need to concat on the token dimension (dim=1)
        global_encoded = torch.cat(all_list, dim=1)

        # Now every rank has the same global_encoded
        return global_encoded

    def _all_gather_encoded(self, encoded_local: torch.Tensor) -> torch.Tensor:
        """
        Gathers local encoder outputs [B_local, L_local, E] from all ranks into a single
        global tensor [B_global, L_global, E], so that your subsequent patch logic won't fail.
        
        This method can be adapted to your data distribution (whether you're splitting by batch or tokens).
        Here we assume you're collecting the entire batch/tokens on rank=0, then broadcasting.
        """
        if not dist.is_initialized():
            # Single-process fallback
            return encoded_local
        
        # Prepare shape info
        local_shape = torch.tensor(encoded_local.shape, dtype=torch.long, device=encoded_local.device)  # [3]
        all_shape_list = [torch.zeros_like(local_shape) for _ in range(dist.get_world_size())]

        # Gather shape from all ranks
        dist.all_gather(all_shape_list, local_shape)
        shape_tuples = [(s[0].item(), s[1].item(), s[2].item()) for s in all_shape_list]

        # For simplicity, let's assume:
        #   1) Everyone has the same L_local if it's truly one sample split by batch
        #   2) We'll sum up B_local for the global B
        B_locals = [st[0] for st in shape_tuples]
        L_locals = [st[1] for st in shape_tuples]
        E_locals = [st[2] for st in shape_tuples]

        # Check that all E are identical, and all L are identical if that's your scenario
        embed_dim = E_locals[0]
        if any(e != embed_dim for e in E_locals):
            raise ValueError("All-gathered embeddings have mismatched E dims among ranks!")
        # Example: L might also be the same if each rank is chunking by B only
        # or you might sum L if you chunked by tokens. Adjust as needed.
        L_global = L_locals[0]  # if the same
        B_global = sum(B_locals)

        rank = dist.get_rank()
        world_size = dist.get_world_size()
        
        if rank == 0:
            global_encoded = torch.zeros(
                (B_global, L_global, embed_dim),
                dtype=encoded_local.dtype,
                device=encoded_local.device
            )
        else:
            # allocate minimal placeholder
            global_encoded = torch.zeros(
                (1, 1, embed_dim),
                dtype=encoded_local.dtype,
                device=encoded_local.device
            )
        
        # Now gather the data content
        # We'll do a naive gather_by_rank approach
        # Another approach is "dist.all_gather_into_tensor" if you have a recent PyTorch
        offset_b = 0
        for i in range(world_size):
            b_i = B_locals[i]
            l_i = L_locals[i]
            # We'll gather rank i's data
            if rank == i:
                # We copy our local data to the global buffer slice (on rank=0)
                dist.send(encoded_local, dst=0)
            elif rank == 0:
                # We receive that slice from rank i
                tmp = torch.zeros((b_i, l_i, embed_dim), dtype=encoded_local.dtype, device=encoded_local.device)
                dist.recv(tmp, src=i)
                global_encoded[offset_b : offset_b + b_i, :l_i, :] = tmp
            offset_b += b_i
        
        # Now broadcast global_encoded from rank=0 back to all ranks
        dist.broadcast(global_encoded, src=0)
        
        return global_encoded
    

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()


def main():
    """Main function for testing the BFM implementation."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    print("\nLoading batches...")
    batches = load_batches("data/")
    print(f"Loaded {len(batches)} batches")

    # some info about the first batch
    batch = batches[0]
    print("\nFirst batch variable counts:")
    print(f"Surface variables: {len(batch['surface_variables'])} vars")
    print(f"Single variables: {len(batch['single_variables'])} vars")
    print(f"Atmospheric variables: {len(batch['atmospheric_variables'])} vars × {len(batch['batch_metadata']['pressure_levels'])} levels")
    print(f"Species variables: {len(batch['species_extinction_variables'])} vars")
    print(f"Land variables: {len(batch['land_variables'])} vars")
    print(f"Agriculture variables: {len(batch['agriculture_variables'])} vars")
    print(f"Forest variables: {len(batch['forest_variables'])} vars")

    # Determine original spatial dimensions from metadata lists
    H = len(batch["batch_metadata"]["latitudes"])
    W = len(batch["batch_metadata"]["longitudes"])

    # crop dimensions to be divisible by patch size
    patch_size = 4
    new_H = (H // patch_size) * patch_size
    new_W = (W // patch_size) * patch_size

    print(f"\nOriginal spatial dimensions: {H}×{W}")
    print(f"Cropped spatial dimensions: {new_H}×{new_W}")

    print("\nProcessing batches...")
    for i, batch in enumerate(batches):
        print(f"\nProcessing batch {i+1}/{len(batches)}")
        batch["surface_variables"] = crop_variables(batch["surface_variables"], new_H, new_W)
        batch["single_variables"] = crop_variables(batch["single_variables"], new_H, new_W)
        batch["atmospheric_variables"] = crop_variables(batch["atmospheric_variables"], new_H, new_W)
        batch["species_extinction_variables"] = crop_variables(batch["species_extinction_variables"], new_H, new_W)
        batch["land_variables"] = crop_variables(batch["land_variables"], new_H, new_W)
        batch["agriculture_variables"] = crop_variables(batch["agriculture_variables"], new_H, new_W)
        batch["forest_variables"] = crop_variables(batch["forest_variables"], new_H, new_W)

        # crop metadata dimensions
        batch["batch_metadata"]["latitudes"] = batch["batch_metadata"]["latitudes"][:new_H]
        batch["batch_metadata"]["longitudes"] = batch["batch_metadata"]["longitudes"][:new_W]

    print("\nSpatial dimensions after cropping:")
    print(f"Grid size: {new_H}×{new_W}")
    print(f"Number of patches: {new_H//patch_size}×{new_W//patch_size}")


    # init model
    model = BFM(
        surface_vars=tuple(batch['surface_variables'].keys()),
        single_vars=tuple(batch['single_variables'].keys()),
        atmos_vars=tuple(batch['atmospheric_variables'].keys()),
        species_vars=tuple(batch['species_extinction_variables'].keys()),
        land_vars=tuple(batch['land_variables'].keys()),
        agriculture_vars=tuple(batch['agriculture_variables'].keys()),
        forest_vars=tuple(batch['forest_variables'].keys()),
        atmos_levels=batch['batch_metadata']['pressure_levels'],
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
        timestamp = batch['batch_metadata']['timestamp']

        dt_format = "%Y-%m-%dT%H:%M:%S"

        # Convert the two timestamps into datetime objects
        start = datetime.strptime(timestamp[0], dt_format)
        end = datetime.strptime(timestamp[1], dt_format)

        # Compute lead time in hours
        lead_time_hours = (end - start).total_seconds() / 3600.0

        print("Lead time (hours):", lead_time_hours)
        try:
            with torch.no_grad():
                output = model(batch, lead_time_hours)  # noqa
                print(output)
        except Exception as e:
            print(f"\nError processing batch {i+1}:")
            print(f"Error type: {type(e).__name__}")
            print(f"Error message: {str(e)}")
            raise


# if __name__ == "__main__":
#     main()
