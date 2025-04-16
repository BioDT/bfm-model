"""
Copyright (C) 2025 TNO, The Netherlands. All rights reserved.

Encoder module for processing biodiversity data using a Perceiver IO architecture.

This module provides a flexible encoder that processes multiple types of environmental variables
(surface, atmospheric, species, etc.) through patch embeddings and structured latent transformations.
It uses a Perceiver IO architecture to handle variable-length inputs and outputs fixed-size latent
representations.

The encoder handles:
- Surface variables (e.g. temperature)
- Single-level variables (e.g. elevation)
- Atmospheric variables at multiple pressure levels
- Species-related variables
- Land use variables
- Agricultural variables
- Forest variables

Key Features:
- Patch-based tokenization of spatial data
- Fourier position encodings
- Structured latent representations for different variable groups
- Time embeddings for temporal information
- Multi-head attention for cross-attention and self-attention
- Dropout and layer normalization

Example:
    encoder = BFMEncoder(
        surface_vars=('temp', 'precip'),
        single_vars=('elev',),
        atmos_vars=('pressure',),
        species_vars=('richness',),
        land_vars=('landcover',),
        agriculture_vars=('cropland',),
        forest_vars=('treecover',),
        atmos_levels=(1000, 850, 700),
        patch_size=4,
        H=152,
        W=320
    )
    output = encoder(batch, lead_time)
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from bfm_model.bfm.dataset_basics import load_batches
from bfm_model.perceiver_components.helpers_io import dropout_seq
from bfm_model.perceiver_components.pos_encoder import build_position_encoding
from bfm_model.perceiver_core.perceiver_io import PerceiverIO


class BFMEncoder(nn.Module):
    """
    Encoder module for processing biodiversity data using a Perceiver IO architecture.

    This encoder handles multiple types of variables (surface, atmospheric, species, etc.) and
    processes them through patch embeddings and structured latent transformations.

    Args:
        surface_vars (tuple[str, ...]): Names of surface variables
        single_vars (tuple[str, ...]): Names of single-level variables
        atmos_vars (tuple[str, ...]): Names of atmospheric variables
        species_vars (tuple[str, ...]): Names of species-related variables
        species_distr_vars (tuple[str, ...]): Species distributions-related variables
        land_vars (tuple[str, ...]): Names of land variables
        agriculture_vars (tuple[str, ...]): Names of agriculture variables
        forest_vars (tuple[str, ...]): Names of forest variables
        atmos_levels (tuple[int, ...]): Atmospheric pressure levels
        species_num (int): Number of species distribution to account for
        patch_size (int, optional): Size of patches for tokenization. Defaults to 4
        perceiver_latents (int, optional): Number of latent tokens. Defaults to 256
        embed_dim (int, optional): Embedding dimension. Defaults to 128
        num_heads (int, optional): Number of attention heads. Defaults to 4
        head_dim (int, optional): Dimension of each attention head. Defaults to 16
        drop_rate (float, optional): Dropout rate. Defaults to 0.1
        depth (int, optional): Number of transformer layers. Defaults to 2
        mlp_ratio (float, optional): MLP expansion ratio. Defaults to 4.0
        max_history_size (int, optional): Maximum history window size. Defaults to 2
        perceiver_ln_eps (float, optional): Layer norm epsilon. Defaults to 1e-5
        num_fourier_bands (int, optional): Number of Fourier bands. Defaults to 64
        max_frequency (float, optional): Maximum frequency for position encoding. Defaults to 224.0
        num_input_axes (int, optional): Number of input axes. Defaults to 1
        position_encoding_type (str, optional): Type of position encoding. Defaults to "fourier"
        H (int, optional): Height of input grid. Defaults to 152
        W (int, optional): Width of input grid. Defaults to 320
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
        atmos_levels: tuple[int, ...],
        species_num: int,
        patch_size: int = 4,
        perceiver_latents: int = 256,
        embed_dim: int = 1024,
        num_heads: int = 16,
        kv_heads: int = 8,
        head_dim: int = 2,
        drop_rate: float = 0.1,
        depth: int = 2,
        mlp_ratio: float = 4.0,
        max_history_size: int = 2,
        perceiver_ln_eps: float = 1e-5,
        num_fourier_bands: int = 64,
        max_frequency: float = 224.0,
        num_input_axes: int = 1,
        position_encoding_type: str = "fourier",
        H: int = 152,
        W: int = 320,
    ):
        super().__init__()
        # basic config
        self.drop_rate = drop_rate
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.max_history_size = max_history_size
        self.num_input_axes = num_input_axes
        self.max_frequency = max_frequency
        self.num_fourier_bands = num_fourier_bands
        self.position_encoding_type = position_encoding_type
        self.perceiver_latents = perceiver_latents
        self.num_heads = num_heads
        self.num_kv_heads = kv_heads
        self.head_dim = head_dim
        self.depth = depth
        self.mlp_ratio = mlp_ratio

        # variable names
        self.surface_vars = surface_vars
        self.single_vars = single_vars
        self.atmos_vars = atmos_vars
        self.species_vars = species_vars
        self.species_distr_vars = species_distr_vars
        self.land_vars = land_vars
        self.agriculture_vars = agriculture_vars
        self.forest_vars = forest_vars
        self.atmos_levels = atmos_levels
        self.species_num = species_num

        # variable mappings
        self.var_maps = {
            "surface": {v: i for i, v in enumerate(surface_vars)},
            "single": {v: i for i, v in enumerate(single_vars)},
            "atmos": {v: i for i, v in enumerate(atmos_vars)},
            "species": {v: i for i, v in enumerate(species_vars)},
            "species_distr": {v: i for i, v in enumerate(species_distr_vars)},
            "land": {v: i for i, v in enumerate(land_vars)},
            "agriculture": {v: i for i, v in enumerate(agriculture_vars)},
            "forest": {v: i for i, v in enumerate(forest_vars)},
        }

        # init embeddings
        pos_encoding_dim = self._calculate_pos_encoding_dim()

        # position and coordinate embeddings
        self.pos_embed = nn.Linear(pos_encoding_dim, embed_dim)  # fourier features + original coords
        self.lead_time_embed = nn.Linear(embed_dim, embed_dim)
        self.absolute_time_embed = nn.Linear(embed_dim, embed_dim)

        # embedding for atmospheric pressure levels
        self.atmos_levels_embed = nn.Linear(pos_encoding_dim, embed_dim)

        # embedding for species distributions
        self.species_distribution_embed = nn.Linear(pos_encoding_dim, embed_dim)
        # token embeddings for each variable type
        self.surface_token_embeds = self._create_patch_embed(len(surface_vars), patch_size, embed_dim, max_history_size)
        self.single_token_embeds = self._create_patch_embed(len(single_vars), patch_size, embed_dim, max_history_size)
        self.atmos_token_embeds = self._create_patch_embed(len(atmos_vars), patch_size, embed_dim, max_history_size)
        self.species_token_embeds = self._create_patch_embed(len(species_vars), patch_size, embed_dim, max_history_size)
        self.species_distr_token_embeds = self._create_patch_embed(
            len(species_distr_vars), patch_size, embed_dim, max_history_size
        )
        self.land_token_embeds = self._create_patch_embed(len(land_vars), patch_size, embed_dim, max_history_size)
        self.agriculture_token_embeds = self._create_patch_embed(len(agriculture_vars), patch_size, embed_dim, max_history_size)
        self.forest_token_embeds = self._create_patch_embed(len(forest_vars), patch_size, embed_dim, max_history_size)

        # init latent queries
        self.latents = nn.Parameter(torch.randn(perceiver_latents, embed_dim))

        # dropout
        self.pos_drop = nn.Dropout(p=drop_rate)

        # add normalization layer
        self.pre_perceiver_norm = nn.LayerNorm(embed_dim)

        # Add H and W as instance variables
        self.H = H
        self.W = W

        # Initialize perceiver
        self._initialize_perceiver(H, W)

        # init weights
        self.apply(self._init_weights)

    @property
    def patch_shape(self):
        """
        Calculate the patch shape based on current dimensions.

        Returns:
            tuple: Shape of patches as (num_latents, height_patches, width_patches)
        """
        return (self.perceiver_latents, self.H // self.patch_size, self.W // self.patch_size)

    def _calculate_pos_encoding_dim(self):
        """
        Calculate dimension for positional encoding.

        Returns:
            int: Dimension of positional encoding
        """
        return 2 * self.num_fourier_bands * 2 + 2

    def _create_patch_embed(self, num_vars: int, patch_size: int, embed_dim: int, max_history_size: int) -> nn.Linear:
        """
        Create patch embedding layer for a variable type.

        Args:
            num_vars (int): Number of variables
            patch_size (int): Size of patches
            embed_dim (int): Embedding dimension
            max_history_size (int): Maximum history window size

        Returns:
            nn.Linear: Linear layer for patch embedding
            Shape: [num_vars * patch_size * patch_size * max_history_size, embed_dim]
        """
        return nn.Linear(num_vars * patch_size * patch_size * max_history_size, embed_dim)

    def _init_weights(self, m):
        """
        Initialize weights for linear layers and layer norm.

        Args:
            m (nn.Module): Module to initialize
        """
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _initialize_perceiver(self, H: int, W: int):
        """
        Initialize the Perceiver IO model with structured latents.

        Args:
            H (int): Height of input grid
            W (int): Width of input grid
        """
        # get number of patches
        # TODO Check why this gives weird error. For now hardcode the # of patches
        num_patches = (H // self.patch_size) * (W // self.patch_size)
        # num_patches = 3040
        print(f"Num of patches in Encoder: {num_patches}")
        # set the device from an existing parameter or default to CPU
        device = (
            next(self.parameters()).device
            if list(self.parameters())
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

        # Calculate structured latent tokens
        surface_latents = num_patches if self.surface_vars else 0
        single_latents = num_patches if self.single_vars else 0
        atmos_latents = num_patches * len(self.atmos_levels) if self.atmos_vars else 0
        species_latents = num_patches if self.species_vars else 0
        species_distr_latents = num_patches * self.species_num if self.species_distr_vars else 0
        land_latents = num_patches if self.land_vars else 0
        agri_latents = num_patches if self.agriculture_vars else 0
        forest_latents = num_patches if self.forest_vars else 0

        # store latent sizes for forward pass
        self.latent_sizes = {
            "surface": surface_latents,
            "single": single_latents,
            "atmos": atmos_latents,
            "species": species_latents,
            "species_distr": species_distr_latents,
            "land": land_latents,
            "agriculture": agri_latents,
            "forest": forest_latents,
        }

        # initialize structured latents only if needed
        latent_list = []
        if surface_latents > 0:
            self.surface_latents = nn.Parameter(torch.randn(surface_latents, self.embed_dim, device=device))
            latent_list.append(self.surface_latents)
        if single_latents > 0:
            self.single_latents = nn.Parameter(torch.randn(single_latents, self.embed_dim, device=device))
            latent_list.append(self.single_latents)
        if atmos_latents > 0:
            self.atmos_latents = nn.Parameter(torch.randn(atmos_latents, self.embed_dim, device=device))
            latent_list.append(self.atmos_latents)
        if species_latents > 0:
            self.species_latents = nn.Parameter(torch.randn(species_latents, self.embed_dim, device=device))
            latent_list.append(self.species_latents)
        if species_distr_latents > 0:
            self.species_distr_latents = nn.Parameter(torch.randn(species_distr_latents, self.embed_dim, device=device))
            latent_list.append(self.species_distr_latents)
        if land_latents > 0:
            self.land_latents = nn.Parameter(torch.randn(land_latents, self.embed_dim, device=device))
            latent_list.append(self.land_latents)
        if agri_latents > 0:
            self.agri_latents = nn.Parameter(torch.randn(agri_latents, self.embed_dim, device=device))
            latent_list.append(self.agri_latents)
        if forest_latents > 0:
            self.forest_latents = nn.Parameter(torch.randn(forest_latents, self.embed_dim, device=device))
            latent_list.append(self.forest_latents)

        # initialize Perceiver IO with total latents
        total_latents = sum(self.latent_sizes.values())
        print(f"total latens {total_latents}")

        # combine all latents for backward compatibility
        self.latents = nn.Parameter(
            torch.cat(latent_list, dim=0) if latent_list else torch.randn(total_latents, self.embed_dim, device=device)
        )
        print(f"latens shape: {self.latents.shape}")
        # Initialize Perceiver IO
        self.perceiver_io = PerceiverIO(
            num_layers=self.depth,
            dim=self.embed_dim,
            queries_dim=self.embed_dim,
            logits_dimension=None,
            num_latent_tokens=total_latents,
            latent_dimension=self.embed_dim,
            cross_attention_heads=self.num_heads,
            latent_attention_heads=self.num_heads,
            cross_attention_head_dim=self.head_dim,
            latent_attention_head_dim=self.head_dim,
            num_kv_heads=self.num_kv_heads,
            sequence_dropout_prob=self.drop_rate,
            num_fourier_bands=self.num_fourier_bands,
            max_frequency=self.max_frequency,
            num_input_axes=self.num_input_axes,
            position_encoding_type=self.position_encoding_type,
        ).to(device)

    def _apply_position_encoding(self, input_data):
        """
        Apply position encoding to input data.

        Args:
            input_data (torch.Tensor): Input tensor to encode
            Shape: [batch_size, sequence_length, input_dim]

        Returns:
            torch.Tensor: Position encoded tensor
            Shape: [batch_size, sequence_length, input_dim + pos_encoding_dim]
        """
        if self.position_encoding_type in ["fourier", "trainable"]:
            pos_encoder = self._build_position_encoding(input_data.shape)
            pos_encoding = pos_encoder(batch_size=input_data.shape[0])
            self._check_tensor(pos_encoding, "Position encoding")

            input_data = torch.cat((input_data, pos_encoding), dim=-1)

        return input_data

    def _check_tensor(self, tensor, name, replace_values=True, group=None):
        """
        Check tensor for NaN, inf, and extreme values with optional group context.

        Args:
            tensor (torch.Tensor): Tensor to check
            name (str): Name of tensor for logging
            replace_values (bool, optional): Whether to replace invalid values. Defaults to True
            group (str, optional): Group context for logging. Defaults to None

        Returns:
            torch.Tensor: Cleaned tensor if replace_values=True, else bool indicating issues
        """
        stats = {
            "nan_count": torch.isnan(tensor).sum().item(),
            "inf_count": torch.isinf(tensor).sum().item(),
            "mean": tensor.mean().item() if not torch.isnan(tensor.mean()) else "NaN",
            "std": tensor.std().item() if not torch.isnan(tensor.std()) else "NaN",
            "min": tensor.min().item() if not torch.isnan(tensor.min()) else "NaN",
            "max": tensor.max().item() if not torch.isnan(tensor.max()) else "NaN",
        }

        has_issues = stats["nan_count"] > 0 or stats["inf_count"] > 0

        if has_issues and replace_values:
            tensor = torch.nan_to_num(tensor, nan=0.0, posinf=1e6, neginf=-1e6)

        return tensor if replace_values else has_issues

    # V3 trying to fix batch size > 1
    def process_variable_group(self, variables, token_embeds, group_name):
        """
        Process a group of variables through tokenization and embedding in a standard multi-batch manner.

        Each variable is assumed to be:
        - [B, H, W]   (no time dimension), or
        - [B, T, H, W] (time dimension = T)
        We will unify the 'var_count * T' dimension into 'channels'.

        Steps:
        1) Stack variables along a new dim => shape [var_count, B, (T,) H, W].
        2) Permute to [B, var_count, (T,) H, W].
        3) Merge var_count*(T) => channels => shape [B, C, H, W].
        4) Patchify => rearrange to [B, #patches, C*(p1*p2)].
        5) token_embeds => [B, #patches, embed_dim].
        """
        if not variables:
            print(f"\n{group_name}: No variables found")
            return None

        # Stack variables => shape [var_count, B, [T,] H, W].
        # e.g. if 2 variables, each [2, 152, 320], then stacking => [2, 2, 152, 320].
        # or if each is [2, 3, 152, 320], => [2, 2, 3, 152, 320].
        values = list(variables.values())  # list of Tensors
        x = torch.stack(values, dim=0)
        # print(f"{group_name}: after stacking => {x.shape}")

        # Move the 'var_count' dimension after batch dim => [B, var_count, ...].
        # e.g. [2, 2, 152, 320] => [2, 2, 152, 320] if var_count=2 is dimension 0, B=2 dimension 1 => we do a transpose:
        # dims = list(x.shape)
        # dims[0] = var_count, dims[1] = B, ...
        # We want => [B, var_count, ...].
        # so do: x = x.permute(1, 0, 2, 3, ...) if 4D
        if x.dim() == 4:
            # shape => [V, B, H, W]
            x = x.permute(1, 0, 2, 3)  # => [B, V, H, W]
        elif x.dim() == 5:
            # shape => [V, B, T, H, W]
            x = x.permute(1, 0, 2, 3, 4)  # => [B, V, T, H, W]
        else:
            raise ValueError(f"Unsupported shape {x.shape} in {group_name}")

        # print(f"{group_name}: after permute => {x.shape}")

        # Merge    into a single channel dimension => shape [B, C, H, W].
        # if x.dim()==4, => [B, V, H, W]. then C = V
        # if x.dim()==5, => [B, V, T, H, W]. then C = V * T
        if x.dim() == 4:
            B, V, H, W = x.shape
            x = x.reshape(B, V, H, W)  # trivial, no time
            # channel_dim = V
        else:
            # x.dim()==5 => [B, V, T, H, W]
            B, V, T, H, W = x.shape
            x = x.reshape(B, V * T, H, W)
            # channel_dim = V * T
        # print(f"{group_name}: merged var/time => channels => {x.shape} (C={channel_dim})")

        # Now do patchify:
        # We want => [B, (H/p1)*(W/p2), C*(p1*p2)]
        # einops pattern: "b c (h p1) (w p2) -> b (h w) (c p1 p2)"
        # Make sure H,W are multiples of p1,p2
        x = rearrange(x, "b c (h p1) (w p2) -> b (h w) (c p1 p2)", p1=self.patch_size, p2=self.patch_size)
        # print(f"{group_name}: after patchify => {x.shape}")

        # Now x is [B, num_patches, C*(p1*p2)] => feed token_embeds
        # token_embeds expects the last dim = in_features that matches its linear layer
        x = token_embeds(x)  # => [B, num_patches, embed_dim]
        # print(f"{group_name}: after token_embeds => {x.shape}")

        return x

    def forward(self, batch, lead_time, batch_size):
        """
        Forward pass of the encoder.

        Args:
            batch: Batch of data containing various variable groups
            lead_time: Time difference between input and target

        Returns:
            torch.Tensor: Encoded representation
            Shape: [batch_size, num_latents, embed_dim]
        """
        B = batch_size  # the assumption is that we are taking one batch at a time, but that can be changed of course
        H, W = len(batch.batch_metadata.latitudes[0]), len(batch.batch_metadata.longitudes[0])
        # print(f"Encoder H, W {H}, {W}")
        if not hasattr(self, "perceiver_io"):
            self._initialize_perceiver(H, W)

        # process each variable group
        embeddings = []
        embedding_groups = {}
        # print("process surface")
        surface_embed = self.process_variable_group(
            batch.surface_variables, self.surface_token_embeds, "Surface Variables"
        )  # shape: [num_patches, embed_dim]
        if surface_embed is not None:
            embeddings.append(surface_embed)
            embedding_groups["surface"] = surface_embed
        # print("process signle")
        single_embed = self.process_variable_group(
            batch.single_variables, self.single_token_embeds, "Single Variables"
        )  # shape: [num_patches, embed_dim]
        if single_embed is not None:
            embeddings.append(single_embed)
            embedding_groups["single"] = single_embed
        # print("process atmospheric")

        # atmos = []
        if batch.atmospheric_variables:
            for level_idx, level in enumerate(self.atmos_levels):
                # For each variable in atmospheric_variables, slice out dimension=2 (the levels)
                # shape => [v, b, H, W] for that single level.
                level_vars = {}
                for var_name, var_data in batch.atmospheric_variables.items():
                    # var_data is [v, b, l, H, W], but typically v=1 if "var_data" is truly one variable
                    # or if we're grouping multiple variables.
                    # We slice out the level dimension at index 2:
                    sliced = var_data[..., level_idx, :, :]  # shape: [v, b, H, W]
                    level_vars[var_name] = sliced

                # Now pass that dictionary (one level) to the correct embedding
                level_embed = self.process_variable_group(
                    level_vars, self.atmos_token_embeds, f"Atmospheric Level {level}"  # sized for 2 variables, 1 level at a time
                )

                if level_embed is not None:
                    embeddings.append(level_embed)
                    embedding_groups["atmos"] = level_embed
                    # atmos.append(level_embed)
            # TODO Revise the logic why we need to stack them
            # TODO For now we just add them to the list of embeddings.
            # TODO v1
            # If we processed any levels, stack them [L, num_patches, embed_dim]
            # if len(atmos) > 0:
            #     stacked_atmos = torch.stack(atmos, dim=0)  # shape: [num_levels, num_patches, embed_dim]
            # embeddings.append(stacked_atmos)
            # embedding_groups["atmos"] = stacked_atmos

        # print("process species distribution")
        # species_distr = []
        if batch.species_variables:
            for level_idx in range(self.species_num):
                # For each variable in atmospheric_variables, slice out dimension=2 (the levels)
                # shape => [v, b, H, W] for that single level.
                species_num_vars = {}
                for var_name, var_data in batch.species_variables.items():
                    # print(f"var_data_shape {var_data.shape}")
                    # var_data is [v, b, l, H, W], but typically v=1 if "var_data" is truly one variable
                    # or if we're grouping multiple variables.
                    # We slice out the level dimension at index 2:
                    sliced = var_data[..., level_idx, :, :]  # shape: [v, b, H, W]
                    # print(f"var_data shape {var_data.shape} | sliced shape {sliced.shape}")
                    species_num_vars[var_name] = sliced
                # print("species num vars", species_num_vars)
                # print("species token emb", self.species_distr_token_embeds)
                # Now pass that dictionary (one level) to the correct embedding
                species_num_emb = self.process_variable_group(
                    species_num_vars,
                    self.species_distr_token_embeds,  # sized for 2 variables, 1 level at a time
                    f"Species Distribution {level_idx}",
                )
                if species_num_emb is not None:
                    # species_distr.append(species_num_emb)
                    embeddings.append(species_num_emb)
                    embedding_groups["species_distr"] = species_num_emb

        # print("process species extinction")
        species_embed = self.process_variable_group(
            batch.species_extinction_variables, self.species_token_embeds, "Species Variables"
        )  # shape: [num_patches, embed_dim]
        if species_embed is not None:
            embeddings.append(species_embed)
            embedding_groups["species"] = species_embed
        # print("process land")
        land_embed = self.process_variable_group(
            batch.land_variables, self.land_token_embeds, "Land Variables"
        )  # shape: [num_patches, embed_dim]
        if land_embed is not None:
            embeddings.append(land_embed)
            embedding_groups["land"] = land_embed
        # print("process agri")
        agriculture_embed = self.process_variable_group(
            batch.agriculture_variables, self.agriculture_token_embeds, "Agriculture Variables"
        )  # shape: [num_patches, embed_dim]
        if agriculture_embed is not None:
            embeddings.append(agriculture_embed)
            embedding_groups["agriculture"] = agriculture_embed

        forest_embed = self.process_variable_group(
            batch.forest_variables, self.forest_token_embeds, "Forest Variables"
        )  # shape: [num_patches, embed_dim]
        if forest_embed is not None:
            embeddings.append(forest_embed)
            embedding_groups["forest"] = forest_embed

        # Combine embeddings while maintaining group structure
        # x = torch.cat(
        #     [emb.view(1, -1, self.embed_dim) for emb in embeddings], dim=1
        # )  # shape: [batch_size, num_patches * num_variable_groups, embed_dim]
        x = torch.cat(embeddings, dim=1)
        # print("Combined embeddings shape", x.shape)
        x = self._check_tensor(x, "Combined embeddings")

        # add position encodings #squeeze() was working after lat lon
        lat, lon = batch.batch_metadata.latitudes[-1], batch.batch_metadata.longitudes[-1]
        # print(f"encoder lat lon {lat.shape} | {lon.shape} and with squeeze {lat}")
        lat_grid, lon_grid = torch.meshgrid(lat, lon, indexing="ij")
        pos_input = torch.stack((lat_grid, lon_grid), dim=-1).to(x.device)
        # make the positions grid

        # reshape to patches
        H, W = pos_input.shape[:2]
        # print(f"H, W in pos_input {H} | {W}")
        pos_input = pos_input.reshape(H // self.patch_size, self.patch_size, W // self.patch_size, self.patch_size, 2)
        pos_input = pos_input[:, self.patch_size // 2, :, self.patch_size // 2, :]

        # flatten patches before position encoding
        pos_input_flat = pos_input.reshape(-1, 2)  # shape: [num_patches, 2]

        # build position encoder and move to correct device
        pos_encoder = build_position_encoding(
            position_encoding_type=self.position_encoding_type,
            index_dims=(3040,),
            fourier_position_encoding_kwargs={
                "num_bands": self.num_fourier_bands,
                "max_freq": self.max_frequency,
                "concat_pos": True,
                "sine_only": False,
            },
        ).to(x.device)

        # normalize positions to [-1, 1] range
        pos_input_norm = (
            2 * (pos_input_flat - pos_input_flat.min(dim=0)[0]) / (pos_input_flat.max(dim=0)[0] - pos_input_flat.min(dim=0)[0])
            - 1
        )

        # add batch dimension and apply position encoding
        pos_input_norm = pos_input_norm.unsqueeze(0).expand(B, -1, -1)  # [batch_size, num_patches, 2]
        pos_encode = pos_encoder(batch_size=B, pos=pos_input_norm)  # shape: [batch_size, num_patches, pos_encoding_dim]

        # apply linear embedding
        pos_encode = self.pos_embed(pos_encode)  # shape: [num_patches, embed_dim]
        num_var_groups = x.shape[1] // (H // self.patch_size * W // self.patch_size)
        pos_encode = pos_encode.repeat_interleave(
            num_var_groups, dim=1
        )  # shape: [batch_size, num_patches * num_variable_groups, embed_dim]  (middle dimension - rough estimate, since we have atmos levels)
        # add position encoding
        x = x + pos_encode
        x = self._check_tensor(
            x, "After adding position encoding"
        )  # shape: [batch_size, num_patches * num_variable_groups, embed_dim]

        # add time embeddings
        lead_hours = lead_time.total_seconds() / 3600
        lead_times = lead_hours * torch.ones(B, dtype=x.dtype, device=x.device)
        lead_time_encode = self._time_encoding(lead_times)  # shape: [batch_size, embed_dim]
        lead_time_encode = self._check_tensor(lead_time_encode, "Lead time encoding")
        lead_time_emb = self.lead_time_embed(lead_time_encode)  # shape: [batch_size, embed_dim]
        lead_time_emb = self._check_tensor(lead_time_emb, "Lead time embedding")
        x = x + lead_time_emb.unsqueeze(1)  # shape: [batch_size, num_patches * num_variable_groups, embed_dim]
        x = self._check_tensor(x, "After adding time embedding")

        # normalize before Perceiver IO
        x = self.pre_perceiver_norm(x)
        self._check_tensor(x, "Normalized input")

        # Apply Perceiver IO with structured latents
        latents = self.latents.to(x.device)
        # print(f"Encoder forward latents shape: {latents.shape}")
        latents = self._check_tensor(latents, "Latent queries")  # shape: [num_latents, embed_dim]

        # Track latent splits for potential future use
        current_idx = 0
        structured_latents = {}
        for group, size in self.latent_sizes.items():
            if size > 0:
                structured_latents[group] = latents[current_idx : current_idx + size]
                current_idx += size

        latents = latents.unsqueeze(0).expand(B, -1, -1)  # shape: [batch_size, num_latents, embed_dim]
        x = self.perceiver_io(x, queries=latents)
        x = self._check_tensor(x, "After Perceiver IO")  # shape: [batch_size, num_latents, embed_dim]

        x = self.pos_drop(x)
        x = self._check_tensor(x, "Final output")  # shape: [batch_size, num_latents, embed_dim]

        return x

    def _time_encoding(self, times):
        """
        Create time encoding using sinusoidal embeddings.

        Args:
            times (torch.Tensor): Time values to encode
            Shape: [batch_size] or [batch_size, sequence_length]

        Returns:
            torch.Tensor: Time encodings
            Shape: [batch_size, embed_dim] or [batch_size, sequence_length, embed_dim]
        """
        device = times.device
        half_dim = self.embed_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)

        # handle 2D input
        if times.dim() == 2:
            emb = times.unsqueeze(-1) * emb.unsqueeze(0).unsqueeze(0)
        else:
            emb = times.unsqueeze(-1) * emb.unsqueeze(0)

        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


# def main():
#     from datetime import datetime, timedelta

#     import torch.cuda as cuda

#     # set device
#     device = torch.device("cuda:2" if cuda.is_available() else "cpu")
#     print(f"\nUsing device: {device}")

#     # load batches from data
#     print("\nLoading batches...")
#     batches = load_batches("data/", device=device)
#     print(f"Loaded {len(batches)} batches")

#     # get first batch
#     batch = batches[0]

#     # crop dimensions to be divisible by patch size
#     patch_size = 4
#     H, W = batch.batch_metadata.latitudes.shape[0], batch.batch_metadata.longitudes.shape[0]
#     new_H = (H // patch_size) * patch_size
#     new_W = (W // patch_size) * patch_size

#     def crop_variables(variables):
#         processed_vars = {}
#         for k, v in variables.items():
#             # crop dimensions
#             cropped = v[..., :new_H, :new_W]

#             # handle infinities and NaNs
#             inf_mask = torch.isinf(cropped)
#             nan_mask = torch.isnan(cropped)

#             # fix issues!
#             valid_values = cropped[~inf_mask & ~nan_mask]
#             if len(valid_values) > 0:
#                 mean_val = valid_values.mean().item()
#                 cropped = torch.nan_to_num(cropped, nan=mean_val, posinf=1e6, neginf=-1e6)
#             else:
#                 print(f"Warning: No valid values found in {k}")
#                 cropped = torch.zeros_like(cropped)

#             processed_vars[k] = cropped
#         return processed_vars

#     print("\nProcessing batches...")
#     for i, batch in enumerate(batches):
#         print(f"\nProcessing batch {i+1}/{len(batches)}")
#         batch.surface_variables = crop_variables(batch.surface_variables)
#         batch.single_variables = crop_variables(batch.single_variables)
#         batch.atmospheric_variables = crop_variables(batch.atmospheric_variables)
#         batch.species_extinction_variables = crop_variables(batch.species_extinction_variables)
#         batch.land_variables = crop_variables(batch.land_variables)
#         batch.agriculture_variables = crop_variables(batch.agriculture_variables)
#         batch.forest_variables = crop_variables(batch.forest_variables)

#         # crop metadata dimensions
#         batch.batch_metadata.latitudes = batch.batch_metadata.latitudes[:new_H]
#         batch.batch_metadata.longitudes = batch.batch_metadata.longitudes[:new_W]

#     # init encoder
#     print("\nInitializing encoder...")
#     encoder = BFMEncoder(
#         surface_vars=tuple(batch.surface_variables.keys()),
#         single_vars=tuple(batch.single_variables.keys()),
#         atmos_vars=tuple(batch.atmospheric_variables.keys()),
#         species_vars=tuple(batch.species_extinction_variables.keys()),
#         land_vars=tuple(batch.land_variables.keys()),
#         agriculture_vars=tuple(batch.agriculture_variables.keys()),
#         forest_vars=tuple(batch.forest_variables.keys()),
#         atmos_levels=batch.batch_metadata.pressure_levels,
#         patch_size=patch_size,
#         H=new_H,
#         W=new_W,
#     ).to(device)

#     # process each batch
#     print("\nRunning forward pass on batches...")
#     for i, batch in enumerate(batches):
#         print(f"\nProcessing batch {i+1}/{len(batches)}")

#         # find lead time
#         t1, t2 = batch.batch_metadata.timestamp
#         lead_time = t2 - t1

#         try:
#             with torch.no_grad():
#                 output = encoder(batch, lead_time)
#                 print(f"Successfully processed batch {i+1}, shape: {output.shape}")

#         except Exception as e:
#             print(f"\nError processing batch {i+1}:")
#             print(f"Error message: {str(e)}")
#             raise e


# if __name__ == "__main__":
#     main()
