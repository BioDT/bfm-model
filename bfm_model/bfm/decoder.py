"""
Copyright 2025 (C) TNO. Licensed under the MIT license.

BFM (Biodiversity Foundation Model) Decoder Module.

This module contains the decoder component of the BFM architecture, responsible for transforming
encoded latent representations back into interpretable climate and biosphere variables.

The decoder uses a Perceiver IO architecture to process queries for different variable types
and outputs predictions in a spatially-organized format.

Key Components:
    - Position encoding using Fourier features
    - Time encoding for both lead time and absolute time
    - Variable-specific token projections
    - Perceiver IO for flexible decoding
    - Multi-category variable handling (surface, atmospheric, species, etc.)
"""

import math
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F

from bfm_model.perceiver_components.pos_encoder import build_position_encoding
from bfm_model.perceiver_core.perceiver_io import PerceiverIO


class BFMDecoder(nn.Module):
    """
    Biodiversity Foundation Model Decoder.

    This decoder takes encoded representations and transforms them back into interpretable
    climate and biosphere variables using a Perceiver IO architecture.

    Args:
        surface_vars (tuple[str, ...]): Names of surface-level variables
        edaphic_vars (tuple[str, ...]): Names of edaphic-related variables
        atmos_vars (tuple[str, ...]): Names of atmospheric-related variables
        climate_vars (tuple[str, ...]): Names of climate-related variables
        species_vars (tuple[str, ...]): Names of species variables
        vegetation_vars (tuple[str, ...]): Names of vegetation-related variables
        land_vars (tuple[str, ...]): Names of land-related variables
        agriculture_vars (tuple[str, ...]): Names of agriculture-related variables
        forest_vars (tuple[str, ...]): Names of forest-related variables
        redlist_vars (tuple[str, ...]): Names of red list-related variables
        misc_vars (tuple[str, ...]): Names of miscellaneous variables
        atmos_levels (list[int]): Pressure levels for atmospheric variables
        species_num (int): Number of species distribution to account for
        patch_size (int, optional): Size of spatial patches. Defaults to 4.
        embed_dim (int, optional): Embedding dimension. Defaults to 1024.
        num_heads (int, optional): Number of attention heads. Defaults to 16.
        head_dim (int, optional): Dimension of each attention head. Defaults to 64.
        drop_rate (float, optional): Dropout rate. Defaults to 0.1.
        depth (int, optional): Number of transformer layers. Defaults to 2.
        mlp_ratio (float, optional): Ratio for MLP hidden dimension. Defaults to 4.0.
        perceiver_ln_eps (float, optional): Layer norm epsilon. Defaults to 1e-5.
        num_fourier_bands (int, optional): Number of Fourier bands for position encoding. Defaults to 64.
        max_frequency (float, optional): Maximum frequency for Fourier features. Defaults to 224.0.
        num_input_axes (int, optional): Number of input axes. Defaults to 1.
        position_encoding_type (str, optional): Type of position encoding. Defaults to "fourier".
        H (int, optional): Height of output grid. Defaults to 32.
        W (int, optional): Width of output grid. Defaults to 64.

    Attributes:
        var_maps (dict): Mappings from variable names to indices for each category
        pos_embed (nn.Linear): Position embedding layer
        lead_time_embed (nn.Linear): Lead time embedding layer
        absolute_time_embed (nn.Linear): Absolute time embedding layer
        surface_token_proj (nn.Linear): Projection layer for surface variables
        single_token_proj (nn.Linear): Projection layer for single variables
        atmos_token_proj (nn.Linear): Projection layer for atmospheric variables
        species_token_proj (nn.Linear): Projection layer for species variables
        species_distr_token_proj (nn.Linear): Projection layer for species distributions variables
        land_token_proj (nn.Linear): Projection layer for land variables
        agriculture_token_proj (nn.Linear): Projection layer for agriculture variables
        forest_token_proj (nn.Linear): Projection layer for forest variables
        perceiver_io (PerceiverIO): Main Perceiver IO model
        pos_drop (nn.Dropout): Position dropout layer
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
        patch_size: int = 4,
        embed_dim: int = 1024,
        num_heads: int = 16,
        kv_heads: int = 8,
        head_dim: int = 2,
        drop_rate: float = 0.1,
        depth: int = 2,
        mlp_ratio: float = 4.0,
        perceiver_ln_eps: float = 1e-5,
        num_fourier_bands: int = 64,
        max_frequency: float = 224.0,
        num_input_axes: int = 1,
        position_encoding_type: str = "fourier",
        H: int = 152,
        W: int = 320,
    ):
        super().__init__()
        # Store variable names
        self.surface_vars = surface_vars
        self.edaphic_vars = edaphic_vars
        self.atmos_vars = atmos_vars
        self.climate_vars = climate_vars
        self.species_vars = species_vars
        self.vegetation_vars = vegetation_vars
        self.land_vars = land_vars
        self.agriculture_vars = agriculture_vars
        self.forest_vars = forest_vars
        self.redlist_vars = redlist_vars
        self.misc_vars = misc_vars
        self.atmos_levels = atmos_levels
        self.species_num = species_num

        # Basic configuration
        self.drop_rate = drop_rate
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.num_input_axes = num_input_axes
        self.max_frequency = max_frequency
        self.num_fourier_bands = num_fourier_bands
        self.position_encoding_type = position_encoding_type
        self.H = H
        self.W = W

        # Create variable mappings for each category
        self.var_maps = {
            "surface": {v: i for i, v in enumerate(surface_vars)},
            "edaphic": {v: i for i, v in enumerate(edaphic_vars)},
            "atmos": {v: i for i, v in enumerate(atmos_vars)},
            "climate": {v: i for i, v in enumerate(climate_vars)},
            "species": {v: i for i, v in enumerate(species_vars)},
            "species_distr": {v: i for i, v in enumerate(vegetation_vars)},
            "land": {v: i for i, v in enumerate(land_vars)},
            "agriculture": {v: i for i, v in enumerate(agriculture_vars)},
            "forest": {v: i for i, v in enumerate(forest_vars)},
            "redlist": {v: i for i, v in enumerate(redlist_vars)},
            "misc": {v: i for i, v in enumerate(misc_vars)},
        }

        pos_encoding_dim = self._calculate_pos_encoding_dim()

        # pos and time embeddings
        self.pos_embed = nn.Linear(pos_encoding_dim + 2, embed_dim)  # +2 for lat and lon
        self.lead_time_embed = nn.Linear(embed_dim, embed_dim)
        self.absolute_time_embed = nn.Linear(embed_dim, embed_dim)

        # token projections for each variable type
        self.surface_token_proj = nn.Linear(embed_dim, H * W)
        self.edaphic_token_proj = nn.Linear(embed_dim, H * W)
        self.atmos_token_proj = nn.Linear(embed_dim, H * W)
        self.climate_token_proj = nn.Linear(embed_dim, H * W)
        self.species_token_proj = nn.Linear(embed_dim, H * W)
        self.vegetation_token_proj = nn.Linear(embed_dim, H * W)
        self.land_token_proj = nn.Linear(embed_dim, H * W)
        self.agriculture_token_proj = nn.Linear(embed_dim, H * W)
        self.forest_token_proj = nn.Linear(embed_dim, H * W)
        self.redlist_token_proj = nn.Linear(embed_dim, H * W)
        self.misc_token_proj = nn.Linear(embed_dim, H * W)

        # total number of tokens needed for all variables
        total_tokens = (
            len(surface_vars)
            + len(edaphic_vars)
            + len(atmos_vars) * len(atmos_levels)
            + len(climate_vars)
            + len(species_vars)
            + len(vegetation_vars)
            + len(land_vars)
            + len(agriculture_vars)
            + len(forest_vars)
            + len(redlist_vars)
            + len(misc_vars)
        )
        print("Total query tokens for Decoder: ", total_tokens)

        self.perceiver_io = PerceiverIO(
            num_layers=depth,
            dim=embed_dim,
            queries_dim=embed_dim,
            logits_dimension=None,
            num_latent_tokens=total_tokens,
            latent_dimension=embed_dim,
            cross_attention_heads=num_heads,
            latent_attention_heads=num_heads,
            cross_attention_head_dim=head_dim,
            latent_attention_head_dim=head_dim,
            num_kv_heads=kv_heads,
            num_fourier_bands=num_fourier_bands,
            max_frequency=max_frequency,
            num_input_axes=num_input_axes,
            position_encoding_type=position_encoding_type,
        )

        self.pos_drop = nn.Dropout(p=drop_rate)
        self.apply(self._init_weights)

    def _calculate_pos_encoding_dim(self):
        """Calculate dimension of position encoding."""
        return 2 * self.num_fourier_bands + 1

    def _init_weights(self, m):
        """Initialize weights for linear and layer norm layers.

        Args:
            m (nn.Module): Module to initialize
        """
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _build_position_encoding(self, shape):
        """Build position encoding based on specified type.

        Args:
            shape (tuple): Shape of input tensor

        Returns:
            callable: Position encoding function
        """
        if self.position_encoding_type is None:
            return None
        return build_position_encoding(
            position_encoding_type=self.position_encoding_type,
            index_dims=shape[1:-1],
            fourier_position_encoding_kwargs={
                "num_bands": self.num_fourier_bands,
                "max_freq": self.max_frequency,
                "concat_pos": True,
                "sine_only": False,
            },
        )

    def _apply_position_encoding(self, input_data):
        """Apply position encoding to input data.

        Args:
            input_data (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Position-encoded input tensor
        """
        batch_size, *_, device = *input_data.shape, input_data.device
        if self.position_encoding_type in ["fourier", "trainable"]:
            pos_encoder = self._build_position_encoding(input_data.shape)
            pos_encoding = pos_encoder(batch_size=batch_size).to(device)
            input_data = torch.cat((input_data, pos_encoding), dim=-1)
        return input_data

    def _time_encoding(self, times):
        """Generate time encoding using sinusoidal features.

        Args:
            times (torch.Tensor): Time values to encode

        Returns:
            torch.Tensor: Time encoding
        """
        device = times.device
        half_dim = self.embed_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)

        if times.dim() == 2:
            emb = times.unsqueeze(-1) * emb.unsqueeze(0).unsqueeze(0)
        else:
            emb = times.unsqueeze(-1) * emb.unsqueeze(0)

        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

    def forward(self, x, batch, lead_time):
        """
        Forward pass of the decoder.

        Args:
            x (torch.Tensor): Encoded input tensor of shape (batch_size, sequence_length, dimension)
            batch: Batch object containing metadata and variables
            lead_time (timedelta): Time difference between input and target

        Returns:
            dict: Dictionary containing decoded outputs for each variable category
        """
        B, L, D = x.shape
        # get dimensions from batch metadata
        H, W = self.H, self.W
        # batch_metadata come with shape [batch_size, lat/lon_dim]
        lat, lon = batch.batch_metadata.latitudes[-1], batch.batch_metadata.longitudes[-1]

        # counting the number of queries for Perceiver IO
        total_queries = (
            len(self.surface_vars)
            + len(self.edaphic_vars)
            + len(self.atmos_vars) * len(self.atmos_levels)
            + len(self.climate_vars)
            + len(self.species_vars)
            + len(self.vegetation_vars)
            + len(self.land_vars)
            + len(self.agriculture_vars)
            + len(self.forest_vars)
            + len(self.redlist_vars)
            + len(self.misc_vars)
        )
        # the queries used to ask Perceiver IO for values of all variables (the main reason of the decoder flexibility in processing the embeddings lies in these)
        queries = torch.zeros(B, total_queries, D, device=x.device)

        # 2D grid of lat and lon
        lat_grid, lon_grid = torch.meshgrid(lat, lon, indexing="ij")
        pos_input = torch.stack((lat_grid, lon_grid), dim=-1)

        # add position encoding
        pos_encode = self._apply_position_encoding(pos_input)
        pos_encode = self.pos_embed(pos_encode)
        pos_encode = pos_encode.view(1, H * W, -1)
        pos_encode = F.interpolate(pos_encode.transpose(1, 2), size=total_queries, mode="linear", align_corners=False).transpose(
            1, 2
        )

        queries = queries + pos_encode

        # Add lead time embedding
        # lead_hours = lead_time.total_seconds() / 3600 # for hourly
        lead_times = lead_time * torch.ones(B, dtype=x.dtype, device=x.device)
        lead_time_encode = self._time_encoding(lead_times)
        lead_time_emb = self.lead_time_embed(lead_time_encode)
        queries = queries + lead_time_emb.unsqueeze(1)

        # Add absolute time embedding
        if hasattr(self, "absolute_time_embed") and self.absolute_time_embed is not None:
            start_timestamp_str = batch.batch_metadata.timestamp[0][0]
            dt_format = "%Y-%m-%d %H:%M:%S"
            start_dt = datetime.strptime(start_timestamp_str, dt_format)

            abs_time_numerical_values = []
            for _ in range(B):
                abs_time_numerical_values.append(start_dt.month)

            abs_times_tensor = torch.tensor(abs_time_numerical_values, dtype=x.dtype, device=x.device)

            absolute_time_encode = self._time_encoding(abs_times_tensor)
            absolute_time_emb_vec = self.absolute_time_embed(absolute_time_encode)
            queries = queries + absolute_time_emb_vec.unsqueeze(1)

        # Apply Perceiver IO
        decoded = self.perceiver_io(x, queries=queries)

        # Split decoded output for different variable types
        current_idx = 0
        output = {}

        # surface variables
        if len(self.surface_vars) > 0:
            next_idx = current_idx + len(self.surface_vars)
            surf_decoded = decoded[:, current_idx:next_idx]
            surf_output = self.surface_token_proj(surf_decoded)
            surf_output = surf_output.view(B, len(self.surface_vars), H, W)
            output["surface_vars"] = {var: surf_output[:, i] for i, var in enumerate(self.surface_vars)}
            current_idx = next_idx

        # edaphic variables
        if len(self.edaphic_vars) > 0:
            next_idx = current_idx + len(self.edaphic_vars)
            edaphic_decoded = decoded[:, current_idx:next_idx]
            edaphic_output = self.edaphic_token_proj(edaphic_decoded)
            edaphic_output = edaphic_output.view(B, len(self.edaphic_vars), H, W)
            output["edaphic_vars"] = {var: edaphic_output[:, i] for i, var in enumerate(self.edaphic_vars)}
            current_idx = next_idx

        # atmospheric variables
        if len(self.atmos_vars) > 0:
            next_idx = current_idx + len(self.atmos_vars) * len(self.atmos_levels)
            atmos_decoded = decoded[:, current_idx:next_idx]
            atmos_output = self.atmos_token_proj(atmos_decoded)
            atmos_output = atmos_output.view(B, len(self.atmos_vars), len(self.atmos_levels), H, W)
            output["atmos_vars"] = {var: atmos_output[:, i] for i, var in enumerate(self.atmos_vars)}
            current_idx = next_idx

        # climate variables
        if len(self.climate_vars) > 0:
            next_idx = current_idx + len(self.climate_vars)
            climate_decoded = decoded[:, current_idx:next_idx]
            climate_output = self.climate_token_proj(climate_decoded)
            climate_output = climate_output.view(B, len(self.climate_vars), H, W)
            output["climate_vars"] = {var: climate_output[:, i] for i, var in enumerate(self.climate_vars)}
            current_idx = next_idx

        # species variables
        if len(self.species_vars) > 0:
            next_idx = current_idx + len(self.species_vars)
            species_decoded = decoded[:, current_idx:next_idx]
            species_output = self.species_token_proj(species_decoded)
            species_output = species_output.view(B, len(self.species_vars), H, W)
            output["species_vars"] = {var: species_output[:, i] for i, var in enumerate(self.species_vars)}
            current_idx = next_idx

        # vegetation variables
        if len(self.vegetation_vars) > 0:
            next_idx = current_idx + len(self.vegetation_vars)
            vegetation_decoded = decoded[:, current_idx:next_idx]
            vegetation_output = self.vegetation_token_proj(vegetation_decoded)
            vegetation_output = vegetation_output.view(B, len(self.vegetation_vars), H, W)
            output["vegetation_vars"] = {var: vegetation_output[:, i] for i, var in enumerate(self.vegetation_vars)}
            current_idx = next_idx

        # land variables
        if len(self.land_vars) > 0:
            next_idx = current_idx + len(self.land_vars)
            land_decoded = decoded[:, current_idx:next_idx]
            land_output = self.land_token_proj(land_decoded)
            land_output = land_output.view(B, len(self.land_vars), H, W)
            output["land_vars"] = {var: land_output[:, i] for i, var in enumerate(self.land_vars)}
            current_idx = next_idx

        # agriculture variables
        if len(self.agriculture_vars) > 0:
            next_idx = current_idx + len(self.agriculture_vars)
            agri_decoded = decoded[:, current_idx:next_idx]
            agri_output = self.agriculture_token_proj(agri_decoded)
            agri_output = agri_output.view(B, len(self.agriculture_vars), H, W)
            output["agriculture_vars"] = {var: agri_output[:, i] for i, var in enumerate(self.agriculture_vars)}
            current_idx = next_idx

        # forest variables
        if len(self.forest_vars) > 0:
            next_idx = current_idx + len(self.forest_vars)
            forest_decoded = decoded[:, current_idx:next_idx]
            forest_output = self.forest_token_proj(forest_decoded)
            forest_output = forest_output.view(B, len(self.forest_vars), H, W)
            output["forest_vars"] = {var: forest_output[:, i] for i, var in enumerate(self.forest_vars)}

        # redlist variables
        if len(self.redlist_vars) > 0:
            next_idx = current_idx + len(self.redlist_vars)
            redlist_decoded = decoded[:, current_idx:next_idx]
            redlist_output = self.redlist_token_proj(redlist_decoded)
            redlist_output = redlist_output.view(B, len(self.redlist_vars), H, W)
            output["redlist_vars"] = {var: redlist_output[:, i] for i, var in enumerate(self.redlist_vars)}

        # misc variables
        if len(self.misc_vars) > 0:
            next_idx = current_idx + len(self.misc_vars)
            misc_decoded = decoded[:, current_idx:next_idx]
            misc_output = self.misc_token_proj(misc_decoded)
            misc_output = misc_output.view(B, len(self.misc_vars), H, W)
            output["misc_vars"] = {var: misc_output[:, i] for i, var in enumerate(self.misc_vars)}

        output = {
            "surface_variables": output.pop("surface_vars"),
            "edaphic_variables": output.pop("edaphic_vars"),
            "atmospheric_variables": output.pop("atmos_vars"),
            "climate_variables": output.pop("climate_vars"),
            "species_variables": output.pop("species_vars"),
            "vegetation_variables": output.pop("vegetation_vars"),
            "land_variables": output.pop("land_vars"),
            "agriculture_variables": output.pop("agriculture_vars"),
            "forest_variables": output.pop("forest_vars"),
            "redlist_variables": output.pop("redlist_vars"),
            "misc_variables": output.pop("misc_vars"),
        }
        return output
