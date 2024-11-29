"""
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

Example usage:
    decoder = BFMDecoder(
        surface_vars=('temperature', 'pressure'),
        single_vars=('humidity',),
        atmos_vars=('wind_u', 'wind_v'),
        species_vars=('species_1', 'species_2'),
        land_vars=('soil_moisture',),
        agriculture_vars=('crop_yield',),
        forest_vars=('tree_coverage',),
        atmos_levels=[1000, 850, 700],
        patch_size=4,
        embed_dim=128,
        H=32,
        W=64
    )
    output = decoder(encoded_data, batch, lead_time)
"""

import math
from collections import namedtuple
from datetime import datetime, timedelta

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from src.bfm.src.dataset_basics import load_batches
from src.perceiver_components.helpers_io import dropout_seq
from src.perceiver_components.pos_encoder import build_position_encoding
from src.perceiver_core.perceiver_io import PerceiverIO


class BFMDecoder(nn.Module):
    """
    Biodiversity Foundation Model Decoder.

    This decoder takes encoded representations and transforms them back into interpretable
    climate and biosphere variables using a Perceiver IO architecture.

    Args:
        surface_vars (tuple[str, ...]): Names of surface-level variables
        single_vars (tuple[str, ...]): Names of single-level variables
        atmos_vars (tuple[str, ...]): Names of atmospheric variables
        species_vars (tuple[str, ...]): Names of species-related variables
        land_vars (tuple[str, ...]): Names of land-related variables
        agriculture_vars (tuple[str, ...]): Names of agriculture-related variables
        forest_vars (tuple[str, ...]): Names of forest-related variables
        atmos_levels (list[int]): Pressure levels for atmospheric variables
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
        land_token_proj (nn.Linear): Projection layer for land variables
        agriculture_token_proj (nn.Linear): Projection layer for agriculture variables
        forest_token_proj (nn.Linear): Projection layer for forest variables
        perceiver_io (PerceiverIO): Main Perceiver IO model
        pos_drop (nn.Dropout): Position dropout layer
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
        patch_size: int = 4,
        embed_dim: int = 1024,
        num_heads: int = 16,
        head_dim: int = 64,
        drop_rate: float = 0.1,
        depth: int = 2,
        mlp_ratio: float = 4.0,
        perceiver_ln_eps: float = 1e-5,
        num_fourier_bands: int = 64,
        max_frequency: float = 224.0,
        num_input_axes: int = 1,
        position_encoding_type: str = "fourier",
        H: int = 32,
        W: int = 64,
    ):
        super().__init__()
        # Store variable names
        self.surface_vars = surface_vars
        self.single_vars = single_vars
        self.atmos_vars = atmos_vars
        self.species_vars = species_vars
        self.land_vars = land_vars
        self.agriculture_vars = agriculture_vars
        self.forest_vars = forest_vars
        self.atmos_levels = atmos_levels

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
            "single": {v: i for i, v in enumerate(single_vars)},
            "atmos": {v: i for i, v in enumerate(atmos_vars)},
            "species": {v: i for i, v in enumerate(species_vars)},
            "land": {v: i for i, v in enumerate(land_vars)},
            "agriculture": {v: i for i, v in enumerate(agriculture_vars)},
            "forest": {v: i for i, v in enumerate(forest_vars)},
        }

        pos_encoding_dim = self._calculate_pos_encoding_dim()

        # pos and time embeddings
        self.pos_embed = nn.Linear(pos_encoding_dim + 2, embed_dim)  # +2 for lat and lon
        self.lead_time_embed = nn.Linear(embed_dim, embed_dim)
        self.absolute_time_embed = nn.Linear(embed_dim, embed_dim)

        # token projections for each variable type
        self.surface_token_proj = nn.Linear(embed_dim, H * W)
        self.single_token_proj = nn.Linear(embed_dim, H * W)
        self.atmos_token_proj = nn.Linear(embed_dim, H * W)
        self.species_token_proj = nn.Linear(embed_dim, H * W)
        self.land_token_proj = nn.Linear(embed_dim, H * W)
        self.agriculture_token_proj = nn.Linear(embed_dim, H * W)
        self.forest_token_proj = nn.Linear(embed_dim, H * W)

        # total number of tokens needed for all variables
        total_tokens = (
            len(surface_vars)
            + len(single_vars)
            + len(atmos_vars) * len(atmos_levels)
            + len(species_vars)
            + len(land_vars)
            + len(agriculture_vars)
            + len(forest_vars)
        )

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

        # Modify this part to handle 2D input
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
            dict: Dictionary containing decoded outputs for each variable category:
                - surface_variables
                - single_variables
                - atmospheric_variables
                - species_extinction_variables
                - land_variables
                - agriculture_variables
                - forest_variables
        """
        B, L, D = x.shape
        print(f"Input shape: {x.shape}")

        # get dimensions from batch metadata
        H, W = self.H, self.W
        lat, lon = batch.batch_metadata.latitudes, batch.batch_metadata.longitudes
        print(f"Grid dimensions (H×W): {H}×{W}")
        print(f"lat shape: {lat.shape}, lon shape: {lon.shape}")

        # counting the number of queries for Perceiver IO
        total_queries = (
            len(self.surface_vars)
            + len(self.single_vars)
            + len(self.atmos_vars) * len(self.atmos_levels)
            + len(self.species_vars)
            + len(self.land_vars)
            + len(self.agriculture_vars)
            + len(self.forest_vars)
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
        lead_hours = lead_time.total_seconds() / 3600
        lead_times = lead_hours * torch.ones(B, dtype=x.dtype, device=x.device)
        lead_time_encode = self._time_encoding(lead_times)
        lead_time_emb = self.lead_time_embed(lead_time_encode)
        queries = queries + lead_time_emb.unsqueeze(1)

        # Add absolute time embedding
        absolute_times = torch.tensor(
            [[t.timestamp() / 3600 for t in time_list] for time_list in [batch.batch_metadata.timestamp]],
            dtype=torch.float32,
            device=x.device,
        )
        absolute_time_encode = self._time_encoding(absolute_times)
        absolute_time_embed = self.absolute_time_embed(absolute_time_encode)
        absolute_time_embed = absolute_time_embed.mean(dim=1, keepdim=True)
        queries = queries + absolute_time_embed

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

        # single variables
        if len(self.single_vars) > 0:
            next_idx = current_idx + len(self.single_vars)
            single_decoded = decoded[:, current_idx:next_idx]
            single_output = self.single_token_proj(single_decoded)
            single_output = single_output.view(B, len(self.single_vars), H, W)
            output["single_vars"] = {var: single_output[:, i] for i, var in enumerate(self.single_vars)}
            current_idx = next_idx

        # atmospheric variables
        if len(self.atmos_vars) > 0:
            next_idx = current_idx + len(self.atmos_vars) * len(self.atmos_levels)
            atmos_decoded = decoded[:, current_idx:next_idx]
            atmos_output = self.atmos_token_proj(atmos_decoded)
            atmos_output = atmos_output.view(B, len(self.atmos_vars), len(self.atmos_levels), H, W)
            output["atmos_vars"] = {var: atmos_output[:, i] for i, var in enumerate(self.atmos_vars)}
            current_idx = next_idx

        # species variables
        if len(self.species_vars) > 0:
            next_idx = current_idx + len(self.species_vars)
            species_decoded = decoded[:, current_idx:next_idx]
            species_output = self.species_token_proj(species_decoded)
            species_output = species_output.view(B, len(self.species_vars), H, W)
            output["species_vars"] = {var: species_output[:, i] for i, var in enumerate(self.species_vars)}
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

        output = {
            "surface_variables": output.pop("surface_vars"),
            "single_variables": output.pop("single_vars"),
            "atmospheric_variables": output.pop("atmos_vars"),
            "species_extinction_variables": output.pop("species_vars"),
            "land_variables": output.pop("land_vars"),
            "agriculture_variables": output.pop("agriculture_vars"),
            "forest_variables": output.pop("forest_vars"),
        }
        return output


def main():
    """Main function for testing the BFM decoder implementation."""
    import torch.cuda as cuda

    from src.bfm.src.encoder import BFMEncoder

    device = torch.device("cuda:2" if cuda.is_available() else "cpu")
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
    H, W = batch.batch_metadata.latitudes.shape[0], batch.batch_metadata.longitudes.shape[0]
    new_H = (H // patch_size) * patch_size
    new_W = (W // patch_size) * patch_size

    print(f"\nOriginal spatial dimensions: {H}×{W}")
    print(f"Cropped spatial dimensions: {new_H}×{new_W}")

    def crop_variables(variables):
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

    print("\nProcessing batches...")
    for i, batch in enumerate(batches):
        print(f"\nProcessing batch {i+1}/{len(batches)}")
        batch.surface_variables = crop_variables(batch.surface_variables)
        batch.single_variables = crop_variables(batch.single_variables)
        batch.atmospheric_variables = crop_variables(batch.atmospheric_variables)
        batch.species_extinction_variables = crop_variables(batch.species_extinction_variables)
        batch.land_variables = crop_variables(batch.land_variables)
        batch.agriculture_variables = crop_variables(batch.agriculture_variables)
        batch.forest_variables = crop_variables(batch.forest_variables)

        # crop metadata dimensions
        batch.batch_metadata.latitudes = batch.batch_metadata.latitudes[:new_H]
        batch.batch_metadata.longitudes = batch.batch_metadata.longitudes[:new_W]

    print("\nSpatial dimensions after cropping:")
    print(f"Grid size: {new_H}×{new_W}")
    print(f"Number of patches: {new_H//patch_size}×{new_W//patch_size}")

    # init encoder and decoder
    print("\nInitializing encoder and decoder...")
    encoder = BFMEncoder(
        surface_vars=tuple(batch.surface_variables.keys()),
        single_vars=tuple(batch.single_variables.keys()),
        atmos_vars=tuple(batch.atmospheric_variables.keys()),
        species_vars=tuple(batch.species_extinction_variables.keys()),
        land_vars=tuple(batch.land_variables.keys()),
        agriculture_vars=tuple(batch.agriculture_variables.keys()),
        forest_vars=tuple(batch.forest_variables.keys()),
        atmos_levels=batch.batch_metadata.pressure_levels,
        patch_size=patch_size,
    ).to(device)

    decoder = BFMDecoder(
        surface_vars=tuple(batch.surface_variables.keys()),
        single_vars=tuple(batch.single_variables.keys()),
        atmos_vars=tuple(batch.atmospheric_variables.keys()),
        species_vars=tuple(batch.species_extinction_variables.keys()),
        land_vars=tuple(batch.land_variables.keys()),
        agriculture_vars=tuple(batch.agriculture_variables.keys()),
        forest_vars=tuple(batch.forest_variables.keys()),
        atmos_levels=batch.batch_metadata.pressure_levels,
        patch_size=patch_size,
        embed_dim=128,
        H=new_H,
        W=new_W,
    ).to(device)

    print("\nModel architectures initialized")

    for i, batch in enumerate(batches):
        print(f"\nProcessing batch {i+1}/{len(batches)}")

        t1, t2 = batch.batch_metadata.timestamp
        lead_time = t2 - t1
        try:
            # pass through encoder and decoder
            with torch.no_grad():
                # we encode first
                encoded = encoder(batch, lead_time)
                # and you guessed right - decode afterwards
                decoded = decoder(encoded, batch, lead_time)  # noqa

                # some output statistics for each variable type (as extra horror, you're welcome)
                # print("\nDecoder output statistics:")
                # for var_type, vars_dict in decoded.items():
                #     print(f"\n{var_type}:")
                #     for var_name, var_tensor in vars_dict.items():
                #         print(f"  {var_name}:")
                #         print(f"    Shape: {var_tensor.shape}")
                #         print(f"    Mean: {var_tensor.mean().item():.4f}")
                #         print(f"    Std: {var_tensor.std().item():.4f}")
                #         print(f"    Min: {var_tensor.min().item():.4f}")
                #         print(f"    Max: {var_tensor.max().item():.4f}")

        except Exception as e:
            print(f"\nError processing batch {i+1}:")
            print(f"Error type: {type(e).__name__}")
            print(f"Error message: {str(e)}")
            raise


if __name__ == "__main__":
    main()
