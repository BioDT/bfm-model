import math
from collections import namedtuple
from datetime import datetime, timedelta

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from src.perceiver_components.helpers_io import dropout_seq
from src.perceiver_components.pos_encoder import build_position_encoding
from src.perceiver_core.perceiver_io import PerceiverIO


class BFMDecoder(nn.Module):
    def __init__(
        self,
        surf_vars: tuple[str, ...],
        atmos_vars: tuple[str, ...],
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
        self.surf_vars = surf_vars
        self.atmos_vars = atmos_vars
        self.atmos_levels = atmos_levels
        self.drop_rate = drop_rate
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.num_input_axes = num_input_axes
        self.max_frequency = max_frequency
        self.num_fourier_bands = num_fourier_bands
        self.position_encoding_type = position_encoding_type
        self.H = H
        self.W = W

        pos_encoding_dim = self._calculate_pos_encoding_dim()
        # print(f"Calculated pos_encoding_dim: {pos_encoding_dim}")

        self.pos_embed = nn.Linear(pos_encoding_dim + 2, embed_dim)  # +2 for lat and lon
        self.lead_time_embed = nn.Linear(embed_dim, embed_dim)
        self.absolute_time_embed = nn.Linear(embed_dim, embed_dim)

        self.surf_token_proj = nn.Linear(embed_dim, H * W)
        self.atmos_token_proj = nn.Linear(embed_dim, H * W)

        self.perceiver_io = PerceiverIO(
            num_layers=depth,
            dim=embed_dim,
            queries_dim=embed_dim,
            logits_dimension=None,
            num_latent_tokens=len(surf_vars) + len(atmos_vars) * len(atmos_levels),
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
        return 2 * self.num_fourier_bands + 1

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _build_position_encoding(self, shape):
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
        batch_size, *_, device = *input_data.shape, input_data.device
        if self.position_encoding_type in ["fourier", "trainable"]:
            pos_encoder = self._build_position_encoding(input_data.shape)
            pos_encoding = pos_encoder(batch_size=batch_size).to(device)
            input_data = torch.cat((input_data, pos_encoding), dim=-1)
        return input_data

    def _time_encoding(self, times):
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
        B, L, D = x.shape
        print(f"Input shape: {x.shape}")

        surf_vars = self.surf_vars
        atmos_vars = self.atmos_vars
        atmos_levels = self.atmos_levels
        print(f"Surface variables: {surf_vars}")
        print(f"Atmospheric variables: {atmos_vars}")
        print(f"Atmospheric levels: {atmos_levels}")

        lat, lon = batch.metadata.lat, batch.metadata.lon
        H, W = self.H, self.W
        print(f"lat shape: {lat.shape}, lon shape: {lon.shape}")

        # Generate queries for Perceiver IO
        num_queries = len(surf_vars) + len(atmos_vars) * len(atmos_levels)
        queries = torch.zeros(B, num_queries, D, device=x.device)
        print(f"Queries shape: {queries.shape}")

        # Create 2D grid of lat and lon
        lat_grid, lon_grid = torch.meshgrid(lat, lon, indexing="ij")
        print(f"Lat grid shape: {lat_grid.shape}, Lon grid shape: {lon_grid.shape}")
        # print(f"Lat grid contents: {lat_grid}")
        # print(f"Lon grid contents: {lon_grid}")
        pos_input = torch.stack((lat_grid, lon_grid), dim=-1)
        print(f"Position input shape: {pos_input.shape}")

        # Add position encoding
        pos_encode = self._apply_position_encoding(pos_input)
        pos_encode = self.pos_embed(pos_encode)
        pos_encode = pos_encode.view(1, H * W, -1)
        pos_encode = F.interpolate(pos_encode.transpose(1, 2), size=num_queries, mode="linear", align_corners=False)
        pos_encode = pos_encode.transpose(1, 2)
        print(f"Position encoding shape: {pos_encode.shape}")

        queries = queries + pos_encode

        # Add lead time embedding
        lead_hours = lead_time.total_seconds() / 3600
        lead_times = lead_hours * torch.ones(B, dtype=x.dtype, device=x.device)
        lead_time_encode = self._time_encoding(lead_times)
        lead_time_emb = self.lead_time_embed(lead_time_encode)
        queries = queries + lead_time_emb.unsqueeze(1)

        # Add absolute time embedding
        absolute_times = torch.tensor(
            [[t.timestamp() / 3600 for t in time_list] for time_list in batch.metadata.time], dtype=torch.float32, device=x.device
        )
        absolute_time_encode = self._time_encoding(absolute_times)
        absolute_time_embed = self.absolute_time_embed(absolute_time_encode)
        absolute_time_embed = absolute_time_embed.mean(dim=1, keepdim=True)
        queries = queries + absolute_time_embed

        print(f"Queries shape after adding embeddings: {queries.shape}")

        # Reshape x to match the expected input shape for PerceiverIO
        x_reshaped = x.view(B, -1, D)
        print(f"Reshaped input shape: {x_reshaped.shape}")

        # Apply Perceiver IO
        decoded = self.perceiver_io(x_reshaped, queries=queries)
        print(f"Decoded shape after Perceiver IO: {decoded.shape}")

        # Split decoded output into surface and atmospheric variables
        surf_decoded = decoded[:, : len(self.surf_vars)]
        atmos_decoded = decoded[:, len(self.surf_vars) :]

        # Project surface variables
        surf_output = self.surf_token_proj(surf_decoded)
        print(f"Surface output shape after projection: {surf_output.shape}")

        # Reshape surface variables
        surf_output = surf_output.view(B, len(self.surf_vars), -1)
        surf_output = surf_output.view(B, len(self.surf_vars), self.H, self.W)
        print(f"Surface output shape: {surf_output.shape}")

        # Project atmospheric variables
        atmos_output = self.atmos_token_proj(atmos_decoded)
        print(f"Atmospheric output shape after projection: {atmos_output.shape}")

        # Reshape atmospheric variables
        atmos_output = atmos_output.view(B, len(self.atmos_vars), len(self.atmos_levels), -1)
        atmos_output = atmos_output.view(B, len(self.atmos_vars), len(self.atmos_levels), self.H, self.W)
        print(f"Atmospheric output shape: {atmos_output.shape}")

        # Construct output dictionary
        output = {
            "surf_vars": {var: surf_output[:, i] for i, var in enumerate(self.surf_vars)},
            "atmos_vars": {var: atmos_output[:, i] for i, var in enumerate(self.atmos_vars)},
        }

        return output


def main():
    # dummy data (similar to the encoder example)
    Batch = namedtuple("Batch", ["surf_vars", "static_vars", "atmos_vars", "metadata"])
    Metadata = namedtuple("Metadata", ["lat", "lon", "time", "atmos_levels"])

    B, T, V_s, V_a, C, H, W = 32, 2, 3, 5, 13, 32, 64
    surf_vars = {f"surf_var_{i}": torch.randn(B, T, H, W) for i in range(V_s)}
    atmos_vars = {f"atmos_var_{i}": torch.randn(B, T, C, H, W) for i in range(V_a)}

    lat = torch.linspace(-90, 90, H)
    lon = torch.linspace(0, 360, W)
    time = [[datetime.now() + timedelta(hours=i) for i in range(T)] for _ in range(B)]
    atmos_levels = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50]

    metadata = Metadata(lat=lat, lon=lon, time=time, atmos_levels=atmos_levels)
    batch = Batch(surf_vars=surf_vars, static_vars={}, atmos_vars=atmos_vars, metadata=metadata)

    decoder = BFMDecoder(
        surf_vars=tuple(surf_vars.keys()),
        atmos_vars=tuple(atmos_vars.keys()),
        atmos_levels=atmos_levels,
        H=H,
        W=W,
    )

    # Simulating encoder output
    x = torch.randn(B, 7, 1024)  # Assuming 7 latent tokens and 1024 embedding dimension

    lead_time = timedelta(hours=6)
    output = decoder(x, batch, lead_time)

    print("\nDecoder output:")
    for var_type, vars_dict in output.items():
        print(f"{var_type}:")
        for var_name, var_tensor in vars_dict.items():
            print(f"  {var_name}: {var_tensor.shape}")


if __name__ == "__main__":
    main()
