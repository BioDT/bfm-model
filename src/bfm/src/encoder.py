import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from src.perceiver_components.helpers_io import dropout_seq
from src.perceiver_components.pos_encoder import build_position_encoding
from src.perceiver_core.perceiver_io import PerceiverIO


class BFMEncoder(nn.Module):
    def __init__(
        self,
        surf_vars: tuple[str, ...],
        static_vars: tuple[str, ...] | None,
        atmos_vars: tuple[str, ...],
        atmos_levels: list[int] | None = None,
        patch_size: int = 4,
        latent_levels: int = 8,
        embed_dim: int = 1024,
        num_heads: int = 16,
        head_dim: int = 64,
        drop_rate: float = 0.1,
        depth: int = 2,
        mlp_ratio: float = 4.0,
        max_history_size: int = 2,
        perceiver_ln_eps: float = 1e-5,
        num_fourier_bands: int = 64,
        max_frequency: float = 224.0,
        num_input_axes: int = 2,
        position_encoding_type: str = "fourier",
    ):
        super().__init__()
        self.drop_rate = drop_rate
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.max_history_size = max_history_size
        self.num_input_axes = num_input_axes
        self.max_frequency = max_frequency
        self.num_fourier_bands = num_fourier_bands
        self.position_encoding_type = position_encoding_type

        self.surf_vars = surf_vars
        self.static_vars = static_vars
        self.atmos_vars = atmos_vars

        surf_vars = surf_vars + static_vars if static_vars is not None else surf_vars
        self.surf_var_map = {v: i for i, v in enumerate(surf_vars)}
        self.atmos_var_map = {v: i for i, v in enumerate(atmos_vars)}

        self.latent_levels = latent_levels
        self.atmos_latents = nn.Parameter(torch.randn(latent_levels - 1, embed_dim))
        self.surf_level_encoding = nn.Parameter(torch.randn(embed_dim))

        pos_encoding_dim = self._calculate_pos_encoding_dim()
        # print(f"Calculated pos_encoding_dim: {pos_encoding_dim}")
        self.pos_embed = nn.Linear(self._calculate_pos_encoding_dim() + 2, embed_dim)  # +2 for lat and lon
        self.scale_embed = nn.Linear(embed_dim, embed_dim)
        self.lead_time_embed = nn.Linear(embed_dim, embed_dim)
        self.absolute_time_embed = nn.Linear(embed_dim, embed_dim)
        self.atmos_levels_embed = nn.Linear(pos_encoding_dim, embed_dim)

        self.surf_token_embeds = self._create_patch_embed(len(surf_vars), patch_size, embed_dim, max_history_size)
        self.atmos_token_embeds = self._create_patch_embed(len(atmos_vars), patch_size, embed_dim, max_history_size)

        # total_tokens = self._calculate_total_tokens(surf_vars, static_vars, atmos_vars)
        self.perceiver_io = PerceiverIO(
            num_layers=depth,
            dim=embed_dim,
            queries_dim=embed_dim,
            logits_dimension=None,
            num_latent_tokens=latent_levels,
            latent_dimension=embed_dim,
            cross_attention_heads=num_heads,
            latent_attention_heads=num_heads,
            cross_attention_head_dim=head_dim,
            latent_attention_head_dim=head_dim,
            num_fourier_bands=num_fourier_bands,
            max_frequency=max_frequency,
            num_input_axes=2,
            position_encoding_type=position_encoding_type,
        )

        self.pos_drop = nn.Dropout(p=drop_rate)

        self.apply(self._init_weights)
        nn.init.trunc_normal_(self.atmos_latents, std=0.02)
        nn.init.trunc_normal_(self.surf_level_encoding, std=0.02)

    def _calculate_pos_encoding_dim(self):
        return 2 * self.num_fourier_bands + 1

    def _calculate_total_tokens(self, surf_vars, static_vars, atmos_vars):
        return len(surf_vars) + len(static_vars or []) + len(atmos_vars)

    def _create_patch_embed(self, num_vars, patch_size, embed_dim, max_history_size):
        in_dim = num_vars * max_history_size * patch_size * patch_size
        return nn.Sequential(nn.Linear(in_dim, embed_dim), nn.LayerNorm(embed_dim))

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

    def forward(self, batch, lead_time):
        surf_vars = self.surf_vars
        static_vars = self.static_vars
        atmos_vars = self.atmos_vars
        atmos_levels = batch.metadata.atmos_levels
        print(f"Surface variables: {surf_vars}")
        print(f"Static variables: {static_vars}")
        print(f"Atmospheric variables: {atmos_vars}")
        print(f"Atmospheric levels: {atmos_levels}")

        x_surf = torch.stack(tuple(batch.surf_vars.values()), dim=2)
        x_atmos = torch.stack(tuple(batch.atmos_vars.values()), dim=2)
        print(f"x_surf shape: {x_surf.shape}")
        print(f"x_atmos shape: {x_atmos.shape}")

        B, T, _, C, H, W = x_atmos.size()
        print(f"Batch size (B): {B}, Time steps (T): {T}, Atmos levels (C): {C}, Height (H): {H}, Width (W): {W}")

        if static_vars:
            print("Processing static variables")
            x_static = torch.stack(tuple(batch.static_vars.values()), dim=0)
            x_static = x_static.unsqueeze(0).unsqueeze(0).expand(B, T, -1, H, W)
            x_surf = torch.cat((x_surf, x_static), dim=2)
            print(f"x_surf shape after adding static vars: {x_surf.shape}")

        lat, lon = batch.metadata.lat, batch.metadata.lon
        print(f"lat shape: {lat.shape}, lon shape: {lon.shape}")

        # Create 2D grid of lat and lon
        lat_grid, lon_grid = torch.meshgrid(lat, lon, indexing="ij")
        print(f"Lat grid shape: {lat_grid.shape}, Lon grid shape: {lon_grid.shape}")
        pos_input = torch.stack((lat_grid, lon_grid), dim=-1)
        print(f"Position input shape: {pos_input.shape}")

        # Add position encoding
        pos_encode = self._apply_position_encoding(pos_input)
        pos_encode = self.pos_embed(pos_encode)
        pos_encode = pos_encode.view(1, H * W, -1)
        pos_encode = F.interpolate(
            pos_encode.transpose(1, 2), size=(H // self.patch_size) * (W // self.patch_size), mode="linear", align_corners=False
        )
        pos_encode = pos_encode.transpose(1, 2)
        print(f"Position encoding shape: {pos_encode.shape}")

        # Process surface variables
        x_surf = rearrange(x_surf, "b t v (h p1) (w p2) -> (b h w) (v t p1 p2)", p1=self.patch_size, p2=self.patch_size)
        x_surf = self.surf_token_embeds(x_surf)
        x_surf = rearrange(x_surf, "(b h w) d -> b (h w) d", h=H // self.patch_size, w=W // self.patch_size)
        x_surf = x_surf + self.surf_level_encoding[None, None, :]
        print(f"x_surf shape after processing: {x_surf.shape}")

        # Process atmospheric variables
        x_atmos = rearrange(x_atmos, "b t v c (h p1) (w p2) -> (b c h w) (v t p1 p2)", p1=self.patch_size, p2=self.patch_size)
        x_atmos = self.atmos_token_embeds(x_atmos)
        x_atmos = rearrange(x_atmos, "(b c h w) d -> b c (h w) d", b=B, c=C, h=H // self.patch_size, w=W // self.patch_size)

        # Add atmospheric level embeddings
        atmos_levels_tensor = torch.tensor(atmos_levels, device=x_atmos.device)
        atmos_levels_encode = self._apply_position_encoding(atmos_levels_tensor.unsqueeze(0).unsqueeze(-1))
        atmos_levels_encode = atmos_levels_encode[..., :-1]
        atmos_levels_embed = self.atmos_levels_embed(atmos_levels_encode.squeeze(0))
        atmos_levels_embed = atmos_levels_embed.unsqueeze(0).unsqueeze(2).expand_as(x_atmos)
        x_atmos = x_atmos + atmos_levels_embed
        print(f"x_atmos shape after processing: {x_atmos.shape}")

        # Combine surface and atmospheric data
        x = torch.cat((x_surf.unsqueeze(1), x_atmos), dim=1)
        print(f"Combined x shape: {x.shape}")

        # Add position encoding
        x = x + pos_encode.unsqueeze(1)

        # Add lead time embedding
        lead_hours = lead_time.total_seconds() / 3600
        lead_times = lead_hours * torch.ones(B, dtype=x.dtype, device=x.device)
        lead_time_encode = self._time_encoding(lead_times)
        lead_time_emb = self.lead_time_embed(lead_time_encode)
        x = x + lead_time_emb.unsqueeze(1).unsqueeze(1)

        # Add absolute time embedding
        absolute_times = torch.tensor(
            [[t.timestamp() / 3600 for t in time_list] for time_list in batch.metadata.time], dtype=torch.float32, device=x.device
        )
        absolute_time_encode = self._time_encoding(absolute_times)
        absolute_time_embed = self.absolute_time_embed(absolute_time_encode)
        absolute_time_embed = absolute_time_embed.mean(dim=1, keepdim=True)
        x = x + absolute_time_embed.unsqueeze(1)

        print(f"x shape after adding all embeddings: {x.shape}")

        # considering the surface latents as well, to not have L = latent_levels - 1, but L = latent_levels instead
        x = self.pos_drop(x)
        surface_latent = self.surf_level_encoding.unsqueeze(0).unsqueeze(0).expand(B, 1, -1)
        atmos_latents = self.atmos_latents.unsqueeze(0).expand(B, -1, -1)
        latents = torch.cat([surface_latent, atmos_latents], dim=1)
        x = self.perceiver_io(x, queries=latents)

        return x
        # latents = self.atmos_latents.unsqueeze(0).expand(B, -1, -1)
        # x = self.perceiver_io(x, queries=latents)
        # print(f"Final output shape: {x.shape}")

        # return x

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


def main():
    from collections import namedtuple
    from datetime import datetime, timedelta

    Batch = namedtuple("Batch", ["surf_vars", "static_vars", "atmos_vars", "metadata"])
    Metadata = namedtuple("Metadata", ["lat", "lon", "time", "atmos_levels"])

    B, T, V_s, V_a, C, H, W = 32, 2, 3, 5, 13, 32, 64
    surf_vars = {f"surf_var_{i}": torch.randn(B, T, H, W) for i in range(V_s)}
    static_vars = {f"static_var_{i}": torch.randn(H, W) for i in range(2)}
    atmos_vars = {f"atmos_var_{i}": torch.randn(B, T, C, H, W) for i in range(V_a)}

    print("Input shapes:")
    print(f"surf_vars: {', '.join([f'{k}: {v.shape}' for k, v in surf_vars.items()])}")
    print(f"static_vars: {', '.join([f'{k}: {v.shape}' for k, v in static_vars.items()])}")
    print(f"atmos_vars: {', '.join([f'{k}: {v.shape}' for k, v in atmos_vars.items()])}")

    lat = torch.linspace(-90, 90, H)
    lon = torch.linspace(0, 360, W)
    time = [[datetime.now() + timedelta(hours=i) for i in range(T)] for _ in range(B)]
    atmos_levels = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50]

    metadata = Metadata(lat=lat, lon=lon, time=time, atmos_levels=atmos_levels)
    batch = Batch(surf_vars=surf_vars, static_vars=static_vars, atmos_vars=atmos_vars, metadata=metadata)

    encoder = BFMEncoder(
        surf_vars=tuple(surf_vars.keys()),
        static_vars=tuple(static_vars.keys()),
        atmos_vars=tuple(atmos_vars.keys()),
    )

    lead_time = timedelta(hours=6)
    output = encoder(batch, lead_time)
    print(f"Encoder output shape: {output.shape}")


if __name__ == "__main__":
    main()
