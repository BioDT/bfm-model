import math
from collections import namedtuple
from datetime import datetime, timedelta

import torch
import torch.nn as nn

from src.bfm.src.decoder import BFMDecoder
from src.bfm.src.encoder import BFMEncoder
from src.swin_transformer.core.swim_core_v2 import Swin3DTransformer


class BFM(nn.Module):
    def __init__(
        self,
        surf_vars: tuple[str, ...],
        static_vars: tuple[str, ...],
        atmos_vars: tuple[str, ...],
        atmos_levels: list[int],
        H: int = 32,
        W: int = 64,
        embed_dim: int = 1024,
        num_latent_tokens: int = 7,
        **kwargs,
    ):
        super().__init__()
        self.encoder = BFMEncoder(
            surf_vars=surf_vars, latent_levels=16, static_vars=static_vars, atmos_vars=atmos_vars, embed_dim=embed_dim, **kwargs
        )

        self.num_latent_tokens = num_latent_tokens

        # swin backbone
        self.backbone = Swin3DTransformer(
            embed_dim=embed_dim,
            encoder_depths=(1, 1),
            encoder_num_heads=(8, 16),
            decoder_depths=(1, 1),
            decoder_num_heads=(16, 8),
            window_size=(1, 2, 2),
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.1,
            use_lora=False,
        )

        # Latent projection
        self.latent_proj = nn.Linear(embed_dim, num_latent_tokens * embed_dim)

        self.decoder = BFMDecoder(
            surf_vars=surf_vars, atmos_vars=atmos_vars, atmos_levels=atmos_levels, H=H, W=W, embed_dim=embed_dim, **kwargs
        )

    def forward(self, batch, lead_time: timedelta):
        # Encode
        encoded = self.encoder(batch, lead_time)

        # Get the actual shape
        B, L, D = encoded.shape
        print(f"Encoded shape: {encoded.shape}")

        # Calculate patch_res based on L
        C = 1
        H = W = int(math.sqrt(L))  # Assuming square patches
        patch_res = (C, H, W)
        print(f"Calculated patch_res: {patch_res}")

        # Process through Swin Transformer backbone
        backbone_output = self.backbone(encoded, lead_time, 0, patch_res)

        print(f"Backbone output shape: {backbone_output.shape}")

        # Project to fixed number of latent tokens
        # latent = self.latent_proj(backbone_output).view(B, self.num_latent_tokens, -1)

        # print(f"Latent shape: {latent.shape}")

        # Decode

        output = self.decoder(backbone_output, batch, lead_time)

        return output


def main():
    # Test the BFM model
    surf_vars = ("surf_var_1", "surf_var_2", "surf_var_3")
    static_vars = ("static_var_1", "static_var_2")
    atmos_vars = ("atmos_var_1", "atmos_var_2", "atmos_var_3", "atmos_var_4", "atmos_var_5")
    atmos_levels = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50]

    model = BFM(
        surf_vars=surf_vars,
        static_vars=static_vars,
        atmos_vars=atmos_vars,
        atmos_levels=atmos_levels,
        H=32,
        W=64,
        embed_dim=1024,
        num_latent_tokens=7,
    )

    # Create dummy input data
    Batch = namedtuple("Batch", ["surf_vars", "static_vars", "atmos_vars", "metadata"])
    Metadata = namedtuple("Metadata", ["lat", "lon", "time", "atmos_levels"])

    B, T, V_s, V_a, C, H, W = 2, 2, 3, 5, 13, 32, 64
    surf_vars = {f"surf_var_{i}": torch.randn(B, T, H, W) for i in range(V_s)}
    static_vars = {f"static_var_{i}": torch.randn(H, W) for i in range(2)}
    atmos_vars = {f"atmos_var_{i}": torch.randn(B, T, C, H, W) for i in range(V_a)}

    lat = torch.linspace(-90, 90, H)
    lon = torch.linspace(0, 360, W)
    time = [[datetime.now() + timedelta(hours=i) for i in range(T)] for _ in range(B)]

    metadata = Metadata(lat=lat, lon=lon, time=time, atmos_levels=atmos_levels)
    batch = Batch(surf_vars=surf_vars, static_vars=static_vars, atmos_vars=atmos_vars, metadata=metadata)

    lead_time = timedelta(hours=6)

    # Forward pass
    output = model(batch, lead_time)

    print("Model output:")
    for var_type, vars_dict in output.items():
        print(f"{var_type}:")
        for var_name, var_tensor in vars_dict.items():
            print(f"  {var_name}: {var_tensor.shape}")


if __name__ == "__main__":
    main()
