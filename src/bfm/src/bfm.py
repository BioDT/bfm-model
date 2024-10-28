from collections import namedtuple
from datetime import datetime, timedelta
from typing import Literal

import torch
import torch.nn as nn

from src.bfm.src.decoder import BFMDecoder
from src.bfm.src.encoder import BFMEncoder
from src.mvit.mvit_model import MViT
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
        num_latent_tokens: int = 8,
        backbone_type: Literal["swin", "mvit"] = "mvit",
        **kwargs,
    ):
        super().__init__()
        self.encoder = BFMEncoder(
            surf_vars=surf_vars,
            static_vars=static_vars,
            atmos_vars=atmos_vars,
            latent_levels=num_latent_tokens,
            embed_dim=embed_dim,
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
                window_size=(1, 2, 2),
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
            surf_vars=surf_vars, atmos_vars=atmos_vars, atmos_levels=atmos_levels, H=H, W=W, embed_dim=embed_dim, **kwargs
        )

    def forward(self, batch, lead_time: timedelta):
        # Encode
        encoded = self.encoder(batch, lead_time)

        # Process through backbone
        patch_shape = self.encoder.patch_shape

        if self.backbone_type == "mvit":
            # Reshape for MViT
            encoded = encoded.view(encoded.size(0), -1, self.encoder.embed_dim)

        backbone_output = self.backbone(encoded, lead_time=lead_time, rollout_step=0, patch_shape=patch_shape)

        # Decode
        output = self.decoder(backbone_output, batch, lead_time)

        return output


def create_test_input():
    Batch = namedtuple("Batch", ["surf_vars", "static_vars", "atmos_vars", "metadata"])
    Metadata = namedtuple("Metadata", ["lat", "lon", "time", "atmos_levels", "rollout_step"])

    B, T, V_s, V_a, C, H, W = 32, 2, 3, 5, 13, 32, 64
    surf_vars = {f"surf_var_{i}": torch.randn(B, T, H, W) for i in range(V_s)}
    static_vars = {f"static_var_{i}": torch.randn(H, W) for i in range(2)}
    atmos_vars = {f"atmos_var_{i}": torch.randn(B, T, C, H, W) for i in range(V_a)}

    lat = torch.linspace(-90, 90, H)
    lon = torch.linspace(0, 360, W)
    time = [[datetime.now() + timedelta(hours=i) for i in range(T)] for _ in range(B)]
    atmos_levels = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50]

    metadata = Metadata(lat=lat, lon=lon, time=time, atmos_levels=atmos_levels, rollout_step=0)
    batch = Batch(surf_vars=surf_vars, static_vars=static_vars, atmos_vars=atmos_vars, metadata=metadata)

    lead_time = timedelta(hours=6)

    return batch, lead_time


def main():
    batch, lead_time = create_test_input()

    model = BFM(
        surf_vars=tuple(batch.surf_vars.keys()),
        static_vars=tuple(batch.static_vars.keys()),
        atmos_vars=tuple(batch.atmos_vars.keys()),
        atmos_levels=batch.metadata.atmos_levels,
        backbone_type="swin",  # or "mvit"
    )

    output = model(batch, lead_time)

    print("Model output:")
    for var_type, vars_dict in output.items():
        print(f"{var_type}:")
        for var_name, var_tensor in vars_dict.items():
            print(f"  {var_name}: {var_tensor.shape}")


if __name__ == "__main__":
    main()
