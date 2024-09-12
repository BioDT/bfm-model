from collections import namedtuple
from datetime import datetime, timedelta

import torch

from src.bfm.src.decoder import BFMDecoder
from src.bfm.src.encoder import BFMEncoder


class BFM(torch.nn.Module):
    def __init__(
        self,
        surf_vars: tuple[str, ...],
        static_vars: tuple[str, ...],
        atmos_vars: tuple[str, ...],
        H: int = 32,
        W: int = 64,
        embed_dim: int = 1024,
        num_latent_tokens: int = 7,
        **kwargs,
    ):
        super().__init__()
        self.encoder = BFMEncoder(
            surf_vars=surf_vars, static_vars=static_vars, atmos_vars=atmos_vars, embed_dim=embed_dim, **kwargs
        )
        self.decoder = BFMDecoder(surf_vars=surf_vars, atmos_vars=atmos_vars, H=H, W=W, embed_dim=embed_dim, **kwargs)
        self.latent_proj = torch.nn.Linear(embed_dim, num_latent_tokens * embed_dim)

    def forward(self, batch, lead_time: timedelta):
        # encode
        encoded = self.encoder(batch, lead_time)

        # project to fixed number of latent tokens
        B, _, D = encoded.shape
        latent = self.latent_proj(encoded).view(B, -1, D)

        # decode
        output = self.decoder(latent, batch, lead_time)

        return output


def main():
    # dummy data (similar to encoder and decoder examples)
    Batch = namedtuple("Batch", ["surf_vars", "static_vars", "atmos_vars", "metadata"])
    Metadata = namedtuple("Metadata", ["lat", "lon", "time", "atmos_levels"])

    B, T, V_s, V_a, C, H, W = 2, 2, 3, 5, 13, 32, 64
    surf_vars = {f"surf_var_{i}": torch.randn(B, T, H, W) for i in range(V_s)}
    static_vars = {f"static_var_{i}": torch.randn(H, W) for i in range(2)}
    atmos_vars = {f"atmos_var_{i}": torch.randn(B, T, C, H, W) for i in range(V_a)}

    lat = torch.linspace(-90, 90, H)
    lon = torch.linspace(0, 360, W)
    time = [datetime.now() + timedelta(hours=i) for i in range(T)]
    atmos_levels = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50]

    metadata = Metadata(lat=lat, lon=lon, time=time, atmos_levels=atmos_levels)
    batch = Batch(surf_vars=surf_vars, static_vars=static_vars, atmos_vars=atmos_vars, metadata=metadata)

    model = BFM(
        surf_vars=tuple(surf_vars.keys()),
        static_vars=tuple(static_vars.keys()),
        atmos_vars=tuple(atmos_vars.keys()),
        H=H,
        W=W,
        atmos_levels=atmos_levels,
    )

    # set a lead time
    lead_time = timedelta(hours=6)

    output = model(batch, lead_time)

    print("\nBFM output:")
    for var_type, vars_dict in output.items():
        print(f"{var_type}:")
        for var_name, var_tensor in vars_dict.items():
            print(f"  {var_name}: {var_tensor.shape}")


if __name__ == "__main__":
    main()
