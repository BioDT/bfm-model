import functools
from math import pi
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat


def generate_fourier_features(
    pos: torch.Tensor,
    num_bands: int,
    batch_size: int,
    max_freq: float = 10.0,
    concat_pos: bool = True,
    sine_only: bool = False,
) -> torch.Tensor:
    pos = pos.unsqueeze(-1)
    device, dtype, orig_x = pos.device, pos.dtype, pos

    scales = torch.linspace(1.0, max_freq / 2, num_bands, device=device, dtype=dtype)
    scales = scales[(*((None,) * (len(pos.shape) - 1)), Ellipsis)]

    pos = pos * scales * pi

    if sine_only:
        pos = pos.sin()
    else:
        pos = torch.cat([pos.sin(), pos.cos()], dim=-1)

    pos = torch.cat((pos, orig_x), dim=-1)

    # Rearrange dimensions
    side_enc_pos = rearrange(pos, "... n d -> ... (n d)")

    # Repeat for batch size if necessary
    if side_enc_pos.shape[0] != batch_size:
        side_enc_pos = repeat(side_enc_pos, "... -> b ...", b=batch_size)

    if not concat_pos:
        side_enc_pos = side_enc_pos[..., : -pos.shape[-1]]

    return side_enc_pos


def build_linear_positions(index_dims: Tuple[int], output_range: Tuple[float, float] = (-1.0, 1.0)) -> torch.Tensor:
    def _linspace(n_xels_per_dim: int) -> torch.Tensor:
        return torch.linspace(output_range[0], output_range[1], steps=n_xels_per_dim)

    per_dim_ranges = [_linspace(n_xels_per_dim) for n_xels_per_dim in index_dims]
    array_index_grid = torch.meshgrid(*per_dim_ranges, indexing="ij")

    stacked_index_dim = torch.stack(array_index_grid, dim=-1)
    return stacked_index_dim


class AbstractPositionEncoding(nn.Module):
    """Abstract Perceiver encoder."""

    def forward(self, batch_size: int, pos: Optional[torch.Tensor] = None) -> torch.Tensor:
        raise NotImplementedError


class TrainablePositionEncoding(AbstractPositionEncoding):
    """Trainable position encoding."""

    def __init__(self, index_dim: int, num_channels: int = 128, init_scale: float = 0.02):
        super(TrainablePositionEncoding, self).__init__()
        self.pos_embs = nn.Parameter(torch.randn(index_dim, num_channels) * init_scale)
        self.output_size = num_channels

    def forward(self, batch_size: Optional[int] = None, pos: Optional[torch.Tensor] = None) -> torch.Tensor:
        pos_embs = self.pos_embs
        if batch_size is not None:
            pos_embs = pos_embs.unsqueeze(0).expand(batch_size, -1, -1)
        return pos_embs


class FourierPositionEncoding(AbstractPositionEncoding):
    def __init__(
        self,
        index_dims: Tuple[int],
        num_bands: int,
        max_freq: float = 10.0,
        concat_pos: bool = True,
        sine_only: bool = False,
    ):
        super(FourierPositionEncoding, self).__init__()
        self.num_bands = num_bands
        self.max_freq = max_freq
        self.concat_pos = concat_pos
        self.sine_only = sine_only
        self.index_dims = index_dims
        self.output_size = None

    def _check_or_build_spatial_positions(
        self, pos: Optional[torch.Tensor], index_dims: Tuple[int], batch_size: int
    ) -> torch.Tensor:
        if pos is None:
            axis_pos = [torch.linspace(-1.0, 1.0, steps=size) for size in index_dims]
            pos = torch.stack(torch.meshgrid(*axis_pos, indexing="ij"), dim=-1)
            pos = pos.unsqueeze(0).expand(batch_size, *pos.shape)
        else:
            assert pos.shape[1:-1] == index_dims, f"Position shape {pos.shape[1:-1]} does not match index dims {index_dims}."
            assert pos.shape[0] == batch_size, f"Batch size {pos.shape[0]} does not match expected batch size {batch_size}."
        return pos

    def forward(self, batch_size: int, pos: Optional[torch.Tensor] = None) -> torch.Tensor:
        pos = self._check_or_build_spatial_positions(pos, self.index_dims, batch_size)
        side_enc_pos = generate_fourier_features(
            pos=pos,
            num_bands=self.num_bands,
            batch_size=batch_size,
            max_freq=self.max_freq,
            concat_pos=self.concat_pos,
            sine_only=self.sine_only,
        )
        return side_enc_pos


class PositionEncodingProjector(AbstractPositionEncoding):
    """Projects a position encoding to a target size."""

    def __init__(self, output_size: int, base_position_encoding: AbstractPositionEncoding):
        super(PositionEncodingProjector, self).__init__()
        self.output_size = output_size
        self.base_position_encoding = base_position_encoding

    def forward(self, batch_size: int, pos: Optional[torch.Tensor] = None) -> torch.Tensor:
        base_pos = self.base_position_encoding(batch_size, pos)
        if not hasattr(self, "linear"):
            self.linear = nn.Linear(base_pos.shape[-1], self.output_size)  # set linear layer dynamically
        projected_pos = self.linear(base_pos)
        return projected_pos


def build_position_encoding(
    position_encoding_type: str,
    index_dims: Tuple[int],
    project_pos_dim: int = -1,
    trainable_position_encoding_kwargs: Optional[dict] = None,
    fourier_position_encoding_kwargs: Optional[dict] = None,
) -> AbstractPositionEncoding:

    if position_encoding_type == "trainable":
        assert trainable_position_encoding_kwargs is not None
        output_pos_enc = TrainablePositionEncoding(index_dim=int(np.prod(index_dims)), **trainable_position_encoding_kwargs)
    elif position_encoding_type == "fourier":
        assert fourier_position_encoding_kwargs is not None
        output_pos_enc = FourierPositionEncoding(index_dims=index_dims, **fourier_position_encoding_kwargs)
    else:
        raise ValueError(f"Unknown position encoding: {position_encoding_type}.")

    if project_pos_dim > 0:
        output_pos_enc = PositionEncodingProjector(output_size=project_pos_dim, base_position_encoding=output_pos_enc)

    return output_pos_enc


if __name__ == "__main__":
    # Example usage:
    pos_enc = build_position_encoding(
        position_encoding_type="fourier",
        index_dims=(4, 4),
        fourier_position_encoding_kwargs={"num_bands": 3, "max_resolution": (8, 8), "concat_pos": True, "sine_only": False},
    )
    pos = pos_enc(batch_size=23)
    print(pos.shape)  # should be (2, 50176, 256)

    # pos_enc = build_position_encoding(
    #     position_encoding_type="trainable",
    #     index_dims=(224, 224),
    #     project_pos_dim=128,
    #     trainable_position_encoding_kwargs={"num_channels": 128, "init_scale": 0.02},
    # )
    # pos = pos_enc(batch_size=2)
    # print(pos.shape)  # should be (2, 50176, 256)
