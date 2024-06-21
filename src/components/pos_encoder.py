import functools
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def generate_fourier_features(
    pos: torch.Tensor,
    num_bands: int,
    max_resolution: Tuple[int, int] = (224, 224),
    concat_pos: bool = True,
    sine_only: bool = False,
) -> torch.Tensor:
    print(f"Position Shape: {pos.shape}")
    min_freq = 1.0
    freq_bands = torch.stack([torch.linspace(min_freq, res / 2, steps=num_bands) for res in max_resolution], dim=0)

    per_pos_features = pos[:, :, None] * freq_bands[None, :, :]
    per_pos_features = per_pos_features.view(-1, per_pos_features.shape[1] * per_pos_features.shape[2])

    if sine_only:
        per_pos_features = torch.sin(torch.pi * per_pos_features)
    else:
        per_pos_features = torch.cat([torch.sin(torch.pi * per_pos_features), torch.cos(torch.pi * per_pos_features)], dim=-1)

    if concat_pos:
        per_pos_features = torch.cat([pos, per_pos_features], dim=-1)

    print(per_pos_features.shape)
    return per_pos_features


def build_linear_positions(index_dims: Tuple[int], output_range: Tuple[float, float] = (-1.0, 1.0)) -> torch.Tensor:
    def _linspace(n_xels_per_dim: int) -> torch.Tensor:
        return torch.linspace(output_range[0], output_range[1], steps=n_xels_per_dim)

    dim_ranges = [_linspace(n_xels_per_dim) for n_xels_per_dim in index_dims]
    array_index_grid = torch.meshgrid(*dim_ranges, indexing="ij")

    return torch.stack(array_index_grid, dim=-1)


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


def _check_or_build_spatial_positions(pos: Optional[torch.Tensor], index_dims: Tuple[int], batch_size: int) -> torch.Tensor:
    if pos is None:
        pos = build_linear_positions(index_dims)
        pos = pos.unsqueeze(0).expand(batch_size, *pos.shape)
        pos = pos.view(batch_size, -1, pos.shape[-1])
    else:
        assert pos.shape[-1] == len(index_dims)

    return pos


class FourierPositionEncoding(AbstractPositionEncoding):
    """Fourier (Sinusoidal) position encoding."""

    def __init__(
        self,
        index_dims: Tuple[int],
        num_bands: int,
        concat_pos: bool = True,
        max_resolution: Optional[Tuple[int, int]] = None,
        sine_only: bool = False,
    ):
        super(FourierPositionEncoding, self).__init__()
        self.num_bands = num_bands
        self.concat_pos = concat_pos
        self.sine_only = sine_only
        self.index_dims = index_dims
        self.max_resolution = max_resolution or index_dims
        self.output_size = None  # initialize output_size as None

    def forward(self, batch_size: int, pos: Optional[torch.Tensor] = None) -> torch.Tensor:
        pos = _check_or_build_spatial_positions(pos, self.index_dims, batch_size)
        build_ff_fn = functools.partial(
            generate_fourier_features,
            num_bands=self.num_bands,
            max_resolution=self.max_resolution,
            concat_pos=self.concat_pos,
            sine_only=self.sine_only,
        )
        encoding = torch.stack([build_ff_fn(p) for p in pos])
        if self.output_size is None:
            self.output_size = encoding.shape[-1]  # set output_size dynamically
        return encoding


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
        index_dims=(224, 224),
        project_pos_dim=256,
        fourier_position_encoding_kwargs={"num_bands": 4, "concat_pos": True, "sine_only": False},
    )
    pos = pos_enc(batch_size=1)
    print(pos.shape)  # should be (2, 50176, 256)

    # pos_enc = build_position_encoding(
    #     position_encoding_type="trainable",
    #     index_dims=(224, 224),
    #     project_pos_dim=128,
    #     trainable_position_encoding_kwargs={"num_channels": 128, "init_scale": 0.02},
    # )
    # pos = pos_enc(batch_size=2)
    # print(pos.shape)  # should be (2, 50176, 256)
