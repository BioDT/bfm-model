"""
Copyright 2025 (C) TNO. Licensed under the MIT license.
"""

from math import pi
from typing import Literal, Optional, Tuple

import torch
import torch.nn as nn
from einops import rearrange, repeat


class AbstractPositionEncoding(nn.Module):
    """
    Abstract base class for position encodings.

    This class defines the interface for position encodings used in the Perceiver architecture.
    """

    def forward(self, batch_size: int, pos: Optional[torch.Tensor] = None) -> torch.Tensor:
        raise NotImplementedError


class TrainablePositionEncoding(AbstractPositionEncoding):
    """
    Trainable position encoding.

    This class implements trainable position encodings where the embeddings
    are learned during the training process.
    """

    def __init__(self, index_dims: Tuple[int], num_channels: int = 128, init_scale: float = 0.02):
        """
        Initialize the trainable position encoding.

        Args:
            index_dims: Dimensions of the input indices
            num_channels: Number of channels per position encoding (size of each generated positional embeddings)
            init_scale: Initial scale for a bit of randomness in the initialization of embeddings
        """
        super(TrainablePositionEncoding, self).__init__()
        self.index_dims = index_dims
        self.num_channels = num_channels
        self.output_size = num_channels

        # Initialize trainable position embeddings, shape: (*index_dims, num_channels)
        self.pos_embs = nn.Parameter(torch.randn(*index_dims, num_channels) * init_scale)

    def forward(self, batch_size: int, pos: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Expand the learned position embeddings to match the batch size, shape: (batch_size, *index_dims, num_channels)
        pos_embs = self.pos_embs.unsqueeze(0).expand(batch_size, *self.pos_embs.shape)

        return pos_embs

    @staticmethod
    def build_linear_positions(index_dims: Tuple[int], output_range: Tuple[float, float] = (-1.0, 1.0)) -> torch.Tensor:
        """
        Build linear positions for the given index dimensions.

        Args:
            index_dims: Dimensions of the input indices
            output_range: Range of the output values

        Returns:
            Tensor containing linear positions
        """

        def _linspace(n_xels_per_dim: int) -> torch.Tensor:
            return torch.linspace(output_range[0], output_range[1], steps=n_xels_per_dim)

        # Create ranges for each dimension
        per_dim_ranges = [_linspace(n_xels_per_dim) for n_xels_per_dim in index_dims]
        # Create a grid of positions
        array_index_grid = torch.meshgrid(*per_dim_ranges, indexing="ij")
        # Stack the grid dimensions, shape: (*index_dims, len(index_dims))
        stacked_index_dim = torch.stack(array_index_grid, dim=-1)
        return stacked_index_dim


class FourierPositionEncoding(AbstractPositionEncoding):
    """
    Fourier position encoding.

    This class implements a position encoding based on Fourier features.
    """

    def __init__(
        self,
        index_dims: Tuple[int],
        num_bands: int,
        max_freq: float = 10.0,
        concat_pos: bool = True,
        sine_only: bool = False,
    ):
        """
        Initialize the Fourier position encoding.

        Args:
            index_dims: Dimensions of the input indices
            num_bands: Number of frequency bands to construct the features from
            max_freq: Maximum frequency of a band
            concat_pos: Whether to concatenate the original position to the encoding
            sine_only: Whether to use only sine functions (otherwise uses both sine and cosine)
        """
        super(FourierPositionEncoding, self).__init__()
        self.num_bands = num_bands
        self.max_freq = max_freq
        self.concat_pos = concat_pos
        self.sine_only = sine_only
        self.index_dims = index_dims
        self.output_size = None  # will be set after first forward pass

    def _check_or_build_spatial_positions(
        self, pos: Optional[torch.Tensor], index_dims: Tuple[int], batch_size: int
    ) -> torch.Tensor:
        if pos is None:
            # Build linear positions if not provided, shape: (*index_dims, len(index_dims))
            pos = self.build_linear_positions(index_dims)
            # Shape: (batch_size, *index_dims, len(index_dims))
            pos = pos.unsqueeze(0).expand(batch_size, *pos.shape)
        else:
            # Validate the shape of provided positions
            assert pos.shape[1:-1] == index_dims, f"Position shape {pos.shape[1:-1]} does not match index dims {index_dims}."
            assert pos.shape[0] == batch_size, f"Batch size {pos.shape[0]} does not match expected batch size {batch_size}."
        return pos

    @staticmethod
    def build_linear_positions(index_dims: Tuple[int], output_range: Tuple[float, float] = (-1.0, 1.0)) -> torch.Tensor:
        """
        Build linear positions for the given index dimensions.

        Args:
            index_dims: Dimensions of the input indices
            output_range: Range of the output values

        Returns:
            Tensor containing linear positions, shape: (*index_dims, len(index_dims))
        """

        def _linspace(n_xels_per_dim: int) -> torch.Tensor:
            return torch.linspace(output_range[0], output_range[1], steps=n_xels_per_dim)

        per_dim_ranges = [_linspace(n_xels_per_dim) for n_xels_per_dim in index_dims]
        array_index_grid = torch.meshgrid(*per_dim_ranges, indexing="ij")
        stacked_index_dim = torch.stack(array_index_grid, dim=-1)
        return stacked_index_dim

    @staticmethod
    def generate_fourier_features(
        pos: torch.Tensor,
        num_bands: int,
        batch_size: int,
        max_freq: float = 10.0,
        concat_pos: bool = True,
        sine_only: bool = False,
    ) -> torch.Tensor:
        """
        Generate Fourier features from the given positions.

        Args:
            pos: Tensor of input positions, of shape: (batch_size, *index_dims, len(index_dims))
            num_bands: Number of frequency bands to construct the features from
            batch_size: Number of samples in the batch
            max_freq: Maximum frequency of a band
            concat_pos: Whether to concatenate the original position to the encoding
            sine_only: Whether to use only sine functions (otherwise uses both sine and cosine)

        Returns:
            Tensor of Fourier features, shape: (batch_size, *index_dims, num_bands * len(index_dims) * (1 if sine_only else 2) + (len(index_dims) if concat_pos else 0))
        """
        # Shape: (batch_size, *index_dims, len(index_dims), 1)
        pos = pos.unsqueeze(-1)
        device, dtype, orig_x = pos.device, pos.dtype, pos.clone()

        # Generate frequency scales, shape: (num_bands,)
        scales = torch.linspace(1.0, max_freq / 2, num_bands, device=device, dtype=dtype)
        # Shape: (1, 1, ..., 1, num_bands), with len(index_dims) + 1 leading ones
        scales = scales[(*((None,) * (len(pos.shape) - 1)), ...)]

        # Apply the scales to the linear positions, shape: (batch_size, *index_dims, len(index_dims), num_bands)
        pos = pos * scales * pi

        # Apply sine (and cosine if not sine_only)
        if sine_only:
            # Shape: (batch_size, *index_dims, len(index_dims), num_bands)
            pos = pos.sin()
        else:
            # Shape: (batch_size, *index_dims, len(index_dims), 2 * num_bands)
            pos = torch.cat([pos.sin(), pos.cos()], dim=-1)

        # Only concatenate original positions if concat_pos is True
        # Shape: (batch_size, *index_dims, len(index_dims), 2 * num_bands + 1) if not sine_only and concat_pos
        if concat_pos:
            pos = torch.cat((pos, orig_x), dim=-1)

        # Rearrange dimensions
        # Shape: (batch_size, *index_dims, len(index_dims) * (2 * num_bands + 1)) if not sine_only and concat_pos
        side_enc_pos = rearrange(pos, "... n d -> ... (n d)")

        # Repeat for batch size if necessary
        if side_enc_pos.shape[0] != batch_size:
            side_enc_pos = repeat(side_enc_pos, "... -> b ...", b=batch_size)

        return side_enc_pos

    def forward(self, batch_size: int, pos: Optional[torch.Tensor] = None) -> torch.Tensor:
        pos = self._check_or_build_spatial_positions(pos, self.index_dims, batch_size)
        side_enc_pos = self.generate_fourier_features(
            pos=pos,
            num_bands=self.num_bands,
            batch_size=batch_size,
            max_freq=self.max_freq,
            concat_pos=self.concat_pos,
            sine_only=self.sine_only,
        )
        return side_enc_pos


class PositionEncodingProjector(AbstractPositionEncoding):
    """
    Projects a position encoding to a target size.
    This class takes a base position encoding and projects it to a specified output size.
    """

    def __init__(self, output_size: int, base_position_encoding: AbstractPositionEncoding):
        """
        Initialize the position encoding projector.

        Args:
            output_size: Desired size of the output encoding
            base_position_encoding: Base position encoding to be projected
        """
        super(PositionEncodingProjector, self).__init__()
        self.output_size = output_size
        self.base_position_encoding = base_position_encoding

    def forward(self, batch_size: int, pos: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Shape: (batch_size, *index_dims, base_encoding_size)
        base_pos = self.base_position_encoding(batch_size, pos)
        if not hasattr(self, "linear"):
            # Dynamically create the linear projection layer
            self.linear = nn.Linear(base_pos.shape[-1], self.output_size)
        # Shape: (batch_size, *index_dims, output_size)
        projected_pos = self.linear(base_pos)
        return projected_pos


def build_position_encoding(
    position_encoding_type: Literal["trainable", "fourier"],
    index_dims: Tuple[int],
    project_pos_dim: int = -1,
    trainable_position_encoding_kwargs: Optional[dict] = None,
    fourier_position_encoding_kwargs: Optional[dict] = None,
) -> AbstractPositionEncoding:
    """
    Build a position encoding based on the specified type and parameters.

    Args:
        position_encoding_type: Type of position encoding ("trainable" or "fourier")
        index_dims: Dimensions of the input indices
        project_pos_dim: Dimension to project the position encoding to (-1 for no projection)
        trainable_position_encoding_kwargs: Keyword arguments for trainable position encoding
        fourier_position_encoding_kwargs: Keyword arguments for Fourier position encoding

    Returns:
        An instance of the specified position encoding

    Raises:
        ValueError: If an unknown position encoding type is specified
    """
    if position_encoding_type == "trainable":
        assert trainable_position_encoding_kwargs is not None
        output_pos_enc = TrainablePositionEncoding(index_dims=index_dims, **trainable_position_encoding_kwargs)
    elif position_encoding_type == "fourier":
        assert fourier_position_encoding_kwargs is not None
        output_pos_enc = FourierPositionEncoding(index_dims=index_dims, **fourier_position_encoding_kwargs)
    else:
        raise ValueError(f"Unknown position encoding: {position_encoding_type}.")

    if project_pos_dim > 0:  # TODO: Possibly fix after the new added changes?
        output_pos_enc = PositionEncodingProjector(output_size=project_pos_dim, base_position_encoding=output_pos_enc)

    return output_pos_enc


def main():
    # Example usage fourier:
    pos_enc = build_position_encoding(
        position_encoding_type="fourier",
        index_dims=(4, 4),
        fourier_position_encoding_kwargs={"num_bands": 3, "max_freq": 8, "concat_pos": True, "sine_only": False},
    )
    # Output shape: (23, 4, 4, 3 * 2 * 2 + 2) = (23, 4, 4, 14)
    pos = pos_enc(batch_size=23)

    # Example usage trainable:
    trainable_pos_enc = TrainablePositionEncoding(index_dims=(4, 4), num_channels=14)
    # Output shape: (23, 4, 4, 14)
    pos = trainable_pos_enc(batch_size=23)
    print(pos.shape)


if __name__ == "__main__":
    main()
