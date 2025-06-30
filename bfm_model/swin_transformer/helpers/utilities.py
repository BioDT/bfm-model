"""
Copyright 2025 (C) TNO. Licensed under the MIT license.
"""

from typing import TypeVar

import torch
from einops import rearrange
from timm.models.vision_transformer import trunc_normal_
from torch import nn


def unpatchify(x: torch.Tensor, V: int, H: int, W: int, P: int) -> torch.Tensor:
    """
    This function takes a patchified input tensor and reconstructs it into its original spatial dimensions.

    Args:
        x (torch.Tensor): Patchified input tensor. Shape: [B, L, C, V * P^2]
            B: Batch size
            L: Number of patches (H * W / P^2)
            C: Number of channels
            V: Number of variables
            P: Patch size
        V (int): Number of variables. Used to reshape the input tensor.
        H (int): Original height of the unpatchified representation.
        W (int): Original width of the unpatchified representation.
        P (int): Patch size. Used to calculate the number of patches and reshape the tensor.

    Returns:
        torch.Tensor: Unpatchified representation. Shape: [B, V, C, H, W]
            The tensor is reshaped to its original spatial dimensions, with variables as a separate dimension.

    Note:
        This function assumes that the input tensor has been patchified using a non-overlapping patch strategy.
    """
    assert x.dim() == 4, f"Expected 4D tensor, but got {x.dim()}D."
    batch_size, num_channels = x.size(0), x.size(2)

    # Find the number of patches in height and width
    num_patches_height = H // P
    num_patches_width = W // P

    # Verify that the nubmer of patches matches the input tensor's second dimension
    assert x.size(1) == num_patches_height * num_patches_width, "Mismatch in number of patches"
    assert (
        x.size(-1) == V * P**2
    ), "Mismatch in last dimension size"  # and ensure the last dimension matches the expected size for variables and patch

    # Reshape the tensor to separate patches and variables
    unpatchified = x.reshape(
        shape=(batch_size, num_patches_height, num_patches_width, num_channels, P, P, V)
    )  # shape: [batch_size, num_patches_height, num_patches_width, num_channels, P, P, V]

    # Rearrange dimensions to group spatial dimensions and separate variables
    unpatchified = rearrange(
        unpatchified, "B H W C P1 P2 V -> B V C H P1 W P2"
    )  # shape: [batch_size, V, num_channels, num_patches_height, P, num_patches_width, P]

    # Reshape to fnial output format, merging patch dimensions back inot full spatial dimensions
    unpatchified = unpatchified.reshape(shape=(batch_size, V, num_channels, H, W))  # shape: [batch_size, V, num_channels, H, W]

    return unpatchified


def check_lat_lon_dtype(latitude: torch.Tensor, longitude: torch.Tensor) -> None:
    """
    This function checks if the latitude and longitude tensors have sufficient precision
    to avoid numerical instability in subsequent computations.

    Args:
        latitude (torch.Tensor): Latitude tensor to check.
        longitude (torch.Tensor): Longitude tensor to check.

    Raises:
        AssertionError: If either lat or lon is not at least float32 precision.

    Note:
        float32 is considered the minimum precision to ensure numerical stability.
        float64 is also acceptable for even higher precision calculations.
    """
    # Check if latitude tensor has sufficient precision (float32 or float64)
    assert latitude.dtype in [
        torch.float32,
        torch.float64,
    ], f"Latitude precision insufficient: {latitude.dtype}. Use at least float32."
    # Check if longitude tensor has sufficient precision (float32 or float64)
    assert longitude.dtype in [
        torch.float32,
        torch.float64,
    ], f"Longitude precision insufficient: {longitude.dtype}. Use at least float32."


# TypeVar for window size and shift size, which can be either 2D or 3D
DimensionTuple = TypeVar("DimensionTuple", tuple[int, int], tuple[int, int, int])


def adjust_windows(
    window_size: DimensionTuple, shift_size: DimensionTuple, res: DimensionTuple
) -> tuple[DimensionTuple, DimensionTuple]:
    """
    This function ensures that the window and shift sizes are appropriate for the given input resolution.
    If the input resolution in any dimension is smaller than the corresponding window size,
    the window size is adjusted to match the input resolution and the shift size is set to 0 for that dimension.
    That would ensure that the window size is always less than or equal to the input resolution.

    Args:
        window_size (DimensionTuple): Original window size for each dimension. Shape: [D1, D2, ...].
        shift_size (DimensionTuple): Original shift size for each dimension. Shape: [D1, D2, ...].
        res (DimensionTuple): Input resolution for each dimension. Shape: [D1, D2, ...].

    Returns:
        tuple[DimensionTuple, DimensionTuple]: Adjusted window size and shift size.

    Note:
        DimensionTuple is a TypeVar that can be either tuple[int, int] or tuple[int, int, int].
        This allows the function to work with both 2D and 3D inputs.
    """
    # Ensure that window_size, shift_size, and res have the same number of dimensions
    dimension_mismatch_msg = f"Expected same number of dimensions, found {len(window_size)}, {len(shift_size)} and {len(res)}."
    assert len(window_size) == len(shift_size) == len(res), dimension_mismatch_msg

    # Convert tuples to lists for mutability
    adjusted_shift_size, adjusted_window_size = list(shift_size), list(
        window_size
    )  # Shape: [D1, D2, ...] where D1, D2, ... are the number of dimensions

    # Iterate through each dimension
    for dim in range(len(res)):
        # If the input resolution is smaller than or equal to the window size in this dimension
        if res[dim] <= window_size[dim]:
            adjusted_shift_size[dim] = 0  # set shift size to 0 to prevent shifting in this dimension
            adjusted_window_size[dim] = res[dim]  # adjust window size to match the input resolution in this dimension

    # Convert adjusted lists back to tuples and type cast to DimensionTuple
    new_window_size: DimensionTuple = tuple(adjusted_window_size)
    new_shift_size: DimensionTuple = tuple(adjusted_shift_size)
    # Shape of new_window_size and new_shift_size: [D1, D2, ...] where D1, D2, ... are the number of dimensions

    # Ensure all dimensions of the new window size are positive
    assert min(new_window_size) > 0, f"Window size must be positive in all dimensions. Found {new_window_size}."
    # .. and also ensure all dimensions of the new shift size are non-negative
    assert min(new_shift_size) >= 0, f"Shift size must be non-negative in all dimensions. Found {new_shift_size}."

    return new_window_size, new_shift_size


def init_weights(m: nn.Module):
    """Initialise weights of a module with a truncated normal distribution.

    This function is used to initialize the weights of various types of neural network layers.
    Different initialization strategies are applied based on the type of layer:

    - For Linear, Conv2d, Conv3d, ConvTranspose2d, and ConvTranspose3d layers:
      - Weights are initialized from a truncated normal distribution with std=0.02
      - Biases, if present, are initialized to 0

    - For LayerNorm layers:
      - Weight, if present, is initialized to 1.0
      - Bias, if present, is initialized to 0

    Args:
        m (torch.nn.Module): The module whose weights are to be initialized.

    Example:
        >>> model = nn.Sequential(nn.Linear(10, 5), nn.LayerNorm(5))
        >>> model.apply(init_weights)
    """
    # Initialize weights for various types of layers
    if isinstance(m, (nn.Linear, nn.Conv2d, nn.Conv3d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
        # For linear and convolutional layers:
        # Initialize weights using truncated normal distribution
        nn.init.trunc_normal_(m.weight, std=0.02)

        if m.bias is not None:
            # Initialize bias to zero if present
            nn.init.zeros_(m.bias)

    elif isinstance(m, nn.LayerNorm):
        # For LayerNorm layers:
        # Initialize weight to one and bias to zero
        if m.weight is not None:
            nn.init.ones_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
