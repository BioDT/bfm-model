import math

import numpy as np
import torch
import torch.nn as nn

from src.swin_transformer.helpers.patches_area import EARTH_RADIUS, compute_polygon_area


class FourierExpansion(nn.Module):
    """
    This class implements a Fourier-like encoding that maps input values to a higher-dimensional space.
    It's particularly useful for encoding continuous values (like positions or times) into a format
    more suitable for neural networks to process.

    Attributes:
        lower (float): Lower bound of the input range (smallest wavelength).
        upper (float): Upper bound of the input range (largest wavelength).
        assert_range (bool): If True, assert that the encoded tensor is within the specified wavelength range.
    """

    def __init__(self, lower: float, upper: float, assert_range: bool = True) -> None:
        """
        Args:
            lower (float): Lower bound of the input range (smallest wavelength).
            upper (float): Upper bound of the input range (largest wavelength).
            assert_range (bool): If True, assert that the encoded tensor is within the specified
                wavelength range. Default: True
        """
        super().__init__()
        self.lower = lower
        self.upper = upper
        self.assert_range = assert_range

    def forward(self, x: torch.Tensor, d: int) -> torch.Tensor:
        """Perform the Fourier-like expansion.

        Adds a dimension of length `d` to the end of the shape of `x`.

        Args:
            x (torch.Tensor): Input to expand. Shape: [..., n]. All elements of `x` must
                lie within [self.lower, self.upper] if self.assert_range is True.
            d (int): Dimensionality of the expansion. Must be even.

        Raises:
            AssertionError: If self.assert_range is True and not all elements of `x` are
                within [self.lower, self.upper].
            ValueError: If `d` is not even.

        Returns:
            torch.Tensor: Fourier series-style expansion of `x`. Shape: [..., n, d].
        """
        # Check if input is within the configured range (allowing zeros)
        in_range = torch.logical_and(self.lower <= x.abs(), x.abs() <= self.upper)
        in_range_or_zero = torch.logical_or(in_range, x == 0)
        if self.assert_range and not torch.all(in_range_or_zero):
            raise AssertionError(f"Input tensor outside range [{self.lower}, {self.upper}].")

        if d % 2 != 0:
            raise ValueError("Dimensionality must be even.")

        # Cast to float64 for numerical stability
        x = x.to(torch.float64)

        # Generate logarithmically spaced wavelengths
        wavelengths = torch.logspace(
            math.log10(self.lower),
            math.log10(self.upper),
            d // 2,
            base=10,
            device=x.device,
            dtype=x.dtype,
        )  # shape: [d//2]

        # Compute phase angles
        phase_angles = torch.einsum("...i,j->...ij", x, 2 * np.pi / wavelengths)  # shape: [..., n, d//2]

        # Compute sin and cos components
        sin_component = torch.sin(phase_angles)
        cos_component = torch.cos(phase_angles)

        # Concatenate sin and cos components
        encoding = torch.cat((sin_component, cos_component), dim=-1)  # shape: [..., n, d]

        return encoding.float()  # Cast back to float32 for compatibility


# Constants for patch area calculation
_DELTA = 0.01  # Smallest delta in latitude and longitude (degrees)
_EARTH_CIRCUMFERENCE = 2 * np.pi * EARTH_RADIUS  # Earth's circumference (km)

# Calculate minimum patch area (at the poles)
_min_patch_area: float = compute_polygon_area(
    torch.tensor(
        [
            [90, 0],
            [90, _DELTA],
            [90 - _DELTA, _DELTA],
            [90 - _DELTA, 0],
        ],
        dtype=torch.float64,
    )
).item()

# Calculate Earth's surface area
_area_earth = 4 * np.pi * EARTH_RADIUS**2


lead_time_expansion = FourierExpansion(1 / 60, 24 * 7 * 3)  # Encodes lead times (hours). Range: 1 minute to 3 weeks.
