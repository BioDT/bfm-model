# src/dataset_creation/metadata.py

import dataclasses
from datetime import datetime

import torch


@dataclasses.dataclass
class BatchMetadata:
    """
    Encapsulates metadata for a batch of data.

    Args:
        latitudes (torch.Tensor): Tensor containing latitude values.
        longitudes (torch.Tensor): Tensor containing longitude values.
        timestamp (tuple[datetime, ...]): Tuple of datetime objects representing the time associated with each batch element.
        pressure_levels (tuple[int | float, ...]): Atmospheric pressure levels in hPa for the variables.
        prediction_step (int, optional): Number of roll-out steps used for generating the prediction.
    """

    latitudes: torch.Tensor
    longitudes: torch.Tensor
    timestamp: tuple[datetime, ...]
    pressure_levels: tuple[int | float, ...]
    prediction_step: int = 0

    def __post_init__(self):
        """
        Validates the latitude and longitude tensors to ensure they meet the required conditions:
        - Latitudes must be strictly decreasing.
        - Latitudes must be within the range [-90, 90].
        - Longitudes must be strictly increasing.
        - Longitudes must be within the range [0, 360).

        Raises:
            ValueError: If any of the conditions are not met.
        """
        if not torch.all(self.latitudes[1:] - self.latitudes[:-1] < 0):
            raise ValueError("Latitudes must be strictly decreasing.")

        if not (torch.all(self.latitudes <= 90) and torch.all(self.latitudes >= -90)):
            raise ValueError("Latitudes must be within the range [-90, 90].")

        if not torch.all(self.longitudes[1:] - self.longitudes[:-1] > 0):
            raise ValueError("Longitudes must be strictly increasing.")

        if not (torch.all(self.longitudes >= 0) and torch.all(self.longitudes < 360)):
            raise ValueError("Longitudes must be in the range [0, 360).")
