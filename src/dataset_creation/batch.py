# src/dataset_creation/batch.py

from dataclasses import dataclass
from typing import Callable, Dict, Tuple

import torch

from src.data_preprocessing.transformation.era5 import (
    normalise_atmospheric_variables,
    normalise_surface_variable,
    unnormalise_atmospheric_variables,
    unnormalise_surface_variables,
)
from src.dataset_creation.metadata import BatchMetadata


@dataclass
class DataBatch:
    """
    Represents a batch of data including surface, static, and atmospheric variables.

    Args:
        surface_variables (Dict[str, torch.Tensor]): Dictionary of surface-level variables, where each tensor has shape `(b, t, h, w)`.
        static_variables (Dict[str, torch.Tensor]): Dictionary of static variables, where each tensor has shape `(h, w)`.
        atmospheric_variables (Dict[str, torch.Tensor]): Dictionary of atmospheric variables, where each tensor has shape `(b, t, c, h, w)`.
        batch_metadata (Metadata): Metadata associated with this batch, containing information such as latitudes, longitudes, and time.
    """

    surface_variables: Dict[str, torch.Tensor]
    single_variables: Dict[str, torch.Tensor]
    atmospheric_variables: Dict[str, torch.Tensor]
    species_variables: Dict[str, torch.Tensor]
    batch_metadata: BatchMetadata

    @property
    def spatial_dimensions(self) -> Tuple[int, int]:
        """
        Retrieve the spatial dimensions `(height, width)` from any surface-level variable in the batch.

        Returns:
            Tuple[int, int]: The spatial dimensions of the surface-level variables.
        """
        return next(iter(self.surface_variables.values())).shape[-2:]

    def _fmap(self, f: Callable[[torch.Tensor], torch.Tensor]) -> "DataBatch":
        """
        Applies a function `f` to all tensor elements within the DataBatch.

        Args:
            f (Callable[[torch.Tensor], torch.Tensor]): A function that takes a torch.Tensor
            as input and returns a torch.Tensor, applied to each tensor in the DataBatch.

        Returns:
            DataBatch: A new DataBatch instance with the function `f` applied to its tensors.
        """
        surface_variables = {key: f(value) for key, value in self.surface_variables.items()}
        single_variables = {key: f(value) for key, value in self.single_variables.items()}
        atmospheric_variables = {key: f(value) for key, value in self.atmospheric_variables.items()}
        species_variables = {key: f(value) for key, value in self.species_variables.items()}

        return DataBatch(
            surface_variables=surface_variables,
            single_variables=single_variables,
            atmospheric_variables=atmospheric_variables,
            species_variables=species_variables,
            batch_metadata=BatchMetadata(
                latitudes=f(self.batch_metadata.latitudes),
                longitudes=f(self.batch_metadata.longitudes),
                pressure_levels=self.batch_metadata.pressure_levels,
                timestamp=self.batch_metadata.timestamp,
                prediction_step=self.batch_metadata.prediction_step,
            ),
        )

    def to(self, device: str | torch.device) -> "DataBatch":
        """
        Moves all tensors within the DataBatch to the specified device.

        Args:
            device (str | torch.device): The device to which the tensors should be moved.

        Returns:
            DataBatch: A new DataBatch instance with all tensors moved to the specified device.
        """
        return self._fmap(lambda x: x.to(device))

    def type(self, t: type) -> "DataBatch":
        """
        Converts all tensors within the DataBatch to the specified data type.

        Args:
            t (type): The data type to which the tensors should be converted.

        Returns:
            DataBatch: A new DataBatch instance with all tensors converted to the specified type.
        """
        return self._fmap(lambda x: x.type(t))

    def normalize_data(self, locations, scales) -> "DataBatch":
        """
        Normalise all variables in the batch.

        Args:
            locations (Dict[str, float]): A dictionary where the key is the variable name
                                    and the value is the mean for that variable.
            scales (Dict[str, float]): A dictionary where the key is the variable name
                                        and the value is the standard deviation for that variable.

        Returns:
            DataBatch: A new batch with all variables normalized.
        """
        normalized_surface_variables = {
            key: normalise_surface_variable(value, key, locations, scales) for key, value in self.surface_variables.items()
        }
        normalized_single_variables = {
            key: normalise_surface_variable(value, key, locations, scales) for key, value in self.single_variables.items()
        }
        normalized_atmospheric_variables = {
            key: normalise_atmospheric_variables(value, key, self.batch_metadata.pressure_levels, locations, scales)
            for key, value in self.atmospheric_variables.items()
        }

        return DataBatch(
            surface_variables=normalized_surface_variables,
            static_variables=normalized_single_variables,
            atmospheric_variables=normalized_atmospheric_variables,
            batch_metadata=self.batch_metadata,
        )

    def unnormalise_data(self, locations, scales) -> "DataBatch":
        """
        Unnormalise all variables in the batch.

        Args:
            locations (Dict[str, float]): A dictionary where the key is the variable name
                                    and the value is the mean for that variable.
            scales (Dict[str, float]): A dictionary where the key is the variable name
                                        and the value is the standard deviation for that variable.

        Returns:
            DataBatch: A new batch with all variables unnormalized.
        """
        unnormalized_surface_variables = {
            key: unnormalise_surface_variables(value, key, locations, scales) for key, value in self.surface_variables.items()
        }
        unnormalized_single_variables = {
            key: unnormalise_surface_variables(value, key, locations, scales) for key, value in self.single_variables.items()
        }
        unnormalized_atmospheric_variables = {
            key: unnormalise_atmospheric_variables(value, key, self.batch_metadata.pressure_levels, locations, scales)
            for key, value in self.atmospheric_variables.items()
        }

        return DataBatch(
            surface_variables=unnormalized_surface_variables,
            static_variables=unnormalized_single_variables,
            atmospheric_variables=unnormalized_atmospheric_variables,
            batch_metadata=self.batch_metadata,
        )

    def crop(self, patch_size: int) -> "DataBatch":
        """
        Crop the variables in the batch to the specified patch size.

        Args:
            patch_size (int): The target patch size to crop the data to. Default is 4.

        Returns:
            Batch: A new batch with variables cropped to the specified patch size.

        Raises:
            ValueError: If the width of the data is not a multiple of the patch size.
            ValueError: If more than one latitude is not divisible by the patch size.
        """
        height, width = self.spatial_dimensions

        if width % patch_size != 0:
            raise ValueError(f"Data width ({width}) must be a multiple of the patch size ({patch_size}).")

        if height % patch_size == 0:
            return self

        elif height % patch_size == 1:
            cropped_surface = {key: value[..., :-1, :] for key, value in self.surface_variables.items()}
            cropped_static = {key: value[:-1, :] for key, value in self.single_variables.items()}
            cropped_atmospheric = {key: value[..., :-1, :] for key, value in self.atmospheric_variables.items()}
            cropped_species = {key: value[..., :-1, :] for key, value in self.species_variables.items()}

            return DataBatch(
                surface_variables=cropped_surface,
                static_variables=cropped_static,
                atmospheric_variables=cropped_atmospheric,
                species_variables=cropped_species,
                batch_metadata=self.batch_metadata,
            )
        else:
            excess_rows = height % patch_size
            raise ValueError(f"Expected at most one extra latitude row, but found {excess_rows}.")
