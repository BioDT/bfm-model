"""
Copyright 2025 (C) TNO. Licensed under the MIT license.
"""

import os
from collections import namedtuple
from datetime import datetime, timedelta
from typing import Literal

import torch
from omegaconf.dictconfig import DictConfig
from torch.utils.data import Dataset, default_collate

from bfm_model.bfm.dataset_basics import *
from bfm_model.bfm.scaler import (
    _rescale_recursive,
    dimensions_to_keep_by_key,
    load_stats,
)

# Namedtuple definitions
Batch = namedtuple(
    "Batch",
    [
        "batch_metadata",
        "surface_variables",
        "single_variables",
        "species_variables",
        "atmospheric_variables",
        "species_extinction_variables",
        "land_variables",
        "agriculture_variables",
        "forest_variables",
    ],
)

Metadata = namedtuple("Metadata", ["latitudes", "longitudes", "timestamp", "lead_time", "pressure_levels", "species_list"])


def custom_collate(batch_list):
    """
    Custom collate function that handles a batch of Batches.
    Since each file corresponds to a single sample, and we define batch_size in the DataLoader,
    a batch from the DataLoader will be a list of Batch objects.
    """

    if not batch_list:
        return batch_list

    # If we got a list of Batches:
    if isinstance(batch_list[0], Batch):
        # We need to collate them into a single Batch of stacked tensors where appropriate.

        def collate_dicts(dict_list):
            # dict_list: list of dicts, one per sample
            # keys should be identical for all samples
            out = {}
            keys = dict_list[0].keys()
            for key in keys:
                values = [d[key] for d in dict_list]
                # If these are tensors, stack them
                if isinstance(values[0], torch.Tensor):
                    out[key] = default_collate(values)
                else:
                    # If these are lists (like timestamps), just keep as a list of lists
                    out[key] = values
            return out

        # Collate metadata
        # For metadata, lat/lon/pressure_levels are identical or global.
        # timestamp might differ per sample.
        # We'll just take them as is. If needed, we can stack or keep lists.

        latitudes = batch_list[0].batch_metadata.latitudes
        longitudes = batch_list[0].batch_metadata.longitudes
        pressure_levels = batch_list[0].batch_metadata.pressure_levels
        species_list = batch_list[0].batch_metadata.species_list
        # Each sample has its own timestamp (list)
        timestamps = [b.batch_metadata.timestamp for b in batch_list]
        # timestamps_all = [b.batch_metadata.timestamp for b in batch_list]
        # If each sample has a single timestamp, we get a list of timestamps:
        # timestamps = [ts[0] for ts in timestamps_all]
        lead_time = batch_list[0].batch_metadata.lead_time

        metadata = Metadata(
            latitudes=latitudes,
            longitudes=longitudes,
            timestamp=timestamps[0],  # a list of timestamps, one per sample
            lead_time=lead_time,
            pressure_levels=pressure_levels,
            species_list=species_list,
        )

        surface_vars = collate_dicts([b.surface_variables for b in batch_list])
        single_vars = collate_dicts([b.single_variables for b in batch_list])
        atmospheric_vars = collate_dicts([b.atmospheric_variables for b in batch_list])
        species_ext_vars = collate_dicts([b.species_extinction_variables for b in batch_list])
        land_vars = collate_dicts([b.land_variables for b in batch_list])
        agriculture_vars = collate_dicts([b.agriculture_variables for b in batch_list])
        forest_vars = collate_dicts([b.forest_variables for b in batch_list])
        species_vars = collate_dicts([b.species_variables for b in batch_list])

        return Batch(
            batch_metadata=metadata,
            surface_variables=surface_vars,
            single_variables=single_vars,
            atmospheric_variables=atmospheric_vars,
            species_extinction_variables=species_ext_vars,
            land_variables=land_vars,
            agriculture_variables=agriculture_vars,
            forest_variables=forest_vars,
            species_variables=species_vars,
        )

    # Fallback for other types if encountered
    return default_collate(batch_list)


def crop_variables(variables, new_H, new_W, handle_nans=True, nan_mode="mean_clip", fix_dim=False):
    """
    Crop and clean variables to specified dimensions, handling NaN and Inf values.

    Args:
        variables (dict): Dictionary of variable tensors to process
        new_H (int): Target height dimension
        new_W (int): Target width dimension
        handle_nans (bool): Whether to handle NaN values at all
        nan_mode (str): Strategy for NaN handling.
            - "mean_clip": old logic (replace NaNs with mean, clip to mean ± 2*std)
            - "zero": replace all NaNs with 0.0, no extra clipping

    Returns:
        dict: Processed variables with cleaned and cropped tensors
    """
    processed_vars = {}
    for k, v in variables.items():
        # crop dimensions
        if fix_dim:
            cropped = v[:, :new_H, :new_W, :]
        else:
            cropped = v[..., :new_H, :new_W]
        # Handle infinities
        inf_mask = torch.isinf(cropped)
        inf_count = inf_mask.sum().item()
        if inf_count > 0:
            valid_values = cropped[~inf_mask & ~torch.isnan(cropped)]
            if len(valid_values) > 0:
                max_val = valid_values.max().item()
                min_val = valid_values.min().item()
                cropped = torch.clip(cropped, min_val, max_val)
            else:
                cropped = torch.clip(cropped, -1e6, 1e6)

        # Handle NaNs if requested
        if handle_nans:
            nan_mask = torch.isnan(cropped)
            nan_count = nan_mask.sum().item()
            if nan_count > 0:
                if nan_mode == "mean_clip":
                    valid_values = cropped[~nan_mask & ~torch.isinf(cropped)]
                    if len(valid_values) > 0:
                        mean_val = valid_values.mean().item()
                        std_val = valid_values.std().item()
                        clip_min = mean_val - 2 * std_val
                        clip_max = mean_val + 2 * std_val

                        # Replace NaNs with mean
                        cropped = torch.nan_to_num(cropped, nan=mean_val)
                        # Convert to float32 if needed
                        cropped = cropped.to(torch.float32)
                        # Clip
                        cropped = torch.clip(cropped, clip_min, clip_max)
                    else:
                        # If no valid values, just fill with 0 and do a small clip
                        cropped = torch.nan_to_num(cropped, nan=0.0)
                        cropped = torch.clip(cropped, -1.0, 1.0)

                elif nan_mode == "zero":
                    # Simply replace all NaNs with 0.0
                    cropped = torch.nan_to_num(cropped)
                    # cropped = cropped.to(torch.float32)

                else:
                    raise ValueError(f"Unknown nan_mode: {nan_mode}")
        # else: do nothing special for NaNs

        processed_vars[k] = cropped

    return processed_vars


class LargeClimateDataset(Dataset):
    """
    A dataset where each file in `data_dir` is a single sample.
    Each file should have structure:
    {
        "batch_metadata": {...},
        "surface_variables": {...},
        "single_variables": {...},
        "species_variables" {...},
        "atmospheric_variables": {...},
        "species_extinction_variables": {...},
        "land_variables": {...},
        "agriculture_variables": {...},
        "forest_variables": {...}
    }
    """

    def __init__(
        self, data_dir: str, scaling_settings: DictConfig, num_species: int = 2, mode: str = "pretrain", model_patch_size: int = 4
    ):
        self.data_dir = data_dir
        self.num_species = num_species
        self.mode = mode
        self.files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".pt")]
        self.files.sort()
        # print("Files sorted", self.files)
        self.scaling_settings = scaling_settings
        self.scaling_statistics = load_stats(scaling_settings.stats_path)
        self.model_patch_size = model_patch_size
        print(f"We scale the dataset {scaling_settings.enabled} with {scaling_settings.mode}")

    def __len__(self):
        return max(0, len(self.files) - 1)

    def load_and_process_files(self, fpath: str):
        data = torch.load(fpath, map_location="cpu", weights_only=True)

        latitudes = data["batch_metadata"]["latitudes"]
        longitudes = data["batch_metadata"]["longitudes"]
        timestamps = data["batch_metadata"]["timestamp"]
        pressure_levels = data["batch_metadata"]["pressure_levels"]
        species_list = data["batch_metadata"]["species_list"]

        # Determine original spatial dimensions from metadata lists
        H = len(data["batch_metadata"]["latitudes"])
        W = len(data["batch_metadata"]["longitudes"])

        # crop dimensions to be divisible by patch size
        patch_size = 4  # TODO make this configurable
        new_H = (H // patch_size) * patch_size
        new_W = (W // patch_size) * patch_size

        # normalize or standardize variables
        data = self.scale_batch(data, direction="scaled")

        surface_vars = crop_variables(data["surface_variables"], new_H, new_W)
        single_vars = crop_variables(data["single_variables"], new_H, new_W)
        atmospheric_vars = crop_variables(data["atmospheric_variables"], new_H, new_W)
        species_ext_vars = crop_variables(data["species_extinction_variables"], new_H, new_W)
        land_vars = crop_variables(data["land_variables"], new_H, new_W)
        agriculture_vars = crop_variables(data["agriculture_variables"], new_H, new_W)
        forest_vars = crop_variables(data["forest_variables"], new_H, new_W)
        species_vars = crop_variables(data["species_variables"]["dynamic"], new_H, new_W, nan_mode="zero", fix_dim=True)
        species_vars_wanted = {k: v for k, v in species_vars.items() if k in ["Distribution"]}
        # crop metadata dimensions
        latitude_var = torch.tensor(latitudes[:new_H])
        longitude_var = torch.tensor(longitudes[:new_W])
        # print("Latitues in Dataloder",latitude_var.shape, longitude_var.shape)
        # Calculate lead time
        dt_format = "%Y-%m-%dT%H:%M:%S"
        # Convert the two timestamps into datetime objects
        start = datetime.strptime(timestamps[0], dt_format)
        end = datetime.strptime(timestamps[1], dt_format)

        # Compute lead time in hours
        lead_time_hours = (end - start).total_seconds() / 3600.0
        # Fix the species distribution shapes
        dist = species_vars_wanted["Distribution"].permute(0, 3, 1, 2)  # => [T, C=22, H=153, W=152]
        species_vars_wanted["Distribution"] = dist[:, : self.num_species, :, :].to(torch.float32)  # Select only 3 species for now
        species_ids_wanted = species_list[: self.num_species]

        metadata = Metadata(
            latitudes=latitude_var,
            longitudes=longitude_var,
            timestamp=timestamps,
            lead_time=lead_time_hours,
            pressure_levels=pressure_levels,
            species_list=species_ids_wanted,
        )

        return Batch(
            batch_metadata=metadata,
            surface_variables=surface_vars,
            single_variables=single_vars,
            atmospheric_variables=atmospheric_vars,
            species_extinction_variables=species_ext_vars,
            land_variables=land_vars,
            agriculture_variables=agriculture_vars,
            forest_variables=forest_vars,
            species_variables=species_vars_wanted,
        )

    def __getitem__(self, idx):
        fpath_x = self.files[idx]
        fpath_y = self.files[idx + 1]
        if self.mode == "pretrain":
            x = self.load_and_process_files(fpath_x)
            y = self.load_and_process_files(fpath_y)
            return x, y
        else:  # finetune
            x = self.load_and_process_files(fpath_x)
            return x

    def scale_batch(self, batch: dict | Batch, direction: Literal["original", "scaled"] = "scaled"):
        """
        Scale a batch of data back or forward.
        """
        if not self.scaling_settings.enabled:
            # print("Scaling is not enabled in the configuration.")
            return batch
        convert_to_batch = False
        if isinstance(batch, Batch):
            # convert from NamedTuple to dict
            batch = batch._asdict()
            convert_to_batch = True
        _rescale_recursive(
            batch,
            self.scaling_statistics,
            dimensions_to_keep_by_key=dimensions_to_keep_by_key,
            mode=self.scaling_settings.mode,
            direction=direction,
        )
        if convert_to_batch:
            # convert back to NamedTuple
            batch = Batch(**batch)
        return batch
