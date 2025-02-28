import os
from collections import namedtuple
from datetime import datetime
from typing import Literal, NamedTuple

import torch
from omegaconf.dictconfig import DictConfig
from torch.utils.data import DataLoader, Dataset, default_collate

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
        # 1) Handle infinities
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

        # 2) Handle NaNs if requested
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

    def __init__(self, data_dir: str, scaling_settings: DictConfig, num_species: int = 2):
        self.data_dir = data_dir
        self.num_species = num_species
        self.files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".pt")]
        self.files.sort()
        self.scaling_settings = scaling_settings
        self.scaling_statistics = load_stats(scaling_settings.stats_path)
        print(f"We scale the dataset {scaling_settings.enabled} with {scaling_settings.mode}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fpath = self.files[idx]
        data = torch.load(fpath, map_location="cpu", weights_only=True)

        latitudes = data["batch_metadata"]["latitudes"]
        longitudes = data["batch_metadata"]["longitudes"]
        timestamps = data["batch_metadata"]["timestamp"]  # If it's a single sample, possibly a single timestamp or a list
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


def compute_variable_statistics(tensor: torch.Tensor) -> dict:
    """
    Compute basic statistics for a given tensor:
    - min, max, mean, std
    - nan_count, inf_count
    - optional: shape, dtype

    Args:
        tensor (torch.Tensor): The tensor to analyze

    Returns:
        dict: A dictionary of computed statistics
    """
    stats = {}
    # Ensure float to avoid errors with integer types
    # (Optional step; .float() is typically safe if you want stats in float precision.)
    t = tensor.float()

    stats["min"] = float(t.min().item())
    stats["max"] = float(t.max().item())
    stats["mean"] = float(t.mean().item())
    stats["std"] = float(t.std().item())

    # Count special values
    stats["nan_count"] = int(torch.isnan(t).sum().item())
    stats["inf_count"] = int(torch.isinf(t).sum().item())

    # (Optional) Add shape and dtype for reference
    stats["shape"] = list(tensor.shape)
    stats["dtype"] = str(tensor.dtype)

    return stats


def compute_batch_statistics(batch: Batch) -> dict:
    """
    Compute statistics for each sub-dictionary in the batch object.
    The batch object has the following structure:
        batch_metadata,
        surface_variables,
        single_variables,
        atmospheric_variables,
        species_extinction_variables,
        land_variables,
        agriculture_variables,
        forest_variables

    Each of those is a dict of name -> tensor, or a namedtuple for metadata.
    We skip metadata in this function and focus on actual variable tensors.

    Args:
        batch (Batch): A namedtuple containing sub-dicts of variables.

    Returns:
        dict: A nested dictionary of stats for each variable group.
    """
    stats_result = {}

    # We'll define a helper to process a sub-dictionary
    def process_var_dict(var_dict: dict, group_name: str):
        group_stats = {}
        for var_name, var_data in var_dict.items():
            if isinstance(var_data, torch.Tensor):
                # Compute stats
                group_stats[var_name] = compute_variable_statistics(var_data)
            else:
                # Non-tensor data (e.g. lists). Skip or handle differently if needed
                group_stats[var_name] = {"note": "Not a tensor, skipping stats."}
        return group_stats

    # For each field in the Batch namedtuple that is a dictionary, compute stats
    stats_result["surface_variables"] = process_var_dict(batch.surface_variables, "surface_variables")
    stats_result["single_variables"] = process_var_dict(batch.single_variables, "single_variables")
    stats_result["atmospheric_variables"] = process_var_dict(batch.atmospheric_variables, "atmospheric_variables")
    stats_result["species_extinction_variables"] = process_var_dict(
        batch.species_extinction_variables, "species_extinction_variables"
    )
    stats_result["land_variables"] = process_var_dict(batch.land_variables, "land_variables")
    stats_result["agriculture_variables"] = process_var_dict(batch.agriculture_variables, "agriculture_variables")
    stats_result["forest_variables"] = process_var_dict(batch.forest_variables, "forest_variables")
    stats_result["species_variables"] = process_var_dict(batch.species_variables, "species_variables")

    return stats_result


def test_dataset_and_dataloader(data_dir):
    """
    Test function to inspect correctness.
    Print distinctive info from a single batch.
    """
    dataset = LargeClimateDataset(data_dir, num_species=10)
    dataloader = DataLoader(
        dataset,
        batch_size=1,  # Fetch two samples for testing
        num_workers=0,  # For debugging, keep workers=0 to avoid async complexity
        pin_memory=False,
        collate_fn=custom_collate,
        shuffle=False,
    )

    batch = next(iter(dataloader))
    # Now compute stats
    stats = compute_batch_statistics(batch)
    print("=== Variable Statistics ===")
    for group_name, group_dict in stats.items():
        print(f"\nGroup: {group_name}")
        for var_name, var_stats in group_dict.items():
            if "min" in var_stats:
                print(
                    f"  {var_name} => Min: {var_stats['min']}, Max: {var_stats['max']}, Mean: {var_stats['mean']}, Std: {var_stats['std']}"
                )
                print(
                    f"     NaN: {var_stats['nan_count']}, Inf: {var_stats['inf_count']}, shape: {var_stats['shape']}, dtype: {var_stats['dtype']}"
                )
            else:
                print(f"  {var_name} => {var_stats}")

    print("\nTest completed successfully.")


if __name__ == "__main__":
    data_dir = "data_small/batches/"  # Replace this with the actual directory path
    test_dataset_and_dataloader(data_dir)
