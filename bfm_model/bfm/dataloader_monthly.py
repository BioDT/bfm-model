"""
Copyright (C) 2025 TNO, The Netherlands. All rights reserved.
"""

import os
from collections import namedtuple
from copy import deepcopy
from datetime import datetime, timedelta
from typing import Literal, Dict, List, Union, Any
from copy import deepcopy
import torch
from omegaconf.dictconfig import DictConfig
from torch.utils.data import DataLoader, Dataset, default_collate

from bfm_model.bfm.dataset_basics import *
from bfm_model.bfm.scaler import (
    _rescale_recursive,
    dimensions_to_keep_monthly,
    load_stats,
)
from bfm_model.bfm.utils import DictObj

# Namedtuple definitions
Batch = namedtuple(
    "Batch",
    [
        "batch_metadata",
        "surface_variables",
        "edaphic_variables",
        "atmospheric_variables",
        "climate_variables",
        "species_variables",
        "vegetation_variables",
        "land_variables",
        "agriculture_variables",
        "forest_variables",
        "redlist_variables",
        "misc_variables",
    ],
)

Metadata = namedtuple("Metadata", ["latitudes", "longitudes", "timestamp", "lead_time", "pressure_levels", "species_list"])

def normalize_keys(d: Dict[Union[int,str], torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Turn any int or mixed keys into strings.
    """
    return {str(k): v for k, v in d.items()}


def custom_collate(batch_list):
    """
    Custom collate function that handles a batch of Batches.
    Since each file corresponds to a single sample, and we define batch_size in the DataLoader,
    a batch from the DataLoader will be a list of Batch objects.
    """

    if not batch_list:
        return batch_list

    # If we're collating our Batch objects:
    if isinstance(batch_list[0], Batch):

        def collate_dicts(
            dicts: List[Dict[Union[int,str], torch.Tensor]]) -> Dict[Union[int,str], torch.Tensor]:
            out: Dict[Union[int,str], torch.Tensor] = {}
            keys = dicts[0].keys()
            for key in keys:
                values = [d[key] for d in dicts]
                if isinstance(values[0], torch.Tensor):
                    shapes = [tuple(v.shape) for v in values]
                    if len(set(shapes)) != 1:
                        # fail loud & clear
                        raise RuntimeError(
                            f"[collate] key={key!r} has mismatched shapes: {shapes}"
                        )
                    out[key] = torch.stack(values, dim=0)
                else:
                    # keep lists (e.g. timestamps) as-is
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
        edaphic_vars = collate_dicts([b.edaphic_variables for b in batch_list])
        atmospheric_vars = collate_dicts([b.atmospheric_variables for b in batch_list])
        climate_vars = collate_dicts([b.climate_variables for b in batch_list])
        species_vars = collate_dicts([b.species_variables for b in batch_list])
        vegetation_vars = collate_dicts([b.vegetation_variables for b in batch_list])
        land_vars = collate_dicts([b.land_variables for b in batch_list])
        agriculture_vars = collate_dicts([b.agriculture_variables for b in batch_list])
        forest_vars = collate_dicts([b.forest_variables for b in batch_list])
        redlist_vars = collate_dicts([b.redlist_variables for b in batch_list])
        misc_vars = collate_dicts([b.misc_variables for b in batch_list])

        return Batch(
            batch_metadata=metadata,
            surface_variables=surface_vars,
            edaphic_variables=edaphic_vars,
            atmospheric_variables=atmospheric_vars,
            climate_variables=climate_vars,
            species_variables=species_vars,
            vegetation_variables=vegetation_vars,
            land_variables=land_vars,
            agriculture_variables=agriculture_vars,
            forest_variables=forest_vars,
            redlist_variables=redlist_vars,
            misc_variables=misc_vars,
        )

    # Fallback for other types if encountered
    return default_collate(batch_list)


def crop_variables(variables, new_H, new_W, handle_nans=False, nan_mode="zero", fix_dim=False):
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
        "batch_metadata" {...},
        "surface_variables" {...},
        "edaphic_variables" {...},
        "atmospheric_variables" {...},
        "climate_variables" {...},
        "species_variables" {...},
        "vegetation_variables" {...},
        "land_variables" {...},
        "agriculture_variables" {...},
        "forest_variables" {...},
        "redlist_variables" {...}
        "misc_variables" {...},
    }
    """

    def __init__(
        self, data_dir: str, scaling_settings: DictConfig, num_species: int = 2, atmos_levels: list = [50], mode: str = "pretrain", model_patch_size: int = 4
    ):
        self.data_dir = data_dir
        self.num_species = num_species
        self.atmos_levels = atmos_levels
        self.mode = mode
        self.files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".pt")]
        self.files.sort()
        # print("Files sorted", self.files)
        self.scaling_settings = scaling_settings
        self.scaling_statistics = load_stats(scaling_settings.stats_path)
        self.model_patch_size = model_patch_size
        print(f"We scale the dataset {scaling_settings.enabled} with {scaling_settings.mode}")

    def __len__(self):
        if self.mode == "pretrain":
            return max(0, len(self.files) - 1)
        else:
            return len(self.files)

    def load_and_process_files(self, fpath: str):
        data = torch.load(fpath, map_location="cpu", weights_only=False)

        latitudes = data["batch_metadata"]["latitudes"]
        longitudes = data["batch_metadata"]["longitudes"]
        timestamps = data["batch_metadata"]["timestamp"]  # If it's a single sample, possibly a single timestamp or a list
        pressure_levels = data["batch_metadata"]["pressure_levels"]
        species_list = data["batch_metadata"]["species_list"]

        # Determine original spatial dimensions from metadata lists
        H = len(data["batch_metadata"]["latitudes"])
        W = len(data["batch_metadata"]["longitudes"])

        # crop dimensions to be divisible by patch size
        # patch_size = 4  # TODO make this configurable
        new_H = (H // self.model_patch_size) * self.model_patch_size
        new_W = (W // self.model_patch_size) * self.model_patch_size
        # print(f"Grid size: H x W {new_H}x{new_W}")
        # normalize or standardize variables
        data = self.scale_batch(data, direction="scaled")
        # print(data["species_variables"].values().max())

        surface_vars = crop_variables(data["surface_variables"], new_H, new_W)
        edaphic_vars = crop_variables(data["edaphic_variables"], new_H, new_W)
        atmospheric_vars = crop_variables(data["atmospheric_variables"], new_H, new_W)
        climate_vars = crop_variables(data["climate_variables"], new_H, new_W)
        species_vars = crop_variables(data["species_variables"], new_H, new_W)
        land_vars = crop_variables(data["land_variables"], new_H, new_W)
        agriculture_vars = crop_variables(data["agriculture_variables"], new_H, new_W)
        forest_vars = crop_variables(data["forest_variables"], new_H, new_W)
        # For NDVI Nans due to snow/clouds => putting 0 = Values close to zero (-0.1 to 0.1) generally correspond to barren areas of rock, sand, or snow
        vegetation_vars = crop_variables(data["vegetation_variables"], new_H, new_W, handle_nans=True, nan_mode="zero")
        redlist_vars = crop_variables(data["redlist_variables"], new_H, new_W)
        misc_vars = crop_variables(data["misc_variables"], new_H, new_W)
        # crop metadata dimensions
        latitude_var = torch.tensor(latitudes[:new_H])
        longitude_var = torch.tensor(longitudes[:new_W])
        # print("Latitues in Dataloder",latitude_var.shape, longitude_var.shape)
        # Calculate lead time
        dt_format = "%Y-%m-%d %H:%M:%S"
        # Convert the two timestamps into datetime objects
        start = datetime.strptime(timestamps[0], dt_format)
        end = datetime.strptime(timestamps[1], dt_format)
        # shape = atmospheric_vars["z"].shape
        # print(f"atmos shape before: {shape}")
        atmospheric_vars = extract_atmospheric_levels(atmospheric_vars, pressure_levels, self.atmos_levels, level_dim=1)
        # new_shape = atmospheric_vars["z"].shape
        # print(f"atmos shape after: {new_shape}")
        # print("Dataset species vars", species_vars)
        # Compute lead time in hours
        # lead_time_hours = (end - start).total_seconds() / 86400.0 # Its days now
        # lead_time_hours = 30
        lead_months = (end.year - start.year) * 12 + (end.month - start.month) + 1

        # Fix the species distribution shapes
        # print(f"lead time in months", lead_months, type(lead_months))
        metadata = Metadata(
            latitudes=latitude_var,
            longitudes=longitude_var,
            timestamp=timestamps,
            lead_time=lead_months,
            pressure_levels=pressure_levels,
            species_list=species_list,
        )

        species_vars = normalize_keys(species_vars)

        return Batch(
            batch_metadata=metadata,
            surface_variables=surface_vars,
            edaphic_variables=edaphic_vars,
            atmospheric_variables=atmospheric_vars,
            climate_variables=climate_vars,
            species_variables=species_vars,
            vegetation_variables=vegetation_vars,
            land_variables=land_vars,
            agriculture_variables=agriculture_vars,
            forest_variables=forest_vars,
            redlist_variables=redlist_vars,
            misc_variables=misc_vars,
        )

    def __getitem__(self, idx):
        fpath_x = self.files[idx]
        if self.mode == "pretrain":
            fpath_y = self.files[idx + 1]
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
            dimensions_to_keep_by_key=dimensions_to_keep_monthly,
            mode=self.scaling_settings.mode,
            direction=direction,
        )
        if convert_to_batch:
            # convert back to NamedTuple
            batch = Batch(**batch)
        return batch


def extract_atmospheric_levels(
    atmos_vars: Dict[str, torch.Tensor],
    all_levels: List[int],
    desired_levels: List[int],
    level_dim: int = 2,
) -> Dict[str, torch.Tensor]:
    """
    Given a dict mapping variable names → tensors that have a 'levels'
    axis at position `level_dim`, return a new dict where each tensor
    has been sliced to KEEP ONLY those levels in `desired_levels`.

    - atmos_vars: e.g. {"temperature": Tensor[B,C,D,H,W], "humidity": …}
    - all_levels: the full list of level values (length D)
    - desired_levels: the subset of levels you want to extract, e.g. [50, 60, 100]
    - level_dim: the tensor dimension index where levels live (default 2)

    Returns a new dict with the same keys, but each Tensor is now
    shape (..., len(desired_levels), ...).
    """
    # 1) map desired level *values* → their integer *indices* in all_levels
    idxs: List[int] = []
    for lvl in desired_levels:
        try:
            idxs.append(all_levels.index(lvl))
        except ValueError:
            raise ValueError(f"Level {lvl!r} not found in available levels {all_levels}")

    # 2) build a 1D index tensor on the same device as your data
    device = next(iter(atmos_vars.values())).device
    idx_tensor = torch.tensor(idxs, dtype=torch.long, device=device)

    # 3) slice each variable via index_select
    filtered: Dict[str, torch.Tensor] = {}
    for name, tensor in atmos_vars.items():
        filtered[name] = tensor.index_select(dim=level_dim, index=idx_tensor)

    return filtered


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


scalling_dict = {
    "enabled": False,
    "stats_path": "/home/atrantas/bfm-model/batch_statistics/all_batches_stats.json",
    "mode": "normalize",
}
scaling_object = DictObj(scalling_dict)


def test_dataset_and_dataloader(data_dir):
    """
    Test function to inspect correctness.
    Print distinctive info from a single batch.
    """
    example_model_patch_size = 4 
    dataset = LargeClimateDataset(data_dir, num_species=10, scaling_settings=scaling_object, example_model_patch_size = 4)
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
    print(batch)
    stats = compute_batch_statistics(batch)
    print("=== Variable Statistics ===")
    for group_name, group_dict in stats.items():
        print(f"\nGroup: {group_name}")
        for var_name, var_stats in group_dict.items():
            if "min" in var_stats:
                print(
                    f"{var_name} => Min: {var_stats['min']}, Max: {var_stats['max']}, Mean: {var_stats['mean']}, Std: {var_stats['std']}"
                )
                print(
                    f"NaN: {var_stats['nan_count']}, Inf: {var_stats['inf_count']}, shape: {var_stats['shape']}, dtype: {var_stats['dtype']}"
                )
            else:
                print(f"{var_name} => {var_stats}")

    print("\nTest completed successfully.")


def detach_batch(batch: Batch) -> Batch:
    """
    Return a copy of `batch` where every torch.Tensor is
    detached, cloned, and moved to CPU, so it can be torch.save().
    """
    md = batch.batch_metadata
    # lead_time logic unchanged from before...
    lt = md.lead_time
    if isinstance(lt, torch.Tensor):
        arr = lt.detach().cpu().numpy()
        if arr.ndim == 0:
            new_lt = float(arr)
        else:
            new_lt = arr.tolist()
    elif isinstance(lt, list):
        new_lt = [float(x) for x in lt]
    elif isinstance(lt, timedelta):
        new_lt = arr.total_seconds() / 3600.0
    else:
        new_lt = float(lt)

    new_md = md._replace(lead_time=new_lt)

    def _detach_and_clone(grp):
        if grp is None:
            return None
        out = {}
        for k, v in grp.items():
            # detach from graph, clone, move to CPU
            t = v.detach().clone().cpu()
            out[k] = t
        return out

    return Batch(
        batch_metadata=new_md,
        surface_variables=_detach_and_clone(batch.surface_variables),
        edaphic_variables=_detach_and_clone(batch.edaphic_variables),
        atmospheric_variables=_detach_and_clone(batch.atmospheric_variables),
        climate_variables=_detach_and_clone(batch.climate_variables),
        species_variables=_detach_and_clone(batch.species_variables),
        vegetation_variables=_detach_and_clone(batch.vegetation_variables),
        land_variables=_detach_and_clone(batch.land_variables),
        agriculture_variables=_detach_and_clone(batch.agriculture_variables),
        forest_variables=_detach_and_clone(batch.forest_variables),
        redlist_variables=_detach_and_clone(batch.redlist_variables),
        misc_variables=_detach_and_clone(batch.misc_variables)
    )


def detach_preds(preds: dict) -> dict:
    """
    Detach, clone, cpu move every tensor in the prediction dict.
    """
    out = {}
    for g, vars_ in preds.items():
        out[g] = {v: t.detach().clone().cpu() for v, t in vars_.items()}
    return out


def detach_graph_batch(batch: Batch) -> Batch:
    """Detach every tensor in the Batch (break graph), but keep device."""
    md = batch.batch_metadata
    new_md = md  # timestamps/lead_time floats are unchanged

    def det_grp(grp):
        if grp is None:
            return None
        return {k: v.detach().cpu() for k, v in grp.items()}

    return Batch(
        batch_metadata=new_md,
        surface_variables=det_grp(batch.surface_variables),
        edaphic_variables=det_grp(batch.edaphic_variables),
        atmospheric_variables=det_grp(batch.atmospheric_variables),
        climate_variables=det_grp(batch.climate_variables),
        species_variables=det_grp(batch.species_variables),
        vegetation_variables=det_grp(batch.vegetation_variables),
        land_variables=det_grp(batch.land_variables),
        agriculture_variables=det_grp(batch.agriculture_variables),
        forest_variables=det_grp(batch.forest_variables),
        redlist_variables=det_grp(batch.redlist_variables),
        misc_variables=det_grp(batch.misc_variables)
    )


def batch_to_device(batch: Batch, device: torch.device) -> Batch:
    """
    Recursively move every tensor in the Batch to `device`.
    Nontensor fields (timestamps, lists, floats) are left unchanged.
    """
    md = batch.batch_metadata
    lat = md.latitudes.to(device)
    lon = md.longitudes.to(device)
    new_md = Metadata(
        latitudes=lat,
        longitudes=lon,
        timestamp=md.timestamp,
        lead_time=md.lead_time,
        pressure_levels=md.pressure_levels,  # if tensor, also .to(device)
        species_list=md.species_list,  # list of ints, leave as is
    )

    def move_group(grp):
        if grp is None:
            return None
        return {k: v.to(device) for k, v in grp.items()}

    return Batch(
        batch_metadata=new_md,
        surface_variables=move_group(batch.surface_variables),
        single_variables=move_group(batch.single_variables),
        species_variables=move_group(batch.species_variables),
        atmospheric_variables=move_group(batch.atmospheric_variables),
        species_extinction_variables=move_group(batch.species_extinction_variables),
        land_variables=move_group(batch.land_variables),
        agriculture_variables=move_group(batch.agriculture_variables),
        forest_variables=move_group(batch.forest_variables),
    )


def debug_batch_devices(batch: Batch, prefix: str = ""):
    """
    Print out the set of devices present in this Batch.
    """
    devices = set()

    # metadata
    md = batch.batch_metadata
    for name in ("latitudes", "longitudes", "pressure_levels"):
        t = getattr(md, name)
        if isinstance(t, torch.Tensor):
            devices.add((f"{prefix}.batch_metadata.{name}", t.device))

    # variable groups
    for group_name in [
        "surface_variables",
        "single_variables",
        "atmospheric_variables",
        "species_extinction_variables",
        "land_variables",
        "agriculture_variables",
        "forest_variables",
        "species_variables",
    ]:
        grp = getattr(batch, group_name)
        if grp is None:
            continue
        for var_name, tensor in grp.items():
            devices.add((f"{prefix}.{group_name}.{var_name}", tensor.device))

    # lead_time (if it's still a tensor)
    lt = md.lead_time
    if isinstance(lt, torch.Tensor):
        devices.add((f"{prefix}.batch_metadata.lead_time", lt.device))

    # print a sorted list
    print("==== Device check:", prefix, "====")
    for k, dev in sorted(devices):
        print(f"  {k:40s} → {dev}")
    print("======================================")


def detach_output_dict(d):
    """
    Recursively detach & clone any torch.Tensor or Batch found in a dict.
    Leaves other values (ints, floats, lists, etc.) untouched.
    Returns a new dict.
    """
    out = {}
    for k, v in d.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.detach().clone().cpu()
        elif isinstance(v, Batch):
            out[k] = detach_batch(v)
        elif isinstance(v, dict):
            out[k] = detach_output_dict(v)
        elif isinstance(v, (list, tuple)):
            proc = []
            for elem in v:
                if isinstance(elem, torch.Tensor):
                    proc.append(elem.detach().clone().cpu())
                elif isinstance(elem, Batch):
                    proc.append(detach_batch(elem))
                elif isinstance(elem, dict):
                    proc.append(detach_output_dict(elem))
                else:
                    proc.append(deepcopy(elem))
            out[k] = type(v)(proc)
        else:
            # leave alone
            out[k] = deepcopy(v)
    return out

#V1> Works
def _convert(obj: Any, move_cpu: bool = True, target_dtype: torch.dtype = torch.float32) -> Any:
    """Recursively convert to plain-Python containers; move tensors to CPU."""
    if isinstance(obj, torch.Tensor):
        t = obj.detach()
        if move_cpu:
            t = t.cpu()
        if t.dtype != target_dtype:
            t = t.to(dtype=target_dtype)
        return t

    if hasattr(obj, "_fields") or hasattr(obj, "__dict__"):
        attrs = obj._asdict() if hasattr(obj, "_asdict") else vars(obj)
        return {k: _convert(v, move_cpu) for k, v in attrs.items()}

    if isinstance(obj, dict):
        return {k: _convert(v, move_cpu) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        return type(obj)(_convert(v, move_cpu) for v in obj)

    return obj


# @hydra.main(version_base=None, config_path="configs", config_name="train_config")
# def main(cfg):
#     data_path = "/projects/prjs1134/data/projects/biodt/storage/monthly_batches/batches"
#     dataset = LargeClimateDataset(
#         data_dir=data_path, scaling_settings=cfg.data.scaling, num_species=2
#     )
#     print(len(dataset), dataset[0])

# if __name__ == "__main__":
#     main()


if __name__ == "__main__":
    data_path = "/home/atrantas/bfm-model/data/"
    test_dataset_and_dataloader(data_path)
