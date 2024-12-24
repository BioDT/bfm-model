import torch
from torch.utils.data import Dataset, DataLoader, default_collate
from datetime import datetime
import os
from collections import namedtuple
from src.bfm.src.dataset_basics import *

# Namedtuple definitions
Batch = namedtuple("Batch", [
    "batch_metadata",
    "surface_variables",
    "single_variables",
    "atmospheric_variables",
    "species_extinction_variables",
    "land_variables",
    "agriculture_variables",
    "forest_variables",
])

Metadata = namedtuple("Metadata", [
    "latitudes",
    "longitudes",
    "timestamp",
    "lead_time",
    "pressure_levels",
])

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
            pressure_levels=pressure_levels
        )

        surface_vars = collate_dicts([b.surface_variables for b in batch_list])
        single_vars = collate_dicts([b.single_variables for b in batch_list])
        atmospheric_vars = collate_dicts([b.atmospheric_variables for b in batch_list])
        species_ext_vars = collate_dicts([b.species_extinction_variables for b in batch_list])
        land_vars = collate_dicts([b.land_variables for b in batch_list])
        agriculture_vars = collate_dicts([b.agriculture_variables for b in batch_list])
        forest_vars = collate_dicts([b.forest_variables for b in batch_list])

        return Batch(
            batch_metadata=metadata,
            surface_variables=surface_vars,
            single_variables=single_vars,
            atmospheric_variables=atmospheric_vars,
            species_extinction_variables=species_ext_vars,
            land_variables=land_vars,
            agriculture_variables=agriculture_vars,
            forest_variables=forest_vars
        )

    # Fallback for other types if encountered
    return default_collate(batch_list)


def crop_variables(variables, new_H, new_W):
    """
    Crop and clean variables to specified dimensions, handling NaN and Inf values.

    Args:
        variables (dict): Dictionary of variable tensors to process
        new_H (int): Target height dimension
        new_W (int): Target width dimension

    Returns:
        dict: Processed variables with cleaned and cropped tensors
    """
    processed_vars = {}
    for k, v in variables.items():
        # crop dimensions
        cropped = v[..., :new_H, :new_W]

        # infinities first
        inf_mask = torch.isinf(cropped)
        inf_count = inf_mask.sum().item()
        if inf_count > 0:
            print(f"\nHandling Inf values in {k}:")
            print(f"Inf count: {inf_count}")
            valid_values = cropped[~inf_mask & ~torch.isnan(cropped)]
            if len(valid_values) > 0:
                max_val = valid_values.max().item()
                min_val = valid_values.min().item()
                cropped = torch.clip(cropped, min_val, max_val)
            else:
                cropped = torch.clip(cropped, -1e6, 1e6)

        # handle NaNs
        nan_mask = torch.isnan(cropped)
        nan_count = nan_mask.sum().item()
        if nan_count > 0:
            print(f"\nHandling NaN values in {k}:")
            print(f"Shape: {cropped.shape}")
            print(f"Total NaN count: {nan_count}")
            valid_values = cropped[~nan_mask & ~torch.isinf(cropped)]
            if len(valid_values) > 0:
                mean_val = valid_values.mean().item()
                std_val = valid_values.std().item()
                # use mean +- 2*std as clipping bounds
                clip_min = mean_val - 2 * std_val
                clip_max = mean_val + 2 * std_val
                # replace NaNs with clipped mean
                cropped = torch.nan_to_num(cropped, nan=mean_val)
                # TODO This is nasty, may be using a lot of memory
                # TODO Check the advantages and why the data are in this regime
                cropped = cropped.to(torch.float32)
                cropped = torch.clip(cropped, clip_min, clip_max)

            else:
                cropped = torch.nan_to_num(cropped, nan=0.0)
                cropped = torch.clip(cropped, -1.0, 1.0)

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
        "atmospheric_variables": {...},
        "species_extinction_variables": {...},
        "land_variables": {...},
        "agriculture_variables": {...},
        "forest_variables": {...}
    }
    """

    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.pt')]
        self.files.sort()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fpath = self.files[idx]
        data = torch.load(fpath, map_location='cpu')

        latitudes = data["batch_metadata"]["latitudes"]
        longitudes = data["batch_metadata"]["longitudes"]
        timestamps = data["batch_metadata"]["timestamp"]  # If it's a single sample, possibly a single timestamp or a list
        pressure_levels = data["batch_metadata"]["pressure_levels"]

        # Determine original spatial dimensions from metadata lists
        H = len(data["batch_metadata"]["latitudes"])
        W = len(data["batch_metadata"]["longitudes"])

        # crop dimensions to be divisible by patch size
        patch_size = 4 # TODO make this configurable
        new_H = (H // patch_size) * patch_size
        new_W = (W // patch_size) * patch_size

        surface_vars = crop_variables(data["surface_variables"], new_H, new_W)
        single_vars = crop_variables(data["single_variables"], new_H, new_W)
        atmospheric_vars = crop_variables(data["atmospheric_variables"], new_H, new_W)
        species_ext_vars = crop_variables(data["species_extinction_variables"], new_H, new_W)
        land_vars = crop_variables(data["land_variables"], new_H, new_W)
        agriculture_vars = crop_variables(data["agriculture_variables"], new_H, new_W)
        forest_vars = crop_variables(data["forest_variables"], new_H, new_W)

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


        metadata = Metadata(
            latitudes=latitude_var,
            longitudes=longitude_var,
            timestamp=timestamps,
            lead_time=lead_time_hours,
            pressure_levels=pressure_levels
        )

        return Batch(
            batch_metadata=metadata,
            surface_variables=surface_vars,
            single_variables=single_vars,
            atmospheric_variables=atmospheric_vars,
            species_extinction_variables=species_ext_vars,
            land_variables=land_vars,
            agriculture_variables=agriculture_vars,
            forest_variables=forest_vars
        )


def test_dataset_and_dataloader(data_dir):
    """
    Test function to inspect correctness.
    Print distinctive info from a single batch.
    """
    dataset = LargeClimateDataset(data_dir)
    dataloader = DataLoader(
        dataset,
        batch_size=1,        # Fetch two samples for testing
        num_workers=0,       # For debugging, keep workers=0 to avoid async complexity
        pin_memory=False,
        collate_fn=custom_collate,
        shuffle=False
    )

    batch = next(iter(dataloader))

    # print_batch_shapes(batch)
    # print_nan_counts(batch)

    print("=== Test Dataloader Output ===")
    print("Batch Metadata:")
    print("  Latitudes length:", len(batch.batch_metadata.latitudes))
    print("  Longitudes length:", len(batch.batch_metadata.longitudes))
    print("  Timestamps:", batch.batch_metadata.timestamp)
    print(" Datatype of timestamps", type(batch.batch_metadata.timestamp), type(batch.batch_metadata.timestamp[0]))
    print("  Pressure Levels:", batch.batch_metadata.pressure_levels)


    a_time_mod =[[
            datetime.strptime(t_str, "%Y-%m-%dT%H:%M:%S").timestamp() / 3600.0
            for t_str in time_list
        ]
        for time_list in [batch.batch_metadata.timestamp]
    ]

    print(a_time_mod)
    timestamp_tensor = torch.tensor(a_time_mod[0])
    print("timestamp tensor", timestamp_tensor, timestamp_tensor.shape)

    if batch.surface_variables:
        var_name, var_data = next(iter(batch.surface_variables.items()))
        # var_data might be a tensor or a list of tensors if batch_size > 1
        if isinstance(var_data, torch.Tensor):
            print(f"\nSurface variable '{var_name}' shape:", var_data.shape)
        else:
            print(f"\nSurface variable '{var_name}' is a list of length {len(var_data)}")

    if batch.atmospheric_variables:
        var_name, var_data = next(iter(batch.atmospheric_variables.items()))
        if isinstance(var_data, torch.Tensor):
            print(f"Atmospheric variable '{var_name}' shape:", var_data.shape)
        else:
            print(f"Atmospheric variable '{var_name}' is a list of length {len(var_data)}")

    if batch.species_extinction_variables:
        var_name, var_data = next(iter(batch.species_extinction_variables.items()))
        if isinstance(var_data, torch.Tensor):
            print(f"Species extinction variable '{var_name}' shape:", var_data.shape)

    if batch.land_variables:
        var_name, var_data = next(iter(batch.land_variables.items()))
        if isinstance(var_data, torch.Tensor):
            print(f"Land variable '{var_name}' shape:", var_data.shape)

    if batch.forest_variables:
        var_name, var_data = next(iter(batch.forest_variables.items()))
        if isinstance(var_data, torch.Tensor):
            print(f"Argiculture variable '{var_name}' shape:", var_data.shape)

    print("\nTest completed successfully.")


if __name__ == "__main__":
    data_dir = "data/"  # Replace this with the actual directory path
    test_dataset_and_dataloader(data_dir)
