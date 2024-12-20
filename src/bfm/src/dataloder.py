import torch
from torch.utils.data import Dataset, DataLoader, default_collate
from datetime import datetime
import os
from collections import namedtuple

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
        timestamps_all = [b.batch_metadata.timestamp for b in batch_list]
        # If each sample has a single timestamp, we get a list of timestamps:
        timestamps = [ts[0] for ts in timestamps_all]

        metadata = Metadata(
            latitudes=latitudes,
            longitudes=longitudes,
            timestamp=timestamps,  # a list of timestamps, one per sample
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
        timestamps = [data["batch_metadata"]["timestamp"]]  # If it's a single sample, possibly a single timestamp or a list
        pressure_levels = data["batch_metadata"]["pressure_levels"]

        metadata = Metadata(
            latitudes=latitudes,
            longitudes=longitudes,
            timestamp=timestamps,
            pressure_levels=pressure_levels
        )

        def index_dict(d):
            # Each file is a single sample, no indexing needed
            # The data should already be a single sample.
            # If the variables are still higher-dimensional, assume the first dimension is sample, we index [0].
            # If there's no extra dimension, just return them as is.
            # Let's try indexing in case dimension 0 corresponds to sample:
            for k, v in d.items():
                if isinstance(v, torch.Tensor) and v.dim() > 0:
                    # Attempt to index the first dimension if it represents a batch dimension
                    # If the file truly contains only one sample (no batch dimension), no indexing is required.
                    # We'll try without indexing first. If needed, modify here.
                    pass
            return d

        # Since each file has a single sample, we assume no indexing is needed.
        # Just return them as is.
        surface_vars = index_dict(data["surface_variables"])
        single_vars = index_dict(data["single_variables"])
        atmospheric_vars = index_dict(data["atmospheric_variables"])
        species_ext_vars = index_dict(data["species_extinction_variables"])
        land_vars = index_dict(data["land_variables"])
        agriculture_vars = index_dict(data["agriculture_variables"])
        forest_vars = index_dict(data["forest_variables"])

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
        batch_size=4,        # Fetch two samples for testing
        num_workers=0,       # For debugging, keep workers=0 to avoid async complexity
        pin_memory=False,
        collate_fn=custom_collate,
        shuffle=False
    )

    batch = next(iter(dataloader))

    print("=== Test Dataloader Output ===")
    print("Batch Metadata:")
    print("  Latitudes length:", len(batch.batch_metadata.latitudes))
    print("  Longitudes length:", len(batch.batch_metadata.longitudes))
    print("  Timestamps:", batch.batch_metadata.timestamp)
    print("  Pressure Levels:", batch.batch_metadata.pressure_levels)

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


# if __name__ == "__main__":
#     data_dir = "data/"  # Replace this with the actual directory path
#     test_dataset_and_dataloader(data_dir)
