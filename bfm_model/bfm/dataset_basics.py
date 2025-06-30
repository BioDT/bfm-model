"""
Copyright 2025 (C) TNO. Licensed under the MIT license.
"""

import os
from datetime import datetime

import torch


def load_batches(batch_directory: str, device: torch.device = torch.device("cpu")) -> list:
    """
    Load all saved DataBatch objects from the specified directory.

    Args:
        batch_directory (str): The directory where the batch .pt files are stored.

    Returns:
        list: A list of DataBatch objects loaded from the directory.
    """
    batches = []

    # Iterate through all files in the batch directory
    for batch_file in sorted(os.listdir(batch_directory)):
        if batch_file.endswith(".pt"):
            batch_path = os.path.join(batch_directory, batch_file)
            print(f"Loading batch: {batch_path}")

            # Load the batch using torch.load()
            batch = torch.load(batch_path, map_location=device)
        batches.append(batch)

    return batches


def print_batch_shapes(batch: dict) -> None:
    """
    Print the shapes of all variables in the DataBatch (now a dictionary).

    Args:
        batch (dict): The batch dictionary containing climate and species data.
    """
    print("Variable Shapes:")
    print("\nMetadata:")
    print(f"Latitudes shape: {len(batch['batch_metadata']['latitudes'])}")
    print(f"Longitudes shape: {len(batch['batch_metadata']['longitudes'])}")
    print(f"Timestamps length: {len(batch['batch_metadata']['timestamp'])}")
    print(f"Pressure Levels length: {len(batch['batch_metadata']['pressure_levels'])}")

    print("\nMetadata Contents:")
    print(f"Latitudes: {batch['batch_metadata']['latitudes']}")
    print(f"Longitudes: {batch['batch_metadata']['longitudes']}")
    print(f"Timestamps: {batch['batch_metadata']['timestamp']}")
    print(f"Pressure Levels: {batch['batch_metadata']['pressure_levels']}")

    print("\nSurface Variables:")
    for var_name, var_data in batch["surface_variables"].items():
        print(f"{var_name}: {var_data.shape}")

    print("\nSingle Variables:")
    for var_name, var_data in batch["single_variables"].items():
        print(f"{var_name}: {var_data.shape}")

    print("\nAtmospheric Variables:")
    for var_name, var_data in batch["atmospheric_variables"].items():
        print(f"{var_name}: {var_data.shape}")

    print("\nSpecies Extinction Variables:")
    for var_name, var_data in batch["species_extinction_variables"].items():
        print(f"{var_name}: {var_data.shape}")

    print("\nLand Variables:")
    for var_name, var_data in batch["land_variables"].items():
        print(f"{var_name}: {var_data.shape}")

    print("\nAgriculture Variables:")
    for var_name, var_data in batch["agriculture_variables"].items():
        print(f"{var_name}: {var_data.shape}")

    print("\nForest Variables:")
    for var_name, var_data in batch["forest_variables"].items():
        print(f"{var_name}: {var_data.shape}")


def print_batch_variables(batch: dict, output_file=None) -> None:
    """
    Print all the variable values from the given DataBatch (now a dictionary).

    Args:
        batch (dict): The batch dictionary containing climate and species data.
        output_file (str, optional): Path to the output file. If None, generates a timestamped filename.
    """
    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"batch_variables_{timestamp}.txt"

    with open(output_file, "w") as f:
        # First write metadata
        f.write("Metadata:\n")
        f.write(f"Latitudes: {batch['batch_metadata']['latitudes']}\n")
        f.write(f"Longitudes: {batch['batch_metadata']['longitudes']}\n")
        f.write(f"Timestamps: {batch['batch_metadata']['timestamp']}\n")
        f.write(f"Pressure Levels: {batch['batch_metadata']['pressure_levels']}\n\n")

        f.write("Surface Variables:\n")
        for var_name, var_data in batch["surface_variables"].items():
            f.write(f"{var_name}:\n")
            f.write(f"Shape: {var_data.shape}\n")
            f.write(f"Data: {var_data}\n\n")

        # Repeat for other variable categories...
        variable_categories = [
            ("Single Variables", batch["single_variables"]),
            ("Atmospheric Variables", batch["atmospheric_variables"]),
            ("Species Extinction Variables", batch["species_extinction_variables"]),
            ("Land Variables", batch["land_variables"]),
            ("Agriculture Variables", batch["agriculture_variables"]),
            ("Forest Variables", batch["forest_variables"]),
        ]

        for category_name, category_dict in variable_categories:
            f.write(f"\n{category_name}:\n")
            for var_name, var_data in category_dict.items():
                f.write(f"{var_name}:\n")
                f.write(f"Shape: {var_data.shape}\n")
                f.write(f"Data: {var_data}\n\n")

    print(f"Batch variables have been written to: {output_file}")


def print_nan_counts(batch: dict) -> None:
    def analyze_nan(name: str, data) -> None:
        # If data is a list, try converting it to a tensor.
        if isinstance(data, list):
            try:
                data = torch.tensor(data, dtype=torch.float32)
            except ValueError:
                print(f"{name} is a list but cannot be converted to a numeric tensor. Skipping NaN analysis.")
                return

        if isinstance(data, torch.Tensor):
            total_values = data.numel()
            nan_count = torch.isnan(data).sum().item()
            nan_percentage = (nan_count / total_values) * 100 if total_values > 0 else 0.0
            if nan_count > 0:
                print(f"{name}")
                print(f"Total values: {total_values}")
                print(f"NaN count: {nan_count}")
                print(f"NaN percentage: {nan_percentage:.2f}%\n")
            else:
                print(f"{name} has no NaNs")
        else:
            # If data is neither a list nor a tensor, just print a message.
            print(f"{name} is not a tensor and not a list. Skipping NaN analysis.")

    print("\nMetadata Tensors:")
    analyze_nan("Latitudes", batch["batch_metadata"]["latitudes"])
    analyze_nan("Longitudes", batch["batch_metadata"]["longitudes"])

    print("\nSurface Variables:")
    for var_name, var_data in batch["surface_variables"].items():
        analyze_nan(var_name, var_data)

    print("\nSingle Variables:")
    for var_name, var_data in batch["single_variables"].items():
        analyze_nan(var_name, var_data)

    print("\nAtmospheric Variables:")
    for var_name, var_data in batch["atmospheric_variables"].items():
        analyze_nan(var_name, var_data)

    print("\nSpecies Extinction Variables:")
    for var_name, var_data in batch["species_extinction_variables"].items():
        analyze_nan(var_name, var_data)

    print("\nLand Variables:")
    for var_name, var_data in batch["land_variables"].items():
        analyze_nan(var_name, var_data)

    print("\nAgriculture Variables:")
    for var_name, var_data in batch["agriculture_variables"].items():
        analyze_nan(var_name, var_data)

    print("\nForest Variables:")
    for var_name, var_data in batch["forest_variables"].items():
        analyze_nan(var_name, var_data)


# Uncomment to check the batches
# batches = load_batches("data/")
# print_batch_shapes(batches[0])
# print_nan_counts(batches[0])

# # print all the attributes a batch has
# print(batches[0].__dict__.keys())

# print_batch_variables(batches[0])
