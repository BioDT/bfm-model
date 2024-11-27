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


def print_batch_shapes(batch) -> None:
    """
    Print the shapes of all variables in the DataBatch.

    Args:
        batch (DataBatch): The batch object containing climate and species data.
    """
    print("Variable Shapes:")
    print("\nMetadata:")
    print(f"Latitudes shape: {batch.batch_metadata.latitudes.shape}")
    print(f"Longitudes shape: {batch.batch_metadata.longitudes.shape}")
    print(f"Timestamps length: {len(batch.batch_metadata.timestamp)}")
    print(f"Pressure Levels length: {len(batch.batch_metadata.pressure_levels)}")

    print("\nMetadata Contents:")
    print(f"Latitudes: {batch.batch_metadata.latitudes.tolist()}")
    print(f"Longitudes: {batch.batch_metadata.longitudes.tolist()}")
    print(f"Timestamps: {batch.batch_metadata.timestamp}")
    print(f"Pressure Levels: {batch.batch_metadata.pressure_levels}")

    print("\nSurface Variables:")
    for var_name, var_data in batch.surface_variables.items():
        print(f"{var_name}: {var_data.shape}")

    print("\nSingle Variables:")
    for var_name, var_data in batch.single_variables.items():
        print(f"{var_name}: {var_data.shape}")

    print("\nAtmospheric Variables:")
    for var_name, var_data in batch.atmospheric_variables.items():
        print(f"{var_name}: {var_data.shape}")

    print("\nSpecies Extinction Variables:")
    for var_name, var_data in batch.species_extinction_variables.items():
        print(f"{var_name}: {var_data.shape}")

    print("\nLand Variables:")
    for var_name, var_data in batch.land_variables.items():
        print(f"{var_name}: {var_data.shape}")

    print("\nAgriculture Variables:")
    for var_name, var_data in batch.agriculture_variables.items():
        print(f"{var_name}: {var_data.shape}")

    print("\nForest Variables:")
    for var_name, var_data in batch.forest_variables.items():
        print(f"{var_name}: {var_data.shape}")


def print_batch_variables(batch, output_file=None) -> None:
    """
    Print all the variable values from the given DataBatch.

    Args:
    batch (DataBatch): The batch object containing climate and species data.
    output_file (str, optional): Path to the output file. If None, generates a timestamped filename.
    """
    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"batch_variables_{timestamp}.txt"

    with open(output_file, "w") as f:
        # First write metadata
        f.write("Metadata:\n")
        f.write(f"Latitudes: {batch.batch_metadata.latitudes}\n")
        f.write(f"Longitudes: {batch.batch_metadata.longitudes}\n")
        f.write(f"Timestamps: {batch.batch_metadata.timestamp}\n")
        f.write(f"Pressure Levels: {batch.batch_metadata.pressure_levels}\n\n")

        f.write("Surface Variables:\n")
        for var_name, var_data in batch.surface_variables.items():
            f.write(f"{var_name}:\n")
            f.write(f"Shape: {var_data.shape}\n")
            f.write(f"Data: {var_data}\n\n")

        # Repeat for other variable categories...
        variable_categories = [
            ("Single Variables", batch.single_variables),
            ("Atmospheric Variables", batch.atmospheric_variables),
            ("Species Extinction Variables", batch.species_extinction_variables),
            ("Land Variables", batch.land_variables),
            ("Agriculture Variables", batch.agriculture_variables),
            ("Forest Variables", batch.forest_variables),
        ]

        for category_name, category_dict in variable_categories:
            f.write(f"\n{category_name}:\n")
            for var_name, var_data in category_dict.items():
                f.write(f"{var_name}:\n")
                f.write(f"Shape: {var_data.shape}\n")
                f.write(f"Data: {var_data}\n\n")

    print(f"Batch variables have been written to: {output_file}")


def print_nan_counts(batch) -> None:
    def analyze_nan(name: str, tensor: torch.Tensor) -> None:
        total_values = tensor.numel()
        nan_count = torch.isnan(tensor).sum().item()
        nan_percentage = (nan_count / total_values) * 100
        if nan_count > 0:
            print(f"{name}")
            print(f"Total values: {total_values}")
            print(f"NaN count: {nan_count}")
            print(f"NaN percentage: {nan_percentage:.2f}%\n")
        else:
            print(f"{name} has no NaNs")

    print("\nMetadata Tensors:")
    analyze_nan("Latitudes", batch.batch_metadata.latitudes)
    analyze_nan("Longitudes", batch.batch_metadata.longitudes)

    print("\nSurface Variables:")
    for var_name, var_data in batch.surface_variables.items():
        analyze_nan(var_name, var_data)

    print("\nSingle Variables:")
    for var_name, var_data in batch.single_variables.items():
        analyze_nan(var_name, var_data)

    print("\nAtmospheric Variables:")
    for var_name, var_data in batch.atmospheric_variables.items():
        analyze_nan(var_name, var_data)

    print("\nSpecies Extinction Variables:")
    for var_name, var_data in batch.species_extinction_variables.items():
        analyze_nan(var_name, var_data)

    print("\nLand Variables:")
    for var_name, var_data in batch.land_variables.items():
        analyze_nan(var_name, var_data)

    print("\nAgriculture Variables:")
    for var_name, var_data in batch.agriculture_variables.items():
        analyze_nan(var_name, var_data)

    print("\nForest Variables:")
    for var_name, var_data in batch.forest_variables.items():
        analyze_nan(var_name, var_data)


batches = load_batches("data/")
# # print all the attributes a batch has
# print(batches[0].__dict__.keys())

print_batch_shapes(batches[0])
# print_batch_variables(batches[0])
# print_nan_counts(batches[0])
