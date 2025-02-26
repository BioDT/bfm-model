import glob
import os

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import torch


def load_last_run_id(run_id_file: str):
    if os.path.exists(run_id_file):
        with open(run_id_file, "r") as f:
            return f.read().strip()
    return None

def save_run_id(run_id_file: str, run_id: str):
    with open(run_id_file, "w") as f:
        f.write(run_id)

def print_auto_logged_info(r, mlflow_client):
    tags = {k: v for k, v in r.data.tags.items() if not k.startswith("mlflow.")}
    artifacts = [f.path for f in mlflow_client.list_artifacts(r.info.run_id, "model")]
    print(f"run_id: {r.info.run_id}")
    print(f"artifacts: {artifacts}")
    print(f"params: {r.data.params}")
    print(f"metrics: {r.data.metrics}")
    print(f"tags: {tags}")

def get_latest_checkpoint(ckpt_dir, pattern="*.ckpt"):
    checkpoint_files = glob.glob(os.path.join(ckpt_dir, pattern))
    if not checkpoint_files:
        return None
    # Choose the checkpoint with the latest creation time
    latest_ckpt = max(checkpoint_files, key=os.path.getctime)
    return latest_ckpt


def inspect_batch_shapes_dict(
    batch_dict,
    group_names = [
        "surface_variables",
        "single_variables",
        "atmospheric_variables",
        "species_extinction_variables",
        "land_variables",
        "agriculture_variables",
        "forest_variables",
        "species_variables",
    ]
):
    """
    Inspect and print the shapes of each variable in the given (potentially nested) dictionary-based batch.
    We first debug by printing top-level keys. If "batch_metadata" is not found, we check for a single-element list
    or a nested "batch" key.
    """

    print("\n=== Inspecting Batch Shapes (Dictionary Version) ===")

    # Debug: Print top-level info to see if it's nested, or a list
    if isinstance(batch_dict, list):
        print(f"** 'batch_dict' is a list of length {len(batch_dict)}. Top-level content:")
        for i, item in enumerate(batch_dict):
            print(f"  [List index {i}] => keys: {list(item.keys()) if isinstance(item, dict) else item}")
        # If it's a single-element list, we unwrap it
        if len(batch_dict) == 1 and isinstance(batch_dict[0], dict):
            print("Unwrapping single-element list...")
            batch_dict = batch_dict[0]
        else:
            print("Unable to unwrap properly or there's more than one element. Exiting.")
            return

    if not isinstance(batch_dict, dict):
        print(f"** 'batch_dict' is not a dict after unwrapping: {type(batch_dict)}. Exiting.")
        return

    print(f"Top-level keys now: {list(batch_dict.keys())}")

    # Possibly the real batch is under a key "batch" or "Batch"
    if "batch_metadata" not in batch_dict:
        # fallback check
        if "batch" in batch_dict and isinstance(batch_dict["batch"], dict):
            print("Found a nested 'batch' key. Checking inside it for 'batch_metadata'...")
            inner = batch_dict["batch"]
            if "batch_metadata" in inner:
                batch_dict = inner
            else:
                print("No 'batch_metadata' inside 'batch' key. Exiting.")
                return
        else:
            print("No 'batch_metadata' at top-level or in 'batch' key. Exiting.")
            return

    # Now we expect batch_dict["batch_metadata"] to exist
    md = batch_dict.get("batch_metadata", {})
    print("Metadata:")
    # Timestamps
    timestamps = md.get("timestamp", [])
    print(f"  Timestamps: {timestamps}")
    # Lead time
    lead_time = md.get("lead_time", None)
    print(f"  Lead time: {lead_time}")

    # lat/lon
    latitudes = md.get("latitudes", None)
    longitudes = md.get("longitudes", None)
    if hasattr(latitudes, "shape"):
        print(f"  Lat shape: {latitudes.shape}")
    else:
        print(f"  Lat shape: {latitudes}")
    if hasattr(longitudes, "shape"):
        print(f"  Lon shape: {longitudes.shape}")
    else:
        print(f"  Lon shape: {longitudes}")

    # pressure_levels
    pressure_levels = md.get("pressure_levels", None)
    print(f"  Pressure levels: {pressure_levels}")

    # species_list
    species_list = md.get("species_list", None)
    if species_list is not None:
        print(f"  Species list: {species_list}")

    # Inspect each group
    for group_name in group_names:
        group_data = batch_dict.get(group_name, None)
        if group_data is None:
            print(f"[{group_name}]: None or not present")
            continue

        if not isinstance(group_data, dict):
            print(f"[{group_name}] is not a dict: {type(group_data)} => {group_data}")
            continue

        print(f"\nInspecting group: [{group_name}] => keys: {list(group_data.keys())}")
        # group_data: var_name -> tensor
        for var_name, var_tensor in group_data.items():
            if hasattr(var_tensor, "shape"):
                print(f"  {var_name}: shape {var_tensor.shape}")
            else:
                print(f"  {var_name}: NOT a tensor or no .shape attribute")

    print("=== End of Batch Inspection ===\n")

def compute_next_timestamp(old_time_str, hours=6):
    """
    Example function to parse an ISO date, add specified number of hours, return new iso string.
    Adjust to your date/time logic.
    """
    from datetime import datetime, timedelta
    dt_format = "%Y-%m-%dT%H:%M:%S"
    old_dt = datetime.strptime(old_time_str, dt_format)
    new_dt = old_dt + timedelta(hours=hours)
    return new_dt.strftime(dt_format)


def inspect_batch_shapes_namedtuple(
    batch_obj,
    group_names = [
        "surface_variables",
        "single_variables",
        "atmospheric_variables",
        "species_extinction_variables",
        "land_variables",
        "agriculture_variables",
        "forest_variables",
        "species_variables",
    ]
):
    """
    Inspect shapes in a namedtuple-based Batch.
    """
    print("\n=== Inspecting Batch Shapes (Namedtuple Version) ===")

    # Metadata
    md = batch_obj.batch_metadata
    print("Metadata:")
    print(f"  Timestamps: {md.timestamp}")
    print(f"  Lead time: {md.lead_time}")
    # ... lat/lon, etc.

    # Each group
    for group_name in group_names:
        group_data = getattr(batch_obj, group_name, None)
        if group_data is None:
            print(f"[{group_name}]: None or not present")
            continue

        print(f"\nInspecting group: [{group_name}] => keys: {list(group_data.keys())}")
        for var_name, var_tensor in group_data.items():
            if hasattr(var_tensor, "shape"):
                print(f"  {var_name}: shape {var_tensor.shape}")
            else:
                print(f"  {var_name}: not a tensor")
    
    print("=== End of Namedtuple Batch Inspection ===\n")


def plot_europe_timesteps_and_difference(var_name: str,
                                         tensor: torch.Tensor,
                                         timestamps: list,
                                         pressure_levels: list = None,
                                         output_dir: str = "./plots",
                                         channels: list = None,
                                         plot: bool = True,
                                         save: bool = True):
    """
    Plots and saves maps for Timestep 1, Timestep 2, and their difference over Europe,
    using fixed coordinate arrays. It also includes metadata in the titles and filenames.
    
    Parameters:
      var_name (str): Name of the variable (used for titles and filenames).
      tensor (torch.Tensor): Input tensor with shape:
           - 4D: [batch, 2, lat, lon], or
           - 5D: [batch, 2, channel, lat, lon].
      timestamps (list): List of two timestamp strings, e.g. ['2020-05-29T18:00:00', '2020-05-30T00:00:00'].
      pressure_levels (list, optional): List of pressure levels corresponding to each channel (for atmospheric variables).
      output_dir (str): Directory where plots will be saved.
      channels (list, optional): List of channel indices to plot (if tensor is 5D). If not provided, all channels will be plotted.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert tensor to numpy.
    tensor_np = tensor.cpu().numpy()
    
    # Determine whether we have a channel dimension.
    if tensor_np.ndim == 4:
        # [batch, 2, lat, lon] -> treat as single channel.
        channels_to_plot = [0]
    elif tensor_np.ndim == 5:
        num_channels = tensor_np.shape[2]
        channels_to_plot = channels if channels is not None else list(range(num_channels))
    else:
        raise ValueError(f"Unexpected tensor shape: {tensor_np.shape}")
    
    # Fixed coordinate arrays:
    # Latitude: 152 points from 72.0 down to 34.25, then sorted ascending.
    lat_fixed = np.linspace(72, 34.25, 152)
    lat_fixed = np.sort(lat_fixed)  # Now from 34.25 to 72.0.
    # Longitude: 320 points linearly spaced from -30.0 to 40.0.
    lon_fixed = np.linspace(-30, 40, 320)
    # Create a meshgrid.
    Lon, Lat = np.meshgrid(lon_fixed, lat_fixed, indexing='xy')
    
    # Define Europe bounding box.
    europe_extent = [-30, 40, 34.25, 72]
    
    # Unpack timestamps.
    if len(timestamps) != 2:
        raise ValueError("timestamps must be a list of two elements (for Timestep 1 and Timestep 2).")
    tstamp0, tstamp1 = timestamps
    
    # Loop over channels.
    for ch in channels_to_plot:
        # Extract t1 and t2 from tensor.
        if tensor_np.ndim == 4:
            t1_data = tensor_np[0, 0, :, :]
            t2_data = tensor_np[0, 1, :, :]
        else:
            t1_data = tensor_np[0, 0, ch, :, :]
            t2_data = tensor_np[0, 1, ch, :, :]
        
        diff_data = t2_data - t1_data
        
        # Construct additional metadata for title/filename.
        pressure_info = ""
        if "atmospheric_variables" in var_name and pressure_levels is not None:
            try:
                pressure_info = f"_p{pressure_levels[ch]}"
            except IndexError:
                pressure_info = ""
        
        # Create figure with 3 subplots.
        fig, axes = plt.subplots(1, 3, figsize=(18, 6),
                                 subplot_kw={'projection': ccrs.PlateCarree()})
        
        # Plot Timestep 1.
        ax = axes[0]
        ax.set_extent(europe_extent, crs=ccrs.PlateCarree())
        try:
            ax.coastlines(resolution='50m')
        except Exception as e:
            print("Error drawing coastlines on Timestep 1:", e)
        cf1 = ax.contourf(Lon, Lat, t1_data, levels=60, cmap='viridis',
                          transform=ccrs.PlateCarree())
        ax.set_title(f"{var_name} (Ch {ch}{pressure_info})\nTimestep 1\n{tstamp0}")
        fig.colorbar(cf1, ax=ax, orientation='vertical', label="Value")
        
        # Plot Timestep 2.
        ax = axes[1]
        ax.set_extent(europe_extent, crs=ccrs.PlateCarree())
        try:
            ax.coastlines(resolution='50m')
        except Exception as e:
            print("Error drawing coastlines on Timestep 2:", e)
        cf2 = ax.contourf(Lon, Lat, t2_data, levels=60, cmap='viridis',
                          transform=ccrs.PlateCarree())
        ax.set_title(f"{var_name} (Ch {ch}{pressure_info})\nTimestep 2\n{tstamp1}")
        fig.colorbar(cf2, ax=ax, orientation='vertical', label="Value")
        
        # Plot Difference.
        ax = axes[2]
        ax.set_extent(europe_extent, crs=ccrs.PlateCarree())
        try:
            ax.coastlines(resolution='50m')
        except Exception as e:
            print("Error drawing coastlines on Difference plot:", e)
        cf_diff = ax.contourf(Lon, Lat, diff_data, levels=60, cmap='RdBu_r',
                              transform=ccrs.PlateCarree())
        ax.set_title(f"{var_name} (Ch {ch}{pressure_info})\nDifference (T2-T1)")
        fig.colorbar(cf_diff, ax=ax, orientation='vertical', label="Difference")
        
        plt.tight_layout()
        fig.canvas.draw()
        if save and plot:
            # Construct filename.
            filename = os.path.join(
                output_dir,
                f"{var_name}_ch{ch}{pressure_info}_t0_{tstamp0}_t1_{tstamp1}.jpeg"
            )
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.show()
            plt.close(fig)
            print(f"Saved plot: {filename}")
        else:
            plt.show()


def compute_species_occurrences(batch) -> dict:
    """
    Compute the number of occurrences for each species variable in the batch.
    For each variable in batch.species_variables, count the number of nonzero elements
    for Timestep 1 and Timestep 2. Also log the timestamps from batch.batch_metadata.
    
    Args:
        batch: An object (e.g., a namedtuple) with at least:
            - batch_metadata (with attribute 'timestamp': list of two strings)
            - species_variables (dict of variable name -> tensor)
              Each tensor is either:
                  4D: [batch, 2, lat, lon] or
                  5D: [batch, 2, channel, lat, lon]
    
    Returns:
        dict: A dictionary mapping each species variable name to its statistics:
              {
                <var_name>: {
                    "timestamp_t0": <timestamp for Timestep 1>,
                    "timestamp_t1": <timestamp for Timestep 2>,
                    "occurrences": <occurrence counts>
                },
                ...
              }
              
              For a 4D tensor, "occurrences" is a dict:
                  {"t0": count_at_timestep_1, "t1": count_at_timestep_2}
              For a 5D tensor, "occurrences" is a dict mapping channel indices to:
                  {"t0": count_at_timestep_1, "t1": count_at_timestep_2}
    """
    result = {}
    
    # Access metadata using attribute access.
    metadata = batch.batch_metadata
    # Assume metadata.timestamp is a list of two strings.
    try:
        timestamp_t0 = metadata.timestamp[0]
        timestamp_t1 = metadata.timestamp[1]
    except (AttributeError, IndexError):
        timestamp_t0 = "unknown"
        timestamp_t1 = "unknown"
    
    # Process each species variable.
    species_vars = batch.species_variables
    for var_name, tensor in species_vars.items():
        occ_stats = {}
        # Check the tensor dimensions.
        if tensor.ndim == 4:
            # 4D tensor: [batch, 2, lat, lon]
            # We assume batch size 1; count nonzero elements for each time slice.
            count_t0 = int(torch.count_nonzero(tensor[0, 0]).item())
            count_t1 = int(torch.count_nonzero(tensor[0, 1]).item())
            occ_stats = {"t0": count_t0, "t1": count_t1}
        elif tensor.ndim == 5:
            # 5D tensor: [batch, 2, channel, lat, lon]
            occ_stats = {}
            num_channels = tensor.shape[2]
            for ch in range(num_channels):
                count_t0 = int(torch.count_nonzero(tensor[0, 0, ch]).item())
                count_t1 = int(torch.count_nonzero(tensor[0, 1, ch]).item())
                occ_stats[f"channel_{ch}"] = {"t0": count_t0, "t1": count_t1}
        else:
            occ_stats = {"note": "Unsupported tensor shape for occurrence count."}
        
        result[var_name] = {
            "timestamp_t0": timestamp_t0,
            "timestamp_t1": timestamp_t1,
            "occurrences": occ_stats
        }
    
    return result
