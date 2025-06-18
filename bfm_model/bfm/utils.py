"""
Copyright 2025 (C) TNO. Licensed under the MIT license.
"""

import glob
import os
from pathlib import Path

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import torch
from omegaconf import OmegaConf
from plotly.subplots import make_subplots


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
    group_names=[
        "surface_variables",
        "single_variables",
        "atmospheric_variables",
        "species_extinction_variables",
        "land_variables",
        "agriculture_variables",
        "forest_variables",
        "species_variables",
    ],
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
        print(f"batch_dict is not a dict after unwrapping: {type(batch_dict)}. Exiting.")
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
    """
    from datetime import datetime, timedelta

    #           "%Y-%m-%dT%H:%M:%S"
    dt_format = "%Y-%m-%d %H:%M:%S"
    # print("old_dt", old_time_str)
    if isinstance(old_time_str, (tuple, list)):
        old_time_str = old_time_str[0]
    old_dt = datetime.strptime(old_time_str, dt_format)
    new_dt = old_dt + timedelta(hours=hours)
    # print(f"old dt {old_dt} vs new_dt {new_dt}")
    return new_dt.strftime(dt_format)


def inspect_batch_shapes_namedtuple(
    batch_obj,
    group_names=[
        "surface_variables",
        "edaphic_variables",
        "atmospheric_variables",
        "climate_variables",
        "species_variables",
        "vegetation_variables",
        "land_variables",
        "agriculture_variables",
        "forest_variables",
        "misc_variables",
    ],
):
    """
    Inspect shapes in a namedtuple-based Batch.
    """
    print("\n=== Inspecting Batch Shapes (Namedtuple Version) ===")

    # Metadata
    if hasattr(batch_obj, "batch_metadata"):
        md = batch_obj.batch_metadata
        print("Metadata:")
        print(f"  Timestamps: {md.timestamp}")
        print(f"  Lead time: {md.lead_time}")
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


def plot_europe_timesteps_and_difference(
    var_name: str,
    tensor: torch.Tensor,
    timestamps: list,
    pressure_levels: list = None,
    output_dir: str = "./plots",
    channels: list = None,
    plot: bool = True,
    save: bool = True,
):
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
    Lon, Lat = np.meshgrid(lon_fixed, lat_fixed, indexing="xy")

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
        fig, axes = plt.subplots(1, 3, figsize=(18, 6), subplot_kw={"projection": ccrs.PlateCarree()})

        # Plot Timestep 1.
        ax = axes[0]
        ax.set_extent(europe_extent, crs=ccrs.PlateCarree())
        try:
            ax.coastlines(resolution="50m")
        except Exception as e:
            print("Error drawing coastlines on Timestep 1:", e)
        cf1 = ax.contourf(Lon, Lat, t1_data, levels=60, cmap="viridis", transform=ccrs.PlateCarree())
        ax.set_title(f"{var_name} (Ch {ch}{pressure_info})\nTimestep 1\n{tstamp0}")
        fig.colorbar(cf1, ax=ax, orientation="vertical", label="Value")

        # Plot Timestep 2.
        ax = axes[1]
        ax.set_extent(europe_extent, crs=ccrs.PlateCarree())
        try:
            ax.coastlines(resolution="50m")
        except Exception as e:
            print("Error drawing coastlines on Timestep 2:", e)
        cf2 = ax.contourf(Lon, Lat, t2_data, levels=60, cmap="viridis", transform=ccrs.PlateCarree())
        ax.set_title(f"{var_name} (Ch {ch}{pressure_info})\nTimestep 2\n{tstamp1}")
        fig.colorbar(cf2, ax=ax, orientation="vertical", label="Value")

        # Plot Difference.
        ax = axes[2]
        ax.set_extent(europe_extent, crs=ccrs.PlateCarree())
        try:
            ax.coastlines(resolution="50m")
        except Exception as e:
            print("Error drawing coastlines on Difference plot:", e)
        cf_diff = ax.contourf(Lon, Lat, diff_data, levels=60, cmap="RdBu_r", transform=ccrs.PlateCarree())
        ax.set_title(f"{var_name} (Ch {ch}{pressure_info})\nDifference (T2-T1)")
        fig.colorbar(cf_diff, ax=ax, orientation="vertical", label="Difference")

        plt.tight_layout()
        fig.canvas.draw()
        if save and plot:
            # Construct filename.
            filename = os.path.join(output_dir, f"{var_name}_ch{ch}{pressure_info}_t0_{tstamp0}_t1_{tstamp1}.jpeg")
            plt.savefig(filename, dpi=300, bbox_inches="tight")
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

        result[var_name] = {"timestamp_t0": timestamp_t0, "timestamp_t1": timestamp_t1, "occurrences": occ_stats}

    return result


class DictObj:
    def __init__(self, in_dict: dict):
        assert isinstance(in_dict, dict)
        for key, val in in_dict.items():
            if isinstance(val, (list, tuple)):
                setattr(self, key, [DictObj(x) if isinstance(x, dict) else x for x in val])
            else:
                setattr(self, key, DictObj(val) if isinstance(val, dict) else val)


def plot_species_stats_from_lists(actual_list: list, predicted_list: list, group_size: int = 5):
    """
    Plots a comparison of actual vs. predicted observations for each channel.

    Inputs:
      - actual_list: List of dictionaries with original observations.
      - predicted_list: List of dictionaries with model predictions.

    Both lists contain dictionaries of the form:
      {'Distribution': {
           'timestamp_t0': '2000-01-01T00:00:00',
           'timestamp_t1': '2000-01-01T06:00:00',
           'occurrences': {
               'channel_0': {'t0': <value>, 't1': <value>},
               'channel_1': {'t0': <value>, 't1': <value>},
               ...
           }
      }}

    Functionality:
      1. The top subplot shows raw time series (two observation points per record)
         for each channel from both actual and predicted lists.
      2. The middle subplot aggregates values by day (using the date portion of the timestamp)
         and calculates the daily percentage difference:
                % Diff = ((predicted_sum - actual_sum)/actual_sum) * 100
         (if actual_sum != 0, else 0).
      3. The bottom subplot shows a violin plot of residuals (predicted - actual)
         computed per observation (for matching timestamps from paired records).
      4. A dropdown menu (if more than group_size channels exist) lets the user select
         which channels to display.
    """
    channels_set = set()
    for record in actual_list:
        occ = record.get("Distribution", {}).get("occurrences", {})
        channels_set.update(occ.keys())
    channels = sorted(channels_set, key=lambda x: int(x.split("_")[1]) if x.split("_")[1].isdigit() else x)

    # For each channel, we extract two data points per record from both lists.
    ts_actual = {ch: [] for ch in channels}
    ts_pred = {ch: [] for ch in channels}

    for record in actual_list:
        dist = record.get("Distribution", {})
        t0 = dist.get("timestamp_t0")
        t1 = dist.get("timestamp_t1")
        occ = dist.get("occurrences", {})
        for ch in channels:
            if ch in occ:
                # Use the value under 't0' at timestamp_t0 and under 't1' at timestamp_t1.
                val0 = occ[ch].get("t0")
                val1 = occ[ch].get("t1")
                if t0:
                    ts_actual[ch].append((t0, val0))
                if t1:
                    ts_actual[ch].append((t1, val1))

    for record in predicted_list:
        dist = record.get("Distribution", {})
        t0 = dist.get("timestamp_t0")
        t1 = dist.get("timestamp_t1")
        occ = dist.get("occurrences", {})
        for ch in channels:
            if ch in occ:
                val0 = occ[ch].get("t0")
                val1 = occ[ch].get("t1")
                if t0:
                    ts_pred[ch].append((t0, val0))
                if t1:
                    ts_pred[ch].append((t1, val1))

    # Sort each series by timestamp.
    for ch in channels:
        ts_actual[ch].sort(key=lambda x: x[0])
        ts_pred[ch].sort(key=lambda x: x[0])

    def aggregate_by_day(series):
        agg = {}
        for ts, val in series:
            if ts is None:
                continue
            day = ts.split("T")[0]
            agg[day] = agg.get(day, 0) + (val if val is not None else 0)
        return agg

    daily_actual = {ch: aggregate_by_day(ts_actual[ch]) for ch in channels}
    daily_pred = {ch: aggregate_by_day(ts_pred[ch]) for ch in channels}

    daily_perc_diff = {ch: {} for ch in channels}
    for ch in channels:
        days = sorted(set(list(daily_actual[ch].keys()) + list(daily_pred[ch].keys())))
        for day in days:
            a_sum = daily_actual[ch].get(day, 0)
            p_sum = daily_pred[ch].get(day, 0)
            diff = (p_sum - a_sum) / a_sum * 100 if a_sum != 0 else 0
            daily_perc_diff[ch][day] = diff

    # We assume the actual_list and predicted_list are paired (same order).
    residuals = {ch: [] for ch in channels}
    for rec_act, rec_pred in zip(actual_list, predicted_list):
        dist_act = rec_act.get("Distribution", {})
        dist_pred = rec_pred.get("Distribution", {})
        for ts_key in ["timestamp_t0", "timestamp_t1"]:
            ts_act = dist_act.get(ts_key)
            occ_act = dist_act.get("occurrences", {})
            occ_pred = dist_pred.get("occurrences", {})
            # Choose the corresponding value key: 't0' for timestamp_t0, 't1' for timestamp_t1.
            key = "t0" if ts_key == "timestamp_t0" else "t1"
            for ch in channels:
                if ch in occ_act and ch in occ_pred:
                    a_val = occ_act[ch].get(key)
                    p_val = occ_pred[ch].get(key)
                    if a_val is not None and p_val is not None:
                        residuals[ch].append(p_val - a_val)

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=False,
        subplot_titles=(
            "Raw Time Series: Actual vs. Predicted",
            "Daily Percentage Difference",
            "Residual Distribution (Violin Plot)",
        ),
        vertical_spacing=0.12,
    )

    all_traces = []  # To manage dropdown visibility.

    # Top Plot: Add traces for raw time series.
    for ch in channels:
        x_act = [pt[0] for pt in ts_actual[ch]]
        y_act = [pt[1] for pt in ts_actual[ch]]
        x_pred = [pt[0] for pt in ts_pred[ch]]
        y_pred = [pt[1] for pt in ts_pred[ch]]
        trace_act = go.Scatter(x=x_act, y=y_act, mode="lines+markers", name=f"{ch} Actual")
        trace_pred = go.Scatter(x=x_pred, y=y_pred, mode="lines+markers", name=f"{ch} Predicted")
        fig.add_trace(trace_act, row=1, col=1)
        all_traces.append(trace_act)
        fig.add_trace(trace_pred, row=1, col=1)
        all_traces.append(trace_pred)

    # Middle Plot: Daily % difference.
    for ch in channels:
        days = sorted(daily_perc_diff[ch].keys())
        pdiff = [daily_perc_diff[ch][day] for day in days]
        trace_diff = go.Scatter(x=days, y=pdiff, mode="lines+markers", name=f"{ch} Daily % Diff")
        fig.add_trace(trace_diff, row=2, col=1)
        all_traces.append(trace_diff)

    # Bottom Plot: Residual distribution via violin plot.
    for ch in channels:
        trace_violin = go.Violin(y=residuals[ch], name=f"{ch} Residuals", box_visible=True, meanline_visible=True)
        fig.add_trace(trace_violin, row=3, col=1)
        all_traces.append(trace_violin)

    total_traces = len(all_traces)

    # Each channel contributes 4 traces (2 top, 1 middle, 1 bottom).
    traces_per_channel = 4
    channel_groups = [channels[i : i + group_size] for i in range(0, len(channels), group_size)]
    group_trace_indices = []
    for group in channel_groups:
        indices = []
        for ch in group:
            ch_index = channels.index(ch)
            start = ch_index * traces_per_channel
            indices.extend([start, start + 1, start + 2, start + 3])
        group_trace_indices.append(indices)

    # Set initial visibility: if more than one group, show only the first group's traces.
    initial_visible = [False] * total_traces
    if group_trace_indices:
        for idx in group_trace_indices[0]:
            initial_visible[idx] = True
    for i, trace in enumerate(all_traces):
        trace.visible = initial_visible[i]

    # Build dropdown buttons.
    buttons = []
    for i, indices in enumerate(group_trace_indices):
        vis = [False] * total_traces
        for idx in indices:
            vis[idx] = True
        label = f"Channels: {', '.join(channel_groups[i])}"
        button = dict(label=label, method="update", args=[{"visible": vis}, {"title": f"Actual vs. Predicted: {label}"}])
        buttons.append(button)

    if buttons:
        fig.update_layout(
            updatemenus=[
                dict(
                    active=0,
                    buttons=buttons,
                    direction="down",
                    pad={"r": 10, "t": 10},
                    showactive=True,
                    x=0.5,
                    xanchor="center",
                    y=1.05,
                    yanchor="top",
                )
            ]
        )

    fig.update_layout(
        title="Comparison of Actual vs. Predicted Observations, Daily % Difference, and Residual Distribution",
        height=900,
        width=1100,
    )
    fig.update_xaxes(title_text="Timestamp", row=1, col=1, type="date")
    fig.update_yaxes(title_text="Value", row=1, col=1)
    fig.update_xaxes(title_text="Day", row=2, col=1)
    fig.update_yaxes(title_text="Percentage Difference (%)", row=2, col=1)
    fig.update_xaxes(title_text=" ", row=3, col=1)  # Violin plot: x-axis is categorical
    fig.update_yaxes(title_text="Residual (Predicted - Actual)", row=3, col=1)

    fig.show()


def plot_species_stats_from_single_lists(stats_list: list, group_size: int = 5):
    channel_series = {}
    for stats in stats_list:
        dist = stats.get("Distribution", {})
        ts_t0 = dist.get("timestamp_t0")
        ts_t1 = dist.get("timestamp_t1")
        occ = dist.get("occurrences", {})
        for ch, values in occ.items():
            if ch not in channel_series:
                channel_series[ch] = []
            if ts_t0:
                channel_series[ch].append((ts_t0, values.get("t0", 0)))
            if ts_t1:
                channel_series[ch].append((ts_t1, values.get("t1", 0)))

    # Optionally sort each channel's series by timestamp (assuming ISO format).
    for ch, points in channel_series.items():
        points.sort(key=lambda x: x[0])

    fig = go.Figure()
    # We'll also record the channel order.
    channels = sorted(channel_series.keys(), key=lambda x: int(x.split("_")[1]) if x.split("_")[1].isdigit() else x)
    for ch in channels:
        pts = channel_series[ch]
        x_vals = [pt[0] for pt in pts]
        y_vals = [pt[1] for pt in pts]
        fig.add_trace(go.Scatter(x=x_vals, y=y_vals, mode="lines+markers", name=ch))

    total_traces = len(channels)
    # Create a list of buttons.
    buttons = []
    # Button for "All Channels"
    buttons.append(
        dict(
            label="All Channels",
            method="update",
            args=[{"visible": [True] * total_traces}, {"title": "Species Occurrences: All Channels"}],
        )
    )
    # Group channels into groups of group_size.
    groups = [channels[i : i + group_size] for i in range(0, total_traces, group_size)]
    for group in groups:
        # Build a list of booleans for trace visibility.
        visibility = [False] * total_traces
        # For each channel in this group, mark its index as visible.
        for ch in group:
            idx = channels.index(ch)
            visibility[idx] = True
        label = ", ".join(group)
        buttons.append(
            dict(label=label, method="update", args=[{"visible": visibility}, {"title": f"Species Occurrences: {label}"}])
        )

    fig.update_layout(
        updatemenus=[
            dict(
                active=0,
                buttons=buttons,
                direction="down",
                pad={"r": 10, "t": 10},
                showactive=True,
                x=0.5,
                xanchor="center",
                y=1.15,
                yanchor="top",
            )
        ]
    )

    fig.update_layout(
        title="Species Occurrences Over Time (Single Trace per Channel)",
        xaxis_title="Timestamp",
        yaxis_title="Occurrences",
        height=600,
        width=1000,
    )
    fig.update_xaxes(type="date")
    fig.show()


def load_config(output_dir: str | Path, config_file_name: str = "config.yaml"):
    config_path = Path(output_dir) / ".hydra"
    cfg = OmegaConf.load(str(config_path / config_file_name))
    return cfg


def inspect_batch(batch, indent: int = 0) -> None:
    """
    Recursively print out the contents of a Batch-like object:
    - namedtuple or obj with __dict__
    - dicts
    - torch.Tensors (shape & dtype)
    - lists (length & first element type)
    """
    prefix = "  " * indent
    # Namedtuple or object with attributes
    if hasattr(batch, "_fields"):  # likely a namedtuple
        print(f"{prefix}{batch.__class__.__name__}:")
        for field in batch._fields:
            value = getattr(batch, field)
            print(f"{prefix}  {field}:")
            inspect_batch(value, indent + 2)

    # plain dict
    elif isinstance(batch, dict):
        for k, v in batch.items():
            print(f"{prefix}  {k}:")
            inspect_batch(v, indent + 2)

    # torch tensor
    elif isinstance(batch, torch.Tensor):
        print(f"{prefix}  Tensor(shape={tuple(batch.shape)}, dtype={batch.dtype})")

    # list or tuple
    elif isinstance(batch, (list, tuple)):
        print(f"{prefix}  {type(batch).__name__}[len={len(batch)}]")
        if batch:
            print(f"{prefix}    example elem type: {type(batch[0]).__name__}")

    else:
        print(f"{prefix}  {type(batch).__name__}: {batch!r}")
