import os
import glob


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