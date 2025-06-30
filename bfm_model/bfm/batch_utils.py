"""
Copyright 2025 (C) TNO. Licensed under the MIT license.
"""

from datetime import datetime
from typing import List

import torch
import typer
from dateutil.relativedelta import relativedelta

from bfm_model.bfm.dataloder import Batch, Metadata

app = typer.Typer(pretty_exceptions_enable=False)


def convert_namedtuple_to_dict(namedtuple):
    """
    Converts a named tuple to a dictionary.
    """
    value = namedtuple._asdict()
    value["batch_metadata"] = value["batch_metadata"]._asdict()
    # not necessary to recompute timestamps from lead time, they are already in the batch_metadata
    # just remove lead_time
    value["batch_metadata"].pop("lead_time")
    # replace latitudes and longitudes with list of floats
    value["batch_metadata"]["latitudes"] = value["batch_metadata"]["latitudes"].tolist()
    value["batch_metadata"]["longitudes"] = value["batch_metadata"]["longitudes"].tolist()
    # species need to be nested in dynamic
    species = value.pop("species_variables")
    value["species_variables"] = {"dynamic": species}
    # traverse_nested_dicts(value)
    # species dimensions are swapped, push again the species down: from [batch, time, species, lat, lon] to [batch, time, lat, lon, species]
    shape_before = species["Distribution"].shape
    species["Distribution"] = species["Distribution"].permute(0, 1, 3, 4, 2).contiguous().clone()
    shape_after = species["Distribution"].shape
    print(f"Species shape before: {shape_before}, after: {shape_after}")
    return value


def convert_dict_to_namedtuple(obj: dict):
    """
    Converts a dictionary to a named tuple.
    """
    obj["batch_metadata"] = Metadata(**obj["batch_metadata"])
    batch = Batch(**obj)
    return batch


def load_batch(batch_path: str):
    """
    Loads a batch from a file.
    """
    return torch.load(batch_path, weights_only=True)


def save_batch(batch, batch_path: str):
    """
    Saves a batch to a file.
    """
    if isinstance(batch, Batch):
        batch = convert_namedtuple_to_dict(batch)
    traverse_nested_dicts(batch)
    torch.save(batch, batch_path)


def format_prefix(prefix: List[str]) -> str:
    return ".".join([str(el) for el in prefix])


def visit_debug_tensors(obj, prefix: List[str] = []):
    nan_values = torch.isnan(obj.view(-1)).sum().item()
    inf_values = torch.isinf(obj.view(-1)).sum().item()
    tot_values = obj.numel()
    values_not_nan = obj[~torch.isnan(obj)]
    values_valid = values_not_nan[~torch.isinf(values_not_nan)]
    res_str = f"{format_prefix(prefix)} {obj.shape} NaN {nan_values/tot_values:.5%} Inf {inf_values/tot_values:.5%}"
    if values_not_nan.numel():
        min_not_nan = values_valid.min().item()
        max_not_nan = values_valid.max().item()
        mean = values_valid.mean().item()
        std = values_valid.std().item()
        res_str += f" min_max range: [{min_not_nan:.3f}, {max_not_nan:.3f}], mean: {mean:.3f}, std: {std:.3f}"
    print(res_str)


def visit_remove_batch_dimension(obj, prefix: List[str] = []):
    assert isinstance(obj, torch.Tensor)
    print(format_prefix(prefix), "reshape before", obj.shape)
    return torch.reshape(obj, obj.shape[1:])


def remove_batch_dimension(obj):
    return traverse_nested_dicts(obj, visit_tensor_function=visit_remove_batch_dimension)


def traverse_nested_dicts(obj, prefix: List[str] = [], visit_tensor_function=visit_debug_tensors):
    if isinstance(obj, torch.Tensor):
        return visit_tensor_function(obj, prefix)
    elif isinstance(obj, dict):
        for k in sorted(obj.keys()):
            v = obj[k]
            new_v = traverse_nested_dicts(v, prefix + [k], visit_tensor_function=visit_tensor_function)
            if new_v is not None:
                obj[k] = new_v
    elif isinstance(obj, list):
        if len(obj):
            item = obj[0]
            if isinstance(item, float) or isinstance(item, int) or isinstance(item, str):
                print(format_prefix(prefix), len(obj), "list", type(item))
            else:
                for i, v in enumerate(obj):
                    new_v = traverse_nested_dicts(v, prefix + [str(i)], visit_tensor_function=visit_tensor_function)
                    if new_v is not None:
                        obj[i] = new_v
        else:
            print(format_prefix(prefix), obj, "EMPTY LIST")
    else:
        print(format_prefix(prefix), "has type not supported:", type(obj))
    return obj


def build_new_batch_with_prediction(old_batch, prediction_dict, groups=None, time_dim=1, months: int = 1):
    """
    Build a new batch from `old_batch` by:
      - Keeping the last old timestep (since old_batch has T=2)
      - Appending the newly predicted timestep from `prediction_dict`

    This ensures the new batch again has T=2 time steps:
       [ (old_batch's last), (model's new prediction) ]

    Args:
        old_batch (namedtuple `Batch`):
            A batch with exactly 2 timesteps for each variable group, e.g. shape => [B,2,...].
        prediction_dict (dict):
            A dict keyed by group_name, then var_name -> predicted tensor of shape [B, ..., H, W].
            If it lacks a time dimension, we unsqueeze it so shape => [B,1,...,H,W].
        groups (list[str]): The variable group names to process. If None, we use a default set.
        time_dim (int): The dimension index for time, typically 1 if shape => [B,T, ...].

    Returns:
        new_batch (namedtuple `Batch`):
            A new batch of identical structure, also with T=2, but the second time is the newly predicted step.
    """
    if groups is None:
        groups = [
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
        ]

    new_batch = old_batch

    # For each group, unify last old time with predicted new time
    for group_name in groups:
        if not hasattr(new_batch, group_name):
            continue  # skip if group doesn't exist
        group_vars_old = getattr(new_batch, group_name)
        if group_vars_old is None:
            continue

        # predictions for this group
        group_vars_pred = prediction_dict.get(group_name, {})

        # For each variable in old_batch
        for var_name, old_tensor in group_vars_old.items():
            # old_tensor shape => [B, 2, (channels?), H, W], time_dim=1
            # keep last => [B, 1, ...]
            last_slice = old_tensor[:, -1:]  # shape => [B,1, ...]

            # find predicted data
            if var_name in group_vars_pred:
                pred_tensor = group_vars_pred[var_name]
                # print(f"var_name {var_name} pred tensor shape: {pred_tensor.shape}")
                # If missing a time dim, unsqueeze it:
                if pred_tensor.dim() == last_slice.dim() - 1:
                    pred_tensor = pred_tensor.unsqueeze(time_dim)
            else:
                # If no prediction for var_name, we could replicate last or skip
                pred_tensor = last_slice.clone()

            # Concat => shape [B,2,...]
            new_var_tensor = torch.cat([last_slice, pred_tensor], dim=time_dim)
            group_vars_old[var_name] = new_var_tensor

        new_batch = new_batch._replace(**{group_name: group_vars_old})

    # Update only the timestamp and lead_time in the metadata.
    new_metadata = update_batch_metadata(new_batch.batch_metadata, months=months)
    new_batch = new_batch._replace(batch_metadata=new_metadata)
    # print(f"new batch in creation timestamps: {new_batch.batch_metadata.timestamp}")

    return new_batch

    # print("Update batch metadata lead time: ", lt)
    # print(f"[update_meta] {t_last} -> {t_next} | lead={meta['lead_time']} months")


def update_batch_metadata(batch_metadata, months: int = 1):
    """
    Advance monthly timestamps & lead_time for batched metadata.

    Handles lead_time encoded as:
    - scalar int or float
    - 1-D torch tensor [B]
    - list / tuple [B]
    """
    meta = batch_metadata._asdict()

    if meta.get("timestamp"):
        t_last = _last_scalar_ts(meta["timestamp"])
        t_next = _add_months(t_last, months)
        meta["timestamp"] = [(t_last,), (t_next,)]
        # print(f"[update_meta] {t_last} -> {t_next} | lead={meta['lead_time']} months")

    lt = meta.get("lead_time", 0)
    # print("Update batch metadata lead time: ", lt)
    if isinstance(lt, torch.Tensor):
        meta["lead_time"] = lt + months
    elif isinstance(lt, (list, tuple)):
        meta["lead_time"] = [x + months for x in lt]
    else:
        meta["lead_time"] = lt + months

    return batch_metadata._replace(**meta)


def _last_scalar_ts(ts):
    """Return the last scalar timestamp string from (possibly nested) lists."""
    while isinstance(ts, (list, tuple)):
        if not ts:
            return ""
        ts = ts[-1]
    return str(ts)


def _add_months(ts_str: str, months: int) -> str:
    """Return ts_str + `months` months, preserving the DT_FORMAT."""
    DT_FORMAT = "%Y-%m-%d %H:%M:%S"
    dt = datetime.strptime(ts_str, DT_FORMAT)
    return (dt + relativedelta(months=months)).strftime(DT_FORMAT)


def _inspect_batch(file_path: str):
    data = load_batch(batch_path=file_path)
    traverse_nested_dicts(data)


@app.command()
def inspect_batch(file_path: str):
    return _inspect_batch(file_path)


if __name__ == "__main__":
    app()
