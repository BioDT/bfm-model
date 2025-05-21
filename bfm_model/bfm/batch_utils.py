"""
Copyright (C) 2025 TNO, The Netherlands. All rights reserved.
"""

from typing import List

import torch
import typer

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
    # test one shape of tensors to see if we need to remove the batch dimension
    t = batch["agriculture_variables"]["AgricultureIrrLand"]
    print("Shape of tensor to test:", t.shape)
    if len(t.shape) == 4:
        batch = remove_batch_dimension(batch)
    # shape_species = batch["species_variables"]["dynamic"]["Distribution"].shape
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


def _inspect_batch(file_path: str):
    data = load_batch(batch_path=file_path)
    traverse_nested_dicts(data)


@app.command()
def inspect_batch(file_path: str):
    return _inspect_batch(file_path)


if __name__ == "__main__":
    app()
