"""
Copyright (C) 2025 TNO, The Netherlands. All rights reserved.
"""

import json
from typing import List, Literal

import torch

excluded_keys = ["batch_metadata", "metadata"]
dimensions_to_keep_by_key = {
    "species_variables": {
        "dynamic": {
            "Distribution": [3],  # [time, lat, lon, species]
        },
    },
    "atmospheric_variables": {
        "z": [1],  # [time, z, lat, lon]
        "t": [1],  # [time, t, lat, lon]
    },
}
dimensions_to_keep_monthly = {
    "atmospheric_variables": {
        "z": [1],  # [time, z, lat, lon]
        "t": [1],  # [time, t, lat, lon]
        "u": [1],  # [time, t, lat, lon]
        "v": [1],  # [time, t, lat, lon]
        "q": [1],  # [time, t, lat, lon]
    }
}
# TODO: this should be saved in the standardization config!!!


def _rescale_recursive(
    obj: dict,
    stats: dict,
    dimensions_to_keep_by_key: dict | list = {},
    prefix: List[str] = [],
    mode: Literal["standardize", "normalize"] = "normalize",
    direction: Literal["original", "scaled"] = "scaled",
):
    if isinstance(obj, torch.Tensor):
        # print("prefix", prefix)
        if stats:
            # print(f"stats: {format_prefix(prefix)}", stats, dimensions_to_keep_by_key)
            mean_val = stats["mean"]
            std_val = stats["std"]
            min_val = stats["min"]
            max_val = stats["max"]
            if dimensions_to_keep_by_key:
                assert isinstance(
                    dimensions_to_keep_by_key, list
                ), f"dimensions_to_keep_by_key should be a list, got {type(dimensions_to_keep_by_key)}, {dimensions_to_keep_by_key}"
                # print("dimensions_to_keep_by_key:", dimensions_to_keep_by_key)
                # print("obj.shape:", obj.shape)
                assert (
                    len(dimensions_to_keep_by_key) == 1
                ), f"dimensions_to_keep_by_key should have length 1, got {len(dimensions_to_keep_by_key)}"
                dimension_to_keep = dimensions_to_keep_by_key[0]
                shape_on_splitted_dimension = obj.shape[dimension_to_keep]
                splitted = torch.chunk(obj, chunks=shape_on_splitted_dimension, dim=dimension_to_keep)
                # print(format_prefix(prefix), "real mean:", [splitted[i].mean() for i in range(shape_on_splitted_dimension)], "real std:", [splitted[i].std() for i in range(shape_on_splitted_dimension)])
                if not isinstance(mean_val, (list, tuple)):
                    mean_val = [mean_val] * shape_on_splitted_dimension
                    std_val = [std_val] * shape_on_splitted_dimension
                    min_val = [min_val] * shape_on_splitted_dimension
                    max_val = [max_val] * shape_on_splitted_dimension

                if mode == "standardize":
                    if direction == "scaled":
                        splitted_changed = [(splitted[i] - mean_val[i]) / std_val[i] for i in range(shape_on_splitted_dimension)]
                    else:
                        splitted_changed = [splitted[i] * std_val[i] + mean_val[i] for i in range(shape_on_splitted_dimension)]
                elif mode == "normalize":
                    # min-max normalization
                    if direction == "scaled":
                        # print([min_val[i] for i in range(shape_on_splitted_dimension)])
                        splitted_changed = [
                            (splitted[i] - min_val[i]) / (max_val[i] - min_val[i]) for i in range(shape_on_splitted_dimension)
                        ]
                    else:
                        splitted_changed = [
                            splitted[i] * (max_val[i] - min_val[i]) + min_val[i] for i in range(shape_on_splitted_dimension)
                        ]
                res = torch.cat(splitted_changed, dim=dimension_to_keep)
            else:
                # print("mean:", mean, "std:", std)
                if mode == "standardize":
                    if direction == "scaled":
                        res = torch.add(obj, -mean_val) / std_val  # (obj - mean) / std
                    else:
                        res = torch.add(obj / std_val, mean_val)  # (obj / std) + mean
                elif mode == "normalize":
                    # min-max normalization
                    if direction == "scaled":
                        res = torch.add(obj, -min_val) / (max_val - min_val)
                    else:
                        res = torch.add(obj / (max_val - min_val), min_val)
            return res
        else:
            print(f"RESCALE cfg not found: current_key: {format_prefix(prefix)}")
            pass
    elif isinstance(obj, dict):
        for k, v in obj.items():
            if k not in ["batch_metadata", "metadata"]:
                obj[k] = _rescale_recursive(
                    v,
                    stats.get(str(k), {}),
                    dimensions_to_keep_by_key.get(str(k), {}),
                    prefix=prefix + [k],
                    mode=mode,
                    direction=direction,
                )
        return obj
    else:
        # print("type not supported:", type(obj), f"current_key: {current_key}")
        pass


def format_prefix(prefix: List[str]) -> str:
    return ".".join([str(el) for el in prefix])


def visit_obj(obj, prefix: List[str] = []):
    if isinstance(obj, torch.Tensor):
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
            res_str += f" min_max range: [{min_not_nan}, {max_not_nan}], mean: {mean:.5f}, std: {std:.5f}"
        print(res_str)
    elif isinstance(obj, dict):
        for k, v in obj.items():
            visit_obj(v, prefix + [k])
    elif isinstance(obj, list):
        if len(obj):
            item = obj[0]
            if isinstance(item, float) or isinstance(item, int) or isinstance(item, str):
                print(format_prefix(prefix), len(obj), "list", type(item))
            else:
                for i, v in enumerate(obj):
                    visit_obj(v, prefix + [str(i)])
        else:
            print(format_prefix(prefix), obj, "EMPTY LIST")
    else:
        print(format_prefix(prefix), "has type not supported:", type(obj))


def load_stats(path: str):
    with open(path, "r") as f:
        cfg = json.load(f)
    return cfg


# # TEST
# data_path = "/data/projects/biodt/storage/batches/"
# standardization_path = "/data/projects/biodt/storage/batches/statistics.json"
# standardization_cfg = load_stats(standardization_path)
# fpath = f"{data_path}batch_2005-10-14_00-00-00_to_2005-10-14_06-00-00.pt"
# batch_data = torch.load(fpath, map_location='cpu', weights_only=True)
# # dataset = LargeClimateDataset(data_dir=data_path, standardization_cfg=standardization_cfg, num_species=2)
# # batch = dataset[0]

# # print(batch_data["atmospheric_variables"]["z"].mean())
# # visit_obj(batch_data)
# # visit_obj(batch_data["atmospheric_variables"]["z"])
# print("atmospheric_variables.z")
# visit_obj(batch_data["atmospheric_variables"]["t"])
# # visit_obj(batch_data["species_variables"]["dynamic"]["Distribution"])
# _rescale_recursive(batch_data, standardization_cfg, dimensions_to_keep_by_key, mode="standardize", direction="scaled")
# # _rescale_recursive(batch_data, standardization_cfg, dimensions_to_keep_by_key, mode="normalize", direction="scaled")
# # print(batch_data["atmospheric_variables"]["z"].mean())
# # visit_obj(batch_data)
# # visit_obj(batch_data["atmospheric_variables"]["z"])
# print("atmospheric_variables.z STANDARDIZED")
# visit_obj(batch_data["atmospheric_variables"]["t"])
# # visit_obj(batch_data["species_variables"]["dynamic"]["Distribution"])
# _rescale_recursive(batch_data, standardization_cfg, dimensions_to_keep_by_key, mode="standardize", direction="original")
# # _rescale_recursive(batch_data, standardization_cfg, dimensions_to_keep_by_key, mode="normalize", direction="original")
# # visit_obj(batch_data)
# # visit_obj(batch_data["atmospheric_variables"]["z"])
# print("atmospheric_variables.z")
# visit_obj(batch_data["atmospheric_variables"]["t"])
# visit_obj(batch_data["species_variables"]["dynamic"]["Distribution"])
# TODO: should be back to initial point (A->B->A), it is not


# debug
# tensor = batch_data["atmospheric_variables"]["z"]
# standardization_cfg = {
#     "std": [0.1, 0.2, 0.3],
#     "mean": [0.1, 0.2, 0.3],
# }
# print(tensor.shape)
# dimensions_to_keep_by_key = [1]
# dimension_to_keep = dimensions_to_keep_by_key[0]
# shape_on_splitted_dimension = tensor.shape[dimension_to_keep]

# # does not work when we don't know where the dimension is (notation)
# # for i in range(shape_on_splitted_dimension):
# #     tensor[:, i, :, :] = tensor[:, i, :, :] * standardization_cfg["std"][i] + standardization_cfg["mean"][i]

# splitted = torch.chunk(tensor, chunks=shape_on_splitted_dimension, dim=dimension_to_keep)
# splitted_changed = [splitted[i] * standardization_cfg["std"][i] + standardization_cfg["mean"][i] for i in range(shape_on_splitted_dimension)]
# res = torch.cat(splitted_changed, dim=dimension_to_keep)

# monthly batches
# batches_path = "/projects/prjs1134/data/projects/biodt/storage/monthly_batches"
# fpath = f"{batches_path}/batches/batch_2000-01-01_to_2000-02-01.pt"
# standardization_path = f"{batches_path}/statistics/monthly_batches_stats.json"
# standardization_stats = load_stats(standardization_path)
# batch_data = torch.load(fpath, map_location="cpu", weights_only=True)
# visit_obj(batch_data["species_variables"])
# _ = _rescale_recursive(batch_data, standardization_stats, dimensions_to_keep_by_key={}, mode="standardize", direction="original")
# visit_obj(batch_data["species_variables"])
