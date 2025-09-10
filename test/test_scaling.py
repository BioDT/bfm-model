import importlib

import torch

from bfm_model.bfm import scaler

stats_path = (
    "/projects/prjs1134/data/projects/biodt/storage/monthly_batches/statistics/monthly_batches_stats_splitted_channels.json"
)
batch_path = "/projects/prjs1134/data/projects/biodt/storage/final_dataset_monthly/test/batch_2019-05-01_to_2019-06-01.pt"

mode = "normalize"
mode = "standardize"


stats = scaler.load_stats(stats_path)


data = torch.load(batch_path, map_location="cpu", weights_only=False)
scaler.visit_obj(data)

data_scaled = scaler._rescale_recursive(
    data, stats=stats, dimensions_to_keep_by_key=scaler.dimensions_to_keep_monthly, direction="scaled", mode=mode
)
scaler.visit_obj(data_scaled)

data_unscaled = scaler._rescale_recursive(
    data_scaled, stats=stats, dimensions_to_keep_by_key=scaler.dimensions_to_keep_monthly, direction="original", mode=mode
)
scaler.visit_obj(data_unscaled)

# importlib.reload(scaler)

# tensor = torch.tensor([[1.0, 2.0, 3.0],
#                        [4.0, 5.0, 6.0]])

# stats = {
#     "foo": {
#         "mean": tensor.mean().item(),
#         "std": tensor.std().item(),
#         "min": tensor.min().item(),
#         "max": tensor.max().item(),
#     }
# }

# data = {"foo": tensor}
# scaler.visit_obj(data)

# data_scaled = scaler._rescale_recursive(data, stats=stats, direction="scaled", mode=mode)
# scaler.visit_obj(data_scaled)

# data_unscaled = scaler._rescale_recursive(data_scaled, stats=stats, direction="original", mode=mode)
# scaler.visit_obj(data_unscaled)
# print(data_unscaled)
