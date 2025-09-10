import importlib
import pytest

import torch

from bfm_model.bfm import scaler

# mode = "normalize"
# mode = "standardize"

def dict_difference_norm(dict1, dict2):
    diffs = {}
    total_diff = 0.0
    
    for k in dict1.keys() & dict2.keys():
        diff = torch.norm(dict1[k] - dict2[k]).item()
        diffs[k] = diff
        total_diff += diff ** 2   # accumulate squared norm
    
    diffs["overall"] = total_diff ** 0.5  # L2 across all tensors
    return diffs


# stats_path = (
#     "/projects/prjs1134/data/projects/biodt/storage/monthly_batches/statistics/monthly_batches_stats_splitted_channels.json"
# )
# batch_path = "/projects/prjs1134/data/projects/biodt/storage/final_dataset_monthly/test/batch_2019-05-01_to_2019-06-01.pt"

# stats = scaler.load_stats(stats_path)


# data = torch.load(batch_path, map_location="cpu", weights_only=False)
# scaler.visit_obj(data)

# data_scaled = scaler._rescale_recursive(
#     data, stats=stats, dimensions_to_keep_by_key=scaler.dimensions_to_keep_monthly, direction="scaled", mode=mode
# )
# scaler.visit_obj(data_scaled)

# data_unscaled = scaler._rescale_recursive(
#     data_scaled, stats=stats, dimensions_to_keep_by_key=scaler.dimensions_to_keep_monthly, direction="original", mode=mode
# )
# scaler.visit_obj(data_unscaled)

# importlib.reload(scaler)

@pytest.mark.parametrize("mode", ["normalize", "standardize"])
def test_preprocess_mock_data(mode):
    tensor = torch.tensor([[1.0, 2.0, 3.0],
                        [4.0, 5.0, 6.0]])
    stats = {
        "foo": {
            "mean": tensor.mean().item(),
            "std": tensor.std().item(),
            "min": tensor.min().item(),
            "max": tensor.max().item(),
        }
    }

    data = {"foo": tensor}
    scaler.visit_obj(data)

    data_scaled = scaler._rescale_recursive(data, stats=stats, direction="scaled", mode=mode)
    scaler.visit_obj(data_scaled)

    data_unscaled = scaler._rescale_recursive(data_scaled, stats=stats, direction="original", mode=mode)
    scaler.visit_obj(data_unscaled)
    print(data_unscaled)
    diffs = dict_difference_norm(data, data_unscaled)
    l2_across = diffs["overall"]
    assert l2_across < 0.001
