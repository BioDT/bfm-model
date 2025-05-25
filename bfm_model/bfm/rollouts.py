"""
Copyright (C) 2025 TNO, The Netherlands. All rights reserved.
"""
import copy
import os
from datetime import datetime, timedelta

import hydra
import lightning as L
import torch
from hydra.core.hydra_config import HydraConfig
from lightning.pytorch import seed_everything
from lightning.pytorch.loggers import MLFlowLogger
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Dataset

from bfm_model.bfm.batch_utils import save_batch
from bfm_model.bfm.dataloader_monthly import LargeClimateDataset, custom_collate
from bfm_model.bfm.train_lighting import BFM_lighting
from bfm_model.bfm.utils import compute_next_timestamp, inspect_batch_shapes_namedtuple


def rollout_forecast(trainer, model, initial_batch, cfg, steps=2, batch_size=1):

    # Container for results
    rollout_dict = {
        "predictions": [],
        "batches": [],
        "timestamps": [],
        "lead_times": [],
    }

    # current_batch = copy.deepcopy(initial_batch)
    current_batch = initial_batch
    B = batch_size

    class SingleBatchDataset(Dataset):
        def __init__(self, one_batch):
            super().__init__()
            self.one_batch = one_batch

        def __len__(self):
            # we have exactly 1 item in this dataset
            return 1

        def __getitem__(self, idx):
            # Return the namedtuple or dictionary directly
            return self.one_batch

    def single_item_collate(batch_list):
        """
        A custom collate for a single-item dataset.
        batch_list has length=1, so we just return batch_list[0]
        exactly as is (could be a namedtuple or dict).
        """
        return batch_list[0]

    def run_predict_on_batch(batch_dict):
        """
        1) Build single-item dataset
        2) Build DataLoader
        3) run trainer.predict
        4) Combine predictions
        """
        data_dir = "/home/atrantas/bfm-model/data_small/rollout"
        test_dataset = LargeClimateDataset(
        data_dir=data_dir, scaling_settings=cfg.data.scaling, 
        num_species=cfg.data.species_number, atmos_levels=cfg.data.atmos_levels, mode="pretrain")
        print("Reading test data from :", data_dir)
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=1,
            num_workers=cfg.training.workers,
            collate_fn=custom_collate,
            drop_last=True,
            shuffle=False,
            pin_memory=False,
        )

        ds = SingleBatchDataset(batch_dict)

        dl = DataLoader(ds, batch_size=B, collate_fn=single_item_collate, shuffle=False)

        # returns a list of predictions (one per batch item, but we only have 1)
        preds_list = trainer.predict(model, dataloaders=test_dataloader)
        # predictions_unscaled = test_dataset.scale_batch(preds_list, direction="original")

        # print(preds_list)

        return preds_list

    # For each step in the rollout
    for step_idx in range(steps):
        # run predict
        print(f"Step {step_idx}")
        print(current_batch)
        preds = run_predict_on_batch(current_batch)

        # store
        rollout_dict["predictions"].append(preds)
        rollout_dict["batches"].append(copy.deepcopy(current_batch))

        # handle times
        # Suppose your "Batch" has metadata => lead_time, timestamps...
        # Store the new predicted time.
        step_timestamp = current_batch[0].batch_metadata.timestamp[-1]
        rollout_dict["timestamps"].append(step_timestamp)
        rollout_dict["lead_times"].append(current_batch[0].batch_metadata.lead_time)

        # Build a new batch that has (last old time) + (predicted new time).
        new_batch = build_new_batch_with_prediction(current_batch[0], preds[0])

        # This new_batch becomes the "current_batch" for the next iteration
        current_batch[0] = new_batch

    return rollout_dict

def _last_scalar_ts(ts):
    """
    Walk down nested lists/tuples until we hit the last scalar string.
    """
    while isinstance(ts, (list, tuple)):
        ts = ts[-1]
    return ts          # guaranteed str

def update_batch_metadata(batch_metadata, hours: int = 6):
    """
    Keep all original fields but normalise `timestamp` to `[t_last , t_next]`
    where `t_last` is the scalar last timestamp (even if nested) and
    `t_next  = t_last + hours`.
    `lead_time` is incremented by `hours` (kept as float hours).
    """
    meta_dict = batch_metadata._asdict()

    # ---------- timestamps ----------
    ts_field = meta_dict.get("timestamp")
    if ts_field:
        t_last = _last_scalar_ts(ts_field)                  # flatten
        t_next = compute_next_timestamp(t_last, hours)
        meta_dict["timestamp"] = [t_last, t_next]

    # ---------- lead_time -----------
    lt = meta_dict.get("lead_time", 0.0)
    if hasattr(lt, "cpu"):                # Tensor
        lt_val = float(lt.cpu().numpy()[0])
    elif isinstance(lt, timedelta):       # timedelta
        lt_val = lt.total_seconds() / 3600.0
    else:
        lt_val = float(lt)
    meta_dict["lead_time"] = lt_val + hours   # keep plain float

    # DEBUG
    # print(f"[update_meta] in  -> {batch_metadata.timestamp}")
    # print(f"[update_meta] out -> {meta_dict['timestamp']}  lead={meta_dict['lead_time']}h")

    return batch_metadata._replace(**meta_dict)




def build_new_batch_with_prediction(old_batch, prediction_dict, groups=None, time_dim=1, hours: int = 6):
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
            "misc_variables"
        ]

    # Make a copy so we don't modify old_batch in place
    # new_batch = copy.deepcopy(old_batch)
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
                    # e.g. old had 5 dims, new has 4 => unsqueeze time
                    pred_tensor = pred_tensor.unsqueeze(time_dim)
            else:
                # If no prediction for var_name, we could replicate last or skip
                pred_tensor = last_slice.clone()

            # Concat => shape [B,2,...]
            new_var_tensor = torch.cat([last_slice, pred_tensor], dim=time_dim)
            group_vars_old[var_name] = new_var_tensor

        new_batch = new_batch._replace(**{group_name: group_vars_old})

    # Update only the timestamp and lead_time in the metadata.
    new_metadata = update_batch_metadata(new_batch.batch_metadata, hours=hours)
    new_batch = new_batch._replace(batch_metadata=new_metadata)
    # print(f"new batch in creation timestamps: {new_batch.batch_metadata.timestamp}")
    
    return new_batch


@hydra.main(version_base=None, config_path="configs", config_name="train_config")
def main(cfg: DictConfig):
    """
    Test/inference script using a PyTorch Lightning module.

    Args:
        checkpoint_path (str): Path to the trained checkpoint (.ckpt).
        data_dir (str): Directory containing test data.
        batch_size (int): Batch size for test loader.
        num_workers (int): Number of workers for DataLoader.
        gpus (int): Number of GPUs to use (if 0, run on CPU).
        precision (int): Float precision (16 for half, 32 for single, etc.).
        accelerator (str): "gpu", "cpu", "tpu", etc.

    Returns:
        test_results (dict or list): Test metrics returned by trainer.test().
    """

    # Setup config
    print(OmegaConf.to_yaml(cfg))
    # Set the internal precision for TensorCores (H100s)
    torch.set_float32_matmul_precision(cfg.training.precision_in)

    # Seed the experiment for numpy, torch and python.random.
    seed_everything(42, workers=True)

    # Load the Test Dataset
    print("Setting up Dataloader ...")
    # data_dir = "data_small/2009_batches" # Needs a single batch item to start the rollouts from
    data_dir = "/home/atrantas/bfm-model/data_small/rollout" # Need a single item
    test_dataset = LargeClimateDataset(
        data_dir=data_dir, scaling_settings=cfg.data.scaling, 
        num_species=cfg.data.species_number, atmos_levels=cfg.data.atmos_levels, mode="pretrain")
    print("Reading test data from :", data_dir)
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=1,
        num_workers=cfg.training.workers,
        collate_fn=custom_collate,
        drop_last=True,
        shuffle=False,
        pin_memory=False,
    )
    print(f"Length Dataloader {len(test_dataloader)}")

    output_dir = HydraConfig.get().runtime.output_dir

    # Setup logger
    current_time = datetime.now()
    # log the metrics in the hydra folder (easier to find)
    mlf_logger_in_hydra_folder = MLFlowLogger(
        experiment_name="BFM_logs", run_name=f"BFM_{current_time}", save_dir=f"{output_dir}/logs"
    )
    # also log in the .mlruns folder so that you can run mlflow server and see every run together
    # tracking_uri = f"http://0.0.0.0:{cfg.mlflow.port}"
    mlf_logger_in_current_folder = MLFlowLogger(experiment_name="BFM_logs", run_name=f"BFM_{current_time}")

    checkpoint_path = cfg.evaluation.checkpoint_path
    # Load Model from Checkpoint
    print(f"Loading model from checkpoint: {checkpoint_path}")

    device = torch.device(cfg.evaluation.test_device)
    print("weigths device", device)

    loaded_model = BFM_lighting.load_from_checkpoint(
        checkpoint_path,
        map_location=device,
        surface_vars=(["t2m", "msl", "slt", "z", "u10", "v10", "lsm"]),
        edaphic_vars=(["swvl1", "swvl2", "stl1", "stl2"]),
        atmos_vars=(["z", "t", "u", "v", "q"]),
        climate_vars=(["smlt", "tp", "csfr", "avg_sdswrf", "avg_snswrf",
                       "avg_snlwrf", "avg_tprate", "avg_sdswrfcs", "sd", "t2m", "d2m"]),
        species_vars=(["1340361", "1340503", "1536449", "1898286", "1920506", "2430567",
                       "2431885", "2433433", "2434779", "2435240", "2435261", "2437394",
                       "2441454", "2473958", "2491534", "2891770", "3034825", "4408498",
                        "5218786", "5219073", "5219173", "5219219", "5844449", "8002952",
                        "8077224", "8894817", "8909809", "9809229"]),
        vegetation_vars=(["NDVI"]),
        land_vars=(["Land"]),
        agriculture_vars=(["Agriculture", "Arable", "Cropland"]),
        forest_vars=(["Forest"]),
        misc_vars=(["avg_slhtf", "avg_pevr"]),
        atmos_levels=cfg.data.atmos_levels,
        species_num=cfg.data.species_number,
        H=cfg.model.H,
        W=cfg.model.W,
        num_latent_tokens=cfg.model.num_latent_tokens,
        backbone_type=cfg.model.backbone,
        patch_size=cfg.model.patch_size,
        embed_dim=cfg.model.embed_dim,
        num_heads=cfg.model.num_heads,
        head_dim=cfg.model.head_dim,
        depth=cfg.model.depth,
    )

    trainer = L.Trainer(
        accelerator=cfg.training.accelerator,
        devices=cfg.training.devices,
        precision=cfg.training.precision,
        log_every_n_steps=cfg.training.log_steps,
        logger=[mlf_logger_in_hydra_folder, mlf_logger_in_current_folder],
        enable_checkpointing=False,
        enable_progress_bar=True,
    )

    # test_results is typically a list of dicts (one per test dataloader)
    print("=== Test Results ===")

    test_sample = next(iter(test_dataloader))
    rollout_dict = rollout_forecast(trainer, model=loaded_model, initial_batch=test_sample, cfg=cfg, steps=2)
    # print(f"len rollout dict {len(rollout_dict)}| items in", rollout_dict["batches"][0].batch_metadata)

    os.makedirs("rollout_batches", exist_ok=True)
    for i, batch_dict in enumerate(rollout_dict["batches"]):
        timestamps = batch_dict[0].batch_metadata.timestamp
        # save_batch(batch_dict, f"rollout_batches/prediction_{timestamps[0]}_to_{timestamps[1]}.pt")
        print(f"\n--- Inspecting Batch {i} ---")
        inspect_batch_shapes_namedtuple(batch_dict[0])


if __name__ == "__main__":
    main()
