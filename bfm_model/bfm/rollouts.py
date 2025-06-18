"""
Copyright (C) 2025 TNO, The Netherlands. All rights reserved.
"""

import copy
import os
from datetime import datetime

import hydra
import lightning as L
import torch
from hydra.core.hydra_config import HydraConfig
from lightning.pytorch import seed_everything
from lightning.pytorch.loggers import MLFlowLogger
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Dataset

from bfm_model.bfm.batch_utils import build_new_batch_with_prediction, save_batch
from bfm_model.bfm.dataloader_monthly import LargeClimateDataset, custom_collate
from bfm_model.bfm.model import BFM
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
        test_dataset = LargeClimateDataset(
            data_dir=cfg.evaluation.rollout_data,
            scaling_settings=cfg.data.scaling,
            num_species=cfg.data.species_number,
            atmos_levels=cfg.data.atmos_levels,
            mode="pretrain",
        )
        print("Reading test data from :", cfg.evaluation.rollout_data)
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
        # The prediction is scaled, so no need to do futher scaling
        # predictions_unscaled = test_dataset.scale_batch(preds_list, direction="original")

        # print(preds_list)

        return preds_list

    # For each step in the rollout
    for step_idx in range(steps):
        # run predict
        print(f"Step {step_idx}")
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
        precision (int): Float precision (16 for half, 32 for single).
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
    test_dataset = LargeClimateDataset(
        data_dir=cfg.evaluation.rollout_data,
        scaling_settings=cfg.data.scaling,
        num_species=cfg.data.species_number,
        atmos_levels=cfg.data.atmos_levels,
        mode="pretrain",
    )
    print("Reading test data from :", cfg.evaluation.rollout_data)
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
    print(f"Loading model from checkpoint: {checkpoint_path}")
    device = torch.device(cfg.evaluation.test_device)
    print("weights device", device)

    swin_params = {}
    if cfg.model.backbone == "swin":
        selected_swin_config = cfg.model_swin_backbone[cfg.model.swin_backbone_size]
        swin_params = {
            "swin_encoder_depths": tuple(selected_swin_config.encoder_depths),
            "swin_encoder_num_heads": tuple(selected_swin_config.encoder_num_heads),
            "swin_decoder_depths": tuple(selected_swin_config.decoder_depths),
            "swin_decoder_num_heads": tuple(selected_swin_config.decoder_num_heads),
            "swin_window_size": tuple(selected_swin_config.window_size),
            "swin_mlp_ratio": selected_swin_config.mlp_ratio,
            "swin_qkv_bias": selected_swin_config.qkv_bias,
            "swin_drop_rate": selected_swin_config.drop_rate,
            "swin_attn_drop_rate": selected_swin_config.attn_drop_rate,
            "swin_drop_path_rate": selected_swin_config.drop_path_rate,
            "use_lora": selected_swin_config.use_lora,
        }

    loaded_model = BFM.load_from_checkpoint(
        checkpoint_path,
        map_location=device,
        surface_vars=(cfg.model.surface_vars),
        edaphic_vars=(cfg.model.edaphic_vars),
        atmos_vars=(cfg.model.atmos_vars),
        climate_vars=(cfg.model.climate_vars),
        species_vars=(cfg.model.species_vars),
        vegetation_vars=(cfg.model.vegetation_vars),
        land_vars=(cfg.model.land_vars),
        agriculture_vars=(cfg.model.agriculture_vars),
        forest_vars=(cfg.model.forest_vars),
        redlist_vars=(cfg.model.redlist_vars),
        misc_vars=(cfg.model.misc_vars),
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
        **swin_params,
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
    rollout_dict = rollout_forecast(trainer, model=loaded_model, initial_batch=test_sample, cfg=cfg, steps=10)
    # print(f"len rollout dict {len(rollout_dict)}| items in", rollout_dict["batches"][0].batch_metadata)

    os.makedirs("rollout_batches", exist_ok=True)
    for i, batch_dict in enumerate(rollout_dict["batches"]):
        timestamps = batch_dict[0].batch_metadata.timestamp
        save_batch(batch_dict, f"rollout_batches/prediction_{timestamps[0]}_to_{timestamps[1]}.pt")
        print(f"\n--- Inspecting Batch {i} ---")
        inspect_batch_shapes_namedtuple(batch_dict[0])


if __name__ == "__main__":
    main()
