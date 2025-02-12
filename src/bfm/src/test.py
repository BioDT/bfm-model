import os
from datetime import datetime, timedelta
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig

import torch
import lightning as L

from torch.utils.data import DataLoader

from lightning.pytorch import LightningModule, seed_everything
from lightning.pytorch.loggers import MLFlowLogger

from src.bfm.src.bfm import BFM
from src.bfm.src.dataloder import LargeClimateDataset, custom_collate

from src.bfm.src.train import BFM_pipe


# The variable weights come from: w_var = 1 / standard_dev
variable_weights = {
    "surface_variables": {
        "t2m": 0.13,
        "msl": 0.0011,
        # ... add more if surface has more
    },
    "single_variables": {"lsm": 2.27},
    "atmospheric_variables": {"z": 1.22e-5, "t": 0.036},
    "species_extinction_variables": {"ExtinctionValue": 38.0},
    "land_variables": {"Land": 5.84e-4, "NDVI": 19.6},
    "agriculture_variables": {
        "AgricultureLand": 0.053,
        "AgricultureIrrLand": 0.0,  # or skip if purely zero
        "ArableLand": 0.085,
        "Cropland": 0.36,
    },
    "forest_variables": {"Forest": 0.11},
}

def compute_loss(output, batch, logger):

    total_loss = 0.0
    count = 0

    # 1) Loop over each group we care about. For example:
    groups = [
        "surface_variables",
        "single_variables",
        "atmospheric_variables",
        "species_extinction_variables",
        "land_variables",
        "agriculture_variables",
        "forest_variables",
    ]

    for group_name in groups:
        # If group doesn't exist in output or batch, skip
        if group_name not in output or group_name not in batch._asdict():
            continue

        pred_dict = output[group_name]
        true_dict = getattr(batch, group_name)

        # 2) For each variable in this group, compute L1 loss with weighting
        group_loss = 0.0
        var_count = 0

        for var_name, pred_tensor in pred_dict.items():
            # If var_name not in the ground truth dict, skip
            if var_name not in true_dict:
                continue

            # TODO  but ensure your shapes/time dimension logic is consistent for all
            target = true_dict[var_name][:, -1]
            # print(f"{var_name} target: {target.shape}")
            # print(f"{var_name} prediction: {pred_tensor.shape}")
            # Default weight = 1.0 if not in dictionary
            w = variable_weights.get(group_name, {}).get(var_name, 1.0)

            # TODO 1: Test L2 loss 
            # TODO 2: Check Autocast and Half precision inference
            # L1 loss
            loss_var = torch.mean(torch.abs(pred_tensor - target))
            # loss_var = torch.mean(torch.sqrt(pred_tensor - target))
            # Log each variable's raw loss
            logger.log(f"{var_name} raw loss", loss_var)
            group_loss += w * loss_var
            var_count += 1

        if var_count > 0:
            group_loss /= var_count  # average within group
            total_loss += group_loss
            count += 1

    if count > 0:
        total_loss /= count  # average across groups (optiona

    print(f"Loss: {total_loss}")
    return total_loss



def evaluation(model, test_loader, device, logger, batch_size=1, half_precision=False):
    
    model.to(device)
    model.eval()
    lead_time = timedelta(hours=6)  # fixed lead time for pre-training

    eval_loss = []
    with torch.inference_mode():
        for batch, idx in test_loader:

            output = model(batch, lead_time, batch_size=batch_size)

            batch_loss = compute_loss(output, batch, logger)
            eval_loss.append(batch_loss)

    print(eval_loss)

    total_loss = torch.mean(eval_loss)
    return total_loss

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

    #Load the Test Dataset
    print("Setting up Dataloader ...")
    test_dataset = LargeClimateDataset(data_dir=cfg.data.test_data_path, num_species=cfg.data.species_number)  # Adapt
    print("Reading test data from :", cfg.data.test_data_path)
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=1,
        num_workers=cfg.training.workers,
        collate_fn=custom_collate,
        drop_last=True,
        shuffle=False,
    )

    # Setup logger
    current_time = datetime.now()
    remote_server_uri = f"http://0.0.0.0:{cfg.mlflow.port}"
    # tracking_uri="file:./mlruns" (default, goes to files. Serving Mlflow is separate)
    # mlf_logger = MLFlowLoggerWithSystemMetrics(experiment_name="BFM_logs", run_name=f"BFM_{current_time}")
    mlf_logger = MLFlowLogger(experiment_name="BFM_logs", run_name=f"BFM_{current_time}")

    # Setup the model
    model = BFM(
    surface_vars=(["t2m", "msl"]),
    single_vars=(["lsm"]),
    atmos_vars=(["z", "t"]),
    species_vars=(["ExtinctionValue"]),
    species_distr_vars=(["Distribution"]),
    land_vars=(["Land", "NDVI"]),
    agriculture_vars=(["AgricultureLand", "AgricultureIrrLand", "ArableLand", "Cropland"]),
    forest_vars=(["Forest"]),
    atmos_levels=cfg.data.atmos_levels,
    species_num=cfg.data.species_number,
    H=cfg.model.H,
    W=cfg.model.W,
    embed_dim=cfg.model.embed_dim,
    num_latent_tokens=cfg.model.num_latent_tokens,
    backbone_type=cfg.model.backbone,
    patch_size=cfg.model.patch_size,
    )


    checkpoint_path = cfg.evaluation.checkpoint_path
    #Load Model from Checkpoint
    print(f"Loading model from checkpoint: {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location=cfg.evaluation.test_device)
    print(state_dict)
    model.load_state_dict(state_dict)

    test_results = evaluation(model, test_dataloader, device=cfg.evaluation.test_device)

    # test_results is typically a list of dicts (one per test dataloader)
    print("=== Test Results ===")
    print(test_results)


if __name__ == "__main__":
    main()
