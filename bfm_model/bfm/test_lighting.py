from datetime import datetime

import hydra
import lightning as L
import torch
from hydra.core.hydra_config import HydraConfig
from lightning.pytorch import seed_everything
from lightning.pytorch.loggers import MLFlowLogger
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from bfm_model.bfm.dataloder import LargeClimateDataset, custom_collate
from bfm_model.bfm.train_lighting import BFM_lighting


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
    test_dataset = LargeClimateDataset(
        data_dir=cfg.data.test_data_path, scaling_settings=cfg.data.scaling, num_species=cfg.data.species_number
    )  # Adapt
    print("Reading test data from :", cfg.data.test_data_path)
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=cfg.evaluation.batch_size,
        num_workers=cfg.training.workers,
        collate_fn=custom_collate,
        drop_last=True,
        shuffle=False,
    )

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

    trainer = L.Trainer(
        accelerator=cfg.training.accelerator,
        devices=cfg.training.devices,
        precision=cfg.training.precision,
        log_every_n_steps=cfg.training.log_steps,
        limit_test_batches=2,
        limit_predict_batches=2,
        logger=[mlf_logger_in_hydra_folder, mlf_logger_in_current_folder],
        enable_checkpointing=False,
        enable_progress_bar=True,
    )

    bfm_model = BFM_lighting(
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
        num_latent_tokens=cfg.model.num_latent_tokens,
        backbone_type=cfg.model.backbone,
        patch_size=cfg.model.patch_size,
        embed_dim=cfg.model.embed_dim,
        num_heads=cfg.model.num_heads,
        head_dim=cfg.model.head_dim,
        depth=cfg.model.depth,
        batch_size=cfg.evaluation.batch_size,
    )

    checkpoint_path = cfg.evaluation.checkpoint_path
    # Load Model from Checkpoint
    print(f"Loading model from checkpoint: {checkpoint_path}")
    # V1 Do the inference
    test_results = trainer.test(model=bfm_model, ckpt_path=checkpoint_path, dataloaders=test_dataloader)
    predictions = trainer.predict(model=bfm_model, ckpt_path=checkpoint_path, dataloaders=test_dataloader)
    print("=== Test Results ===")
    print(test_results)
    print(predictions)
    predictions_unscaled = test_dataset.scale_batch(predictions, direction="original")
    print(predictions_unscaled)


if __name__ == "__main__":
    main()
