from datetime import datetime
import hydra
from omegaconf import DictConfig, OmegaConf

import torch
import lightning as L

from torch.utils.data import DataLoader

from lightning.pytorch import seed_everything
from lightning.pytorch.loggers import MLFlowLogger

from src.bfm.src.bfm_lighting import BFM_lighting
from src.bfm.src.dataloder import LargeClimateDataset, custom_collate



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
        batch_size=cfg.evaluation.batch_size,
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

    trainer = L.Trainer(
        accelerator=cfg.training.accelerator,
        devices=cfg.training.devices,
        precision=cfg.training.precision,
        log_every_n_steps=cfg.training.log_steps,
        logger=mlf_logger,
        enable_checkpointing=False,
        enable_progress_bar=True,
    )

    bfm_model = BFM_lighting(surface_vars=(["t2m", "msl"]),
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
                            embed_dim= cfg.model.embed_dim,
                            num_heads=cfg.model.num_heads,
                            head_dim=cfg.model.head_dim,
                            depth=cfg.model.depth,
                            batch_size=cfg.evaluation.batch_size,)


    checkpoint_path = cfg.evaluation.checkpoint_path
    #Load Model from Checkpoint
    print(f"Loading model from checkpoint: {checkpoint_path}")
    # V1 Do the inference
    test_results = trainer.test(model=bfm_model, ckpt_path=checkpoint_path, dataloaders=test_dataloader)
    # V2 Do the inference
    # loaded_bfm = BFM_lighting.load_from_checkpoint(checkpoint_path,
                                                    # surface_vars=(["t2m", "msl"]),
                                                    # single_vars=(["lsm"]),
                                                    # atmos_vars=(["z", "t"]),
                                                    # species_vars=(["ExtinctionValue"]),
                                                    # species_distr_vars=(["Distribution"]),
                                                    # land_vars=(["Land", "NDVI"]),
                                                    # agriculture_vars=(["AgricultureLand", "AgricultureIrrLand", "ArableLand", "Cropland"]),
                                                    # forest_vars=(["Forest"]),
                                                    # atmos_levels=cfg.data.atmos_levels,
                                                    # species_num=cfg.data.species_number,
                                                    # H=cfg.model.H,
                                                    # W=cfg.model.W,
                                                    # embed_dim=cfg.model.embed_dim,
                                                    # num_latent_tokens=cfg.model.num_latent_tokens,
                                                    # backbone_type=cfg.model.backbone,
                                                    # patch_size=cfg.model.patch_size,
                                                    # learning_rate=cfg.training.lr,
                                                    # weight_decay=cfg.training.wd,
                                                    # batch_size=cfg.evaluation.batch_size,))
    # loaded_bfm.eval()

    # test_results = trainer.test(loaded_bfm, dataloaders=test_dataloader)
    # test_results is typically a list of dicts (one per test dataloader)
    print("=== Test Results ===")
    print(test_results)


if __name__ == "__main__":
    main()
