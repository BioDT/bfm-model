from collections import namedtuple
from datetime import datetime, timedelta
from typing import Literal
import hydra
import os
import glob
import mlflow.pytorch
from mlflow import MlflowClient

from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

import torch
from torch.utils.data import DataLoader
from torch.distributed.fsdp.wrap import enable_wrap, size_based_auto_wrap_policy, wrap

import lightning as L
from lightning.pytorch import LightningModule, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import MLFlowLogger
from lightning.pytorch.utilities.model_summary import ModelSummary
from lightning.pytorch.strategies import DDPStrategy, FSDPStrategy

from src.bfm.src.decoder import BFMDecoder
from src.bfm.src.encoder import BFMEncoder
from src.mvit.mvit_model import MViT
from src.swin_transformer.core.swim_core_v2 import Swin3DTransformer

from src.bfm.src.dataloder import LargeClimateDataset, custom_collate
from src.bfm.src.utils import save_run_id

DEVICE = "cuda:0"

class BFM_lighting(LightningModule):
    """
    Biodiversity Foundation Model.

    This model combines encoder, backbone and decoder components to process climate and biodiversity-related variables.

    Args:
        surface_vars (tuple[str, ...]): Names of surface-level variables
        single_vars (tuple[str, ...]): Names of single-level variables
        atmos_vars (tuple[str, ...]): Names of atmospheric variables
        species_vars (tuple[str, ...]): Names of species-related variables
        species_distr_vars (tuple[str, ...]): Names of species distributions-related variables
        land_vars (tuple[str, ...]): Names of land-related variables
        agriculture_vars (tuple[str, ...]): Names of agriculture-related variables
        forest_vars (tuple[str, ...]): Names of forest-related variables
        atmos_levels (list[int]): Pressure levels for atmospheric variables
        species_num (int): Number of species distribution to account for
        H (int, optional): Height of output grid. Defaults to 32.
        W (int, optional): Width of output grid. Defaults to 64.
        num_latent_tokens (int, optional): Number of latent tokens. Defaults to 8.
        backbone_type (Literal["swin", "mvit"], optional): Type of backbone architecture. Defaults to "mvit".
        patch_size (int, optional): Size of spatial patches. Defaults to 4.
        embed_dim (int, optional): Embedding dimension. Defaults to 1024.
        num_heads (int, optional): Number of attention heads. Defaults to 16.
        head_dim (int, optional): Dimension of each attention head. Defaults to 64.
        depth (int, optional): Number of transformer layers. Defaults to 2.
        **kwargs: Additional arguments passed to encoder and decoder

    Attributes:
        encoder (BFMEncoder): Encoder component
        backbone (nn.Module): Backbone network (Swin or MViT)
        decoder (BFMDecoder): Decoder component
        backbone_type (str): Type of backbone being used
    """

    def __init__(
        self,
        surface_vars: tuple[str, ...],
        single_vars: tuple[str, ...],
        atmos_vars: tuple[str, ...],
        species_vars: tuple[str, ...],
        species_distr_vars: tuple[str, ...],
        land_vars: tuple[str, ...],
        agriculture_vars: tuple[str, ...],
        forest_vars: tuple[str, ...],
        atmos_levels: list[int],
        species_num: int,
        H: int = 32,
        W: int = 64,
        num_latent_tokens: int = 8,
        backbone_type: Literal["swin", "mvit"] = "mvit",
        patch_size: int = 4,
        embed_dim: int = 1024,
        num_heads: int = 16,
        head_dim: int = 2,
        depth: int = 2,
        learning_rate=1e-3, 
        weight_decay=5e-6, 
        batch_size=1,
        **kwargs,
    ):
        super().__init__()
        self.H = H
        self.W = W

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size

        # The variable weights come from: w_var = 1 / standard_dev
        self.variable_weights = {
            "surface_variables": {
                "t2m": 0.13,
                "msl": 0.00011,
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
            "species_variables": {"Distribution": 1},
        }

        self.encoder = BFMEncoder(
            surface_vars=surface_vars,
            single_vars=single_vars,
            atmos_vars=atmos_vars,
            species_vars=species_vars,
            species_distr_vars=species_distr_vars,
            land_vars=land_vars,
            agriculture_vars=agriculture_vars,
            forest_vars=forest_vars,
            atmos_levels=atmos_levels,
            species_num=species_num,
            H=H,
            W=W,
            patch_size=patch_size,
            embed_dim=embed_dim,
            num_heads=num_heads,
            head_dim=head_dim,
            depth=depth,
            **kwargs,
        )

        patch_shape = (num_latent_tokens, H // self.encoder.patch_size, W // self.encoder.patch_size)

        if backbone_type == "swin":
            self.backbone = Swin3DTransformer(
                embed_dim=embed_dim,
                encoder_depths=(2, 2),
                encoder_num_heads=(8, 16),
                decoder_depths=(2, 2),
                decoder_num_heads=(32, 16),
                window_size=(1, 1, 2),
                mlp_ratio=4.0,
                qkv_bias=True,
                drop_rate=0.0,
                attn_drop_rate=0.0,
                drop_path_rate=0.1,
                use_lora=False,
            )
        elif backbone_type == "mvit":
            self.backbone = MViT(
                patch_shape=patch_shape,
                embed_dim=embed_dim,
                depth=4,
                num_heads=1,
                mlp_ratio=4.0,
                qkv_bias=True,
                path_drop_rate=0.1,
                attn_mode="conv",
                pool_first=False,
                rel_pos=False,
                zero_init_rel=True,
                res_pool=True,
                dim_mul_attn=False,
                dim_scales=[(i, 1.0) for i in range(4)],  # No dimension change
                head_scales=[(1, 2.0), (2, 2.0)],  # Keep head scaling for attention
                pool_kernel=[1, 1, 1],
                kv_stride=[1, 1, 1],
                q_stride=[(0, [1, 1, 1]), (1, [1, 1, 1]), (2, [1, 1, 1])],
            )
        else:
            raise ValueError(f"Unknown backbone type: {backbone_type}")

        self.backbone_type = backbone_type
        self.decoder = BFMDecoder(
            surface_vars=surface_vars,
            single_vars=single_vars,
            atmos_vars=atmos_vars,
            species_vars=species_vars,
            species_distr_vars=species_distr_vars,
            land_vars=land_vars,
            agriculture_vars=agriculture_vars,
            forest_vars=forest_vars,
            atmos_levels=atmos_levels,
            species_num=species_num,
            H=H,
            W=W,
            patch_size=patch_size,
            embed_dim=embed_dim,
            num_heads=num_heads,
            head_dim=head_dim,
            depth=depth,
            **kwargs,
        )

    def forward(self, batch, lead_time, batch_size):
        """
        Forward pass of the model.

        Args:
            batch: Batch object containing input variables and metadata
            lead_time (timedelta): Time difference between input and target

        Returns:
            dict: Dictionary containing decoded outputs for each variable category

        """
        print(f"BFM batch size: {batch_size}")
        ### V1 
        encoded = self.encoder(batch, lead_time, batch_size)
        print("Encoded shape", encoded.shape)

        # calculate number of patches in 2D
        num_patches_h = self.H // self.encoder.patch_size
        num_patches_w = self.W // self.encoder.patch_size
        total_patches = num_patches_h * num_patches_w  # noqa

        # calculate depth to match the sequence length
        depth = encoded.shape[1] // (num_patches_h * num_patches_w)
        print(f"BFM depth: {depth} | patch_size {self.encoder.patch_shape} | encoder shape {encoded.shape}")
        patch_shape = (
            depth,  # depth dimension matches sequence length / (H*W)
            num_patches_h,  # height in patches
            num_patches_w,  # width in patches
        )

        if self.backbone_type == "mvit":
            encoded = encoded.view(encoded.size(0), -1, self.encoder.embed_dim)
            print(f"Reshaped encoded for MViT: {encoded.shape}")

        backbone_output = self.backbone(encoded, lead_time=lead_time, rollout_step=0, patch_shape=patch_shape)
        print("Backbone output", backbone_output.shape)
        # decode
        output = self.decoder(backbone_output, batch, lead_time)
        # print("Decoded output:", output)
        return output

    def validation_step(self, batch, batch_idx):
        lead_time = timedelta(hours=6)  # fixed lead time for pre-training
        output = self(batch, lead_time, batch_size=self.batch_size)

        loss = self.compute_loss(output, batch)
        self.log("val_loss", loss, batch_size=self.batch_size)
        return loss

    def training_step(self, batch, batch_idx):
        lead_time = timedelta(hours=6)  # fixed lead time for pre-training
        output = self(batch, lead_time, batch_size=self.batch_size)

        loss = self.compute_loss(output, batch)
        self.log("train_loss", loss, batch_size=self.batch_size)
        return loss

    def test_step(self, batch, batch_idx):
        lead_time = timedelta(hours=6)  # fixed lead time for pre-training
        output = self(batch, lead_time, batch_size=self.batch_size)

        loss = self.compute_loss(output, batch)
        self.log("test_loss", loss, batch_size=self.batch_size)
        return loss

    def compute_loss(self, output, batch):

        total_loss = 0.0
        count = 0

        groups = [
            "surface_variables",
            "single_variables",
            "atmospheric_variables",
            "species_extinction_variables",
            "land_variables",
            "agriculture_variables",
            "forest_variables",
            "species_variables",
        ]

        for group_name in groups:
            # If group doesn't exist in output or batch, skip
            if group_name not in output or group_name not in batch._asdict():
                continue

            pred_dict = output[group_name]
            true_dict = getattr(batch, group_name)

            group_loss = 0.0
            var_count = 0

            for var_name, pred_tensor in pred_dict.items():
                # If var_name not in the ground truth dict, skip
                if var_name not in true_dict:
                    print(f"{var_name} not in true_dict")
                    continue

                gt_tensor = true_dict[var_name]
                # Log the original shape for debugging.
                # print(f"Original target shape for '{var_name}': {gt_tensor.shape}")
                target = gt_tensor[:, 1]
                # print(f"Selected target shape for '{var_name}': {target.shape}")
                # print(f"Prediction shape for '{var_name}': {pred_tensor.shape}")
                # Compute the L1 loss (reducing all dimensions to produce a scalar).
                loss_var = torch.mean(torch.abs(pred_tensor - target))

                # Determine the weight for this variable.
                group_weights = self.variable_weights.get(group_name, {})
                if isinstance(group_weights, dict):
                    w = group_weights.get(var_name, 1.0)
                else:
                    w = group_weights
                    
                # Log each variable's raw loss
                self.log(f"{var_name} raw loss", loss_var, batch_size=target.size(0))
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

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=150000, eta_min=self.learning_rate / 10)
        return [optimizer], [scheduler]

class OptimizerParamsLogger(L.Callback):
    def on_train_start(self, trainer, pl_module):
        # Ensure the logger is an MLFlowLogger instance.
        if not isinstance(trainer.logger, MLFlowLogger):
            return

        mlflow_logger = trainer.logger
        run_id = mlflow_logger.run_id
        # Loop over each optimizer
        for opt_idx, optimizer in enumerate(trainer.optimizers):
            for pg_idx, param_group in enumerate(optimizer.param_groups):
                lr = param_group.get("lr")
                wd = param_group.get("weight_decay")
                # Log each parameter with a unique key.
                mlflow_logger.experiment.log_param(run_id, 
                    f"optimizer_{opt_idx}_group_{pg_idx}_lr", lr)
                mlflow_logger.experiment.log_param(run_id, 
                    f"optimizer_{opt_idx}_group_{pg_idx}_weight_decay", wd)


@hydra.main(version_base=None, config_path="configs", config_name="train_config")
def main(cfg):
    # Setup config
    print(OmegaConf.to_yaml(cfg))
    # Set the internal precision for TensorCores (H100s)
    torch.set_float32_matmul_precision(cfg.training.precision_in)

    # Seed the experiment for numpy, torch and python.random.
    seed_everything(42, workers=True)

    output_dir = HydraConfig.get().runtime.output_dir

    print("Setting up Dataloader ...")
    dataset = LargeClimateDataset(data_dir=cfg.data.data_path, num_species=cfg.data.species_number)
    test_dataset = LargeClimateDataset(data_dir=cfg.data.test_data_path, num_species=cfg.data.species_number)  # Adapt

    val_dataloader = DataLoader(
        test_dataset,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.training.workers,
        collate_fn=custom_collate,
        drop_last=True,
        shuffle=False,
    )

    train_dataloader = DataLoader(
        dataset,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.training.workers,
        collate_fn=custom_collate,
        drop_last=True,
        shuffle=True,
        pin_memory=True,
    )

    # Setup logger
    current_time = datetime.now()
    remote_server_uri = f"http://0.0.0.0:{cfg.mlflow.port}"
    mlf_logger = MLFlowLogger(experiment_name="BFM_logs", 
                              run_name=f"BFM_{current_time}",
                              tracking_uri=f"{output_dir}/logs")

    logger_run_id = mlf_logger.run_id
    # save_run_id(f"{output_dir}/logs/run_id.txt", logger_run_id)

    print("Done \n Setting up the BFM")
    BFM = BFM_lighting(
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
        embed_dim= cfg.model.embed_dim,
        num_heads=cfg.model.num_heads,
        head_dim=cfg.model.head_dim,
        depth=cfg.model.depth,
        learning_rate=cfg.training.lr,
        weight_decay=cfg.training.wd,
        batch_size=cfg.training.batch_size,
    )
    
    model_summary = ModelSummary(BFM, max_depth=2)
    print(model_summary)
    # /scratch-shared/<username>
    checkpoint_callback = ModelCheckpoint(dirpath=f"{output_dir}/checkpoints", 
                                          save_top_k=1, 
                                          monitor="val_loss",
                                          mode="min")
    
    print(f"Will be saving checkpoints at: {output_dir}/checkpoints")

    if cfg.training.strategy == "fsdp":
        distr_strategy = FSDPStrategy(sharding_strategy="FULL_SHARD", auto_wrap_policy=size_based_auto_wrap_policy)
    elif cfg.training.strategy == "ddp":
        distr_strategy = DDPStrategy()

    trainer = L.Trainer(
        max_epochs=cfg.training.epochs,
        accelerator=cfg.training.accelerator,
        devices=cfg.training.devices,
        precision=cfg.training.precision,
        strategy=distr_strategy,
        log_every_n_steps=cfg.training.log_steps,
        logger=mlf_logger,
        check_val_every_n_epoch=1,  # Do eval every 1 epochs
        # val_check_interval=100    # Do eval every 100 training steps => 100 steps x 5 batch_size = Every 500 Batches
        # limit_train_batches=len(train_dataloader),
        callbacks=[checkpoint_callback, OptimizerParamsLogger()],
    )
    # Experimental
    # mlflow.set_tracking_uri(output_dir)
    # Auto log all MLflow entities
    # mlflow.pytorch.autolog()

    # with mlflow.start_run() as run:
    if cfg.training.checkpoint_path:
            print(f"Loading and resuming training from {cfg.training.checkpoint_path}")
            trainer.fit(BFM, train_dataloaders=train_dataloader, 
                        val_dataloaders=val_dataloader,
                        ckpt_path=cfg.training.checkpoint_path)
    else:
        print("Start training from scratch")
        trainer.fit(BFM, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

    print("Finished training successfully - Lets do a Test!")
    # Manualy save a checkpoint after the end of training
    # trainer.save_checkpoint("test.ckpt")

    # print_auto_logged_info(mlflow.get_run(run_id=run.info.run_id))

    trainer.test(ckpt_path="best", dataloaders=val_dataloader)

    print("Finished testing successfully")
    trainer.print(torch.cuda.memory_summary())

if __name__ == '__main__':
    main()