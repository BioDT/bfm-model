"""
Training script for the BFM (Biodiversity Foundation Model).
TODO: Adapt it according to the new data format. The current version was using a toy format, and was just for testing purposes.
"""

from collections import namedtuple
from typing import Union, Optional
from datetime import datetime, timedelta
import os
import functools
import hydra
import lightning as L
import torch
from lightning.pytorch import LightningModule, seed_everything
from lightning.pytorch.loggers import MLFlowLogger
from lightning.pytorch.strategies import FSDPStrategy

from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Dataset
from torch.utils.data._utils.collate import default_collate
from torch.distributed.fsdp.wrap import wrap, size_based_auto_wrap_policy, enable_wrap

from src.bfm.src.bfm import BFM
from src.bfm.src.dataloder import LargeClimateDataset, custom_collate

# Optional: Enable distributed debug logs
# os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"

class BFM_pipe(LightningModule):
    def __init__(self, model, cfg=None, learning_rate=1e-2, weight_decay=5e-6):
        super().__init__()
        self.model = model
        self.cfg = cfg
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        # The variable weights come from: w_var = 1 / standard_dev
        self.variable_weights = {
            "surface_variables": {
                "t2m": 0.13,
                "msl": 0.0011
                # ... add more if surface has more
            },
            "single_variables": {
                "lsm": 2.27
            },
            "atmospheric_variables": {
                "z": 1.22e-5,
                "t": 0.036
            },
            "species_extinction_variables": {
                "ExtinctionValue": 38.0
            },
            "land_variables": {
                "Land": 5.84e-4,
                "NDVI": 19.6
            },
            "agriculture_variables": {
                "AgricultureLand": 0.053,
                "AgricultureIrrLand": 0.0,    # or skip if purely zero
                "ArableLand": 0.085,
                "Cropland": 0.36
            },
            "forest_variables": {
                "Forest": 0.11
            }
        }

    # def configure_model(self) -> None:
    #     model = BFM(
    #             surface_vars=("t2m", "msl"),
    #             single_vars=("lsm",),
    #             atmos_vars=("z", "t"),
    #             species_vars=("ExtinctionValue",),
    #             land_vars=("Land", "NDVI"),
    #             agriculture_vars=("AgricultureLand", "AgricultureIrrLand", "ArableLand", "Cropland"),
    #             forest_vars=("Forest",),
    #             atmos_levels=self.cfg.data.atmos_levels,
    #             H=self.cfg.model.H,
    #             W=self.cfg.model.W,
    #             embed_dim=self.cfg.model.embed_dim,
    #             num_latent_tokens=self.cfg.model.num_latent_tokens,
    #             backbone_type=self.cfg.model.backbone,
    #             patch_size=self.cfg.model.patch_size,
    #         )

    #     #device=self.trainer.strategy.root_device # TODO Need to put the model to devices
    #     self.model = wrap(model) # , device_id=self.trainer.strategy.root_device


    def forward(self, batch, lead_time):
        return self.model(batch, lead_time)

    def training_step(self, batch, batch_idx):
        lead_time = timedelta(hours=6)  # fixed lead time for pre-training
        output = self(batch, lead_time)

        loss = self.compute_loss(output, batch)
        self.log("train_loss", loss)
        return loss

    def compute_loss(self, output, batch):

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
            "forest_variables"
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
                
                #TODO  but ensure your shapes/time dimension logic is consistent for all
                # If it's timeseries with time axis, pick last step if that is the requirement
                target = true_dict[var_name][:, -1]  # shape depends on your model design

                # Default weight = 1.0 if not in dictionary
                w = self.variable_weights.get(group_name, {}).get(var_name, 1.0)

                # L1 loss
                loss_var = torch.mean(torch.abs(pred_tensor - target))
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
    
    # def configure_gradient_clipping(
    #         self,
    #         optimizer,
    #         optimizer_idx: int,
    #         gradient_clip_val: Optional[Union[int, float]] = None,
    #         gradient_clip_algorithm: Optional[str] = None,
    # ):
    #     assert gradient_clip_algorithm in ('norm', None), gradient_clip_algorithm
    #     self.model.clip_grad_norm_(gradient_clip_val)



@hydra.main(version_base=None, config_path="configs", config_name="test_config")
def main(cfg: DictConfig):
    # Setup config
    print(OmegaConf.to_yaml(cfg))

    # Seed the experiment for numpy, torch and python.random.
    seed_everything(42, workers=True)

    print('Setting up Dataloader ...')
    # dataset = AuroraDataset(cfg.model.B, cfg.model.T, cfg.model.V_surf, cfg.model.V_atmos, cfg.model.C, cfg.model.H, cfg.model.W)
    dataset = LargeClimateDataset(data_dir='data/')
    dataloader = DataLoader(
        dataset, batch_size=cfg.training.batch_size, num_workers=cfg.training.workers, collate_fn=custom_collate, drop_last=True)

    # Setup logger
    current_time = datetime.now()
    remote_server_uri = f"http://0.0.0.0:{cfg.mlflow.port}"
    mlf_logger = MLFlowLogger(experiment_name="BFM_logs", run_name=f"BFM_{current_time}", tracking_uri=remote_server_uri)
    # Setup model
    # model = BFM_pipe(cfg=cfg)

    print('Done \n Setting up the BFM')
    my_auto_wrap_policy = functools.partial(
        size_based_auto_wrap_policy, min_num_params=100
    )

    if cfg.training.strategy == "fsdp":
        distr_strategy = FSDPStrategy(sharding_strategy="FULL_SHARD", auto_wrap_policy=size_based_auto_wrap_policy)
    
    model = BFM(
                surface_vars=("t2m", "msl"),
                single_vars=("lsm",),
                atmos_vars=("z", "t"),
                species_vars=("ExtinctionValue",),
                land_vars=("Land", "NDVI"),
                agriculture_vars=("AgricultureLand", "AgricultureIrrLand", "ArableLand", "Cropland"),
                forest_vars=("Forest",),
                atmos_levels=cfg.data.atmos_levels,
                H=cfg.model.H,
                W=cfg.model.W,
                embed_dim=cfg.model.embed_dim,
                num_latent_tokens=cfg.model.num_latent_tokens,
                backbone_type=cfg.model.backbone,
                patch_size=cfg.model.patch_size,
            )
    train_pipe = BFM_pipe(model)

    trainer = L.Trainer(
        max_epochs=cfg.training.epochs,
        accelerator=cfg.training.accelerator,
        devices=cfg.training.devices,
        # fast_dev_run=True,
        # devices=[1],
        # strategy=distr_strategy,
        precision=cfg.training.precision,
        # gradient_clip_val=cfg.training.gradient_clip, # TODO Errors 
        log_every_n_steps=cfg.training.log_steps,
        # logger=mlf_logger,
    )

    # trainer.fit(model, train_dataloaders=dataloader)

    trainer.fit(train_pipe, train_dataloaders=dataloader)


    print("Finished training successfully")
    trainer.print(torch.cuda.memory_summary())

if __name__ == "__main__":
    main()
