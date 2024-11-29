"""
Training script for the BFM (Biodiversity Foundation Model).
TODO: Adapt it according to the new data format. The current version was using a toy format, and was just for testing purposes.
"""

from collections import namedtuple
from datetime import datetime, timedelta

import hydra
import lightning as L
import torch
from lightning.pytorch import LightningModule, seed_everything
from lightning.pytorch.loggers import MLFlowLogger
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Dataset
from torch.utils.data._utils.collate import default_collate

from src.bfm.src.bfm import BFM

Batch = namedtuple("Batch", ["surf_vars", "static_vars", "atmos_vars", "metadata"])
Metadata = namedtuple("Metadata", ["lat", "lon", "time", "atmos_levels"])


def custom_collate(batch):
    elem = batch[0]
    if isinstance(elem, Batch):
        return Batch(
            surf_vars={k: custom_collate([d.surf_vars[k] for d in batch]) for k in elem.surf_vars},
            static_vars=elem.static_vars,  # Assuming static_vars are the same for all batch elements
            atmos_vars={k: custom_collate([d.atmos_vars[k] for d in batch]) for k in elem.atmos_vars},
            metadata=Metadata(
                lat=elem.metadata.lat,
                lon=elem.metadata.lon,
                time=[d.metadata.time for d in batch],
                atmos_levels=elem.metadata.atmos_levels,
            ),
        )
    elif isinstance(elem, (list, tuple)):
        return [custom_collate(samples) for samples in zip(*batch)]
    elif isinstance(elem, dict):
        return {key: custom_collate([d[key] for d in batch]) for key in elem}
    else:
        return default_collate(batch)


class AuroraDataset(Dataset):
    def __init__(self, B, T, V_s, V_a, C, H, W):
        self.B = B
        self.T = T
        self.V_s = V_s
        self.V_a = V_a
        self.C = C
        self.H = H
        self.W = W

        self.surf_vars = {f"surf_var_{i}": torch.randn(B, T, H, W) for i in range(V_s)}
        self.static_vars = {f"static_var_{i}": torch.randn(H, W) for i in range(2)}
        self.atmos_vars = {f"atmos_var_{i}": torch.randn(B, T, C, H, W) for i in range(V_a)}

        self.lat = torch.linspace(-90, 90, H)
        self.lon = torch.linspace(0, 360, W)
        self.time = [datetime.now() + timedelta(hours=i) for i in range(T)]
        self.atmos_levels = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50]

    def __len__(self):
        return self.B

    def __getitem__(self, idx):
        metadata = Metadata(lat=self.lat, lon=self.lon, time=self.time, atmos_levels=self.atmos_levels)

        return Batch(
            surf_vars={k: v[idx] for k, v in self.surf_vars.items()},
            static_vars=self.static_vars,
            atmos_vars={k: v[idx] for k, v in self.atmos_vars.items()},
            metadata=metadata,
        )


class BFMTrainer(LightningModule):
    def __init__(self, model, learning_rate=5e-4, weight_decay=5e-6):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        # define weights for surface and atmospheric variables
        self.w_S = {"surf_var_0": 1.5, "surf_var_1": 0.77, "surf_var_2": 0.66}
        self.w_A = {f"atmos_var_{i}": 1.0 for i in range(5)}  # assuming 5 atmospheric variables

        # weights for loss functions per data set
        self.gamma = {"ERA5": 2.0, "GFS-T0": 1.5}

        self.alpha = 0.25  # weight for surface variables
        self.beta = 1.0  # weight for atmospheric variables

    def forward(self, batch, lead_time):
        return self.model(batch, lead_time)

    def training_step(self, batch, batch_idx):
        lead_time = timedelta(hours=6)  # fixed lead time for pre-training
        output = self(batch, lead_time)

        loss = self.compute_loss(output, batch)
        self.log("train_loss", loss)
        return loss

    def compute_loss(self, output, batch):
        loss_S = 0
        for k, v in output["surf_vars"].items():
            target = batch.surf_vars[k][:, -1]  # last time step
            loss_S += self.w_S.get(k, 1.0) * torch.mean(torch.abs(v - target))
        loss_S /= len(output["surf_vars"])

        loss_A = 0
        for k, v in output["atmos_vars"].items():
            target = batch.atmos_vars[k][:, -1]  # also last time step
            loss_A += self.w_A.get(k, 1.0) * torch.mean(torch.abs(v - target))
        loss_A /= len(output["atmos_vars"])

        # assuming all data in a batch is from the same dataset
        gamma = self.gamma.get("ERA5", 1.0)  # default to 1.0 if dataset not specified

        loss = gamma * (self.alpha * loss_S + self.beta * loss_A)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=150000, eta_min=self.learning_rate / 10)
        return [optimizer], [scheduler]


@hydra.main(version_base=None, config_path="configs", config_name="test_config")
def main(cfg: DictConfig):
    # Setup config
    print(OmegaConf.to_yaml(cfg))
    # hyperparams -> moved to config file
    # TODO Adapt this way of handling parameters
    # B, T, V_s, V_a, C, H, W = 32, 2, 3, 5, 13, 32, 64
    # embed_dim = 1024
    # num_latent_tokens = 7

    # Seed the experiment for numpy, torch and python.random.
    seed_everything(42, workers=True)

    dataset = AuroraDataset(cfg.model.B, cfg.model.T, cfg.model.V_s, cfg.model.V_a, cfg.model.C, cfg.model.H, cfg.model.W)
    dataloader = DataLoader(
        dataset, batch_size=cfg.training.batch_size, num_workers=cfg.training.workers, collate_fn=custom_collate
    )

    model = BFM(
        surf_vars=tuple(f"surf_var_{i}" for i in range(cfg.model.V_s)),
        static_vars=tuple(f"static_var_{i}" for i in range(2)),
        atmos_vars=tuple(f"atmos_var_{i}" for i in range(cfg.model.V_a)),
        atmos_levels=cfg.data.atmos_levels,
        H=cfg.model.H,
        W=cfg.model.W,
        embed_dim=cfg.model.embed_dim,
        num_latent_tokens=cfg.model.num_latent_tokens,
        patch_size=cfg.model.patch_size,
    )

    # Setup logger
    current_time = datetime.now()
    remote_server_uri = f"http://0.0.0.0:{cfg.mlflow.port}"
    mlf_logger = MLFlowLogger(experiment_name="BFM_logs", run_name=f"BFM_{current_time}", tracking_uri=remote_server_uri)
    # Setup trainer
    trainer = BFMTrainer(model)

    pl_trainer = L.Trainer(
        max_epochs=cfg.training.epochs,
        accelerator=cfg.training.accelerator,
        devices=cfg.training.devices,
        precision=cfg.training.precision,
        gradient_clip_val=cfg.training.gradient_clip,
        log_every_n_steps=cfg.training.log_steps,
        logger=mlf_logger,
    )

    pl_trainer.fit(trainer, train_dataloaders=dataloader)


if __name__ == "__main__":
    main()
