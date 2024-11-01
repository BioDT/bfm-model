from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from test.test_bfm_alternate_version.src.aqfm.aqfm import AQFM
from test.test_bfm_alternate_version.src.data_set import (
    AirQualityDataset,
    collate_aq_batches,
)
from typing import Literal, Optional

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from lightning.pytorch.loggers import MLFlowLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader


class AQFMPredictor(pl.LightningModule):
    def __init__(
        self,
        feature_names,
        embed_dim: int = 128,
        num_latent_tokens: int = 8,
        backbone_type: str = "swin",
        max_history_size: int = 24,
        learning_rate: float = 1e-4,
        checkpoint_path: Optional[str] = None,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()  # saves hyperparameters in a checkpoint and in self.hparams

        # load from checkpoint if provided
        if checkpoint_path:
            self.load_from_checkpoint(checkpoint_path)
        else:
            self.model = AQFM(
                feature_names=feature_names,
                embed_dim=embed_dim,
                num_latent_tokens=num_latent_tokens,
                backbone_type=backbone_type,
                max_history_size=max_history_size,
                **kwargs,
            )

        self.learning_rate = learning_rate
        self.feature_names = feature_names

        # metrics storage
        self.train_metrics_history = defaultdict(list)
        self.val_metrics_history = defaultdict(list)
        self.current_epoch_metrics = defaultdict(list)
        self.test_step_outputs = []

    def forward(self, batch):
        return self.model(batch, timedelta(hours=1))

    def _compute_metrics(
        self,
        predictions,
        targets,
        prefix: Optional[Literal["train_", "val_", "test_"]] = "",  # what kind of metrics are we computing? train, test, val?
    ):
        losses = defaultdict(list)
        metrics = defaultdict(list)
        total_loss = 0

        for name in targets.keys():
            prediction = predictions[name].cpu()  # TODO check if needs detachment and move to cpu
            true = targets[name].cpu()

            mse_loss = F.mse_loss(prediction, true)
            mae_loss = F.l1_loss(prediction, true)

            losses[f"{prefix}{name}_mse"] = mse_loss
            metrics[f"{prefix}{name}_mae"] = mae_loss

            total_loss += mse_loss

            if prefix == "test_":
                rmse = np.sqrt(mean_squared_error(true.numpy(), prediction.numpy()))
                r2 = r2_score(true.numpy(), prediction.numpy())
                metrics[f"{prefix}{name}_rmse"] = rmse
                metrics[f"{prefix}{name}_r2"] = r2

        losses[f"averaged_{prefix}total_loss"] = total_loss / len(targets)
        return losses, metrics

    def training_step(self, batch_obj, _):
        batch, targets = batch_obj
        predictions = self(batch)

        losses, metrics = self._compute_metrics(predictions=predictions, targets=targets, prefix="train_")
        self.logger.log_metrics(metrics)
        for name, value in {**losses, **metrics}.items():
            if isinstance(value, torch.Tensor):
                # cleaning name for MLFlow
                sanitized_name = name.replace("(", "_").replace(")", "_")
                self.logger.experiment.log_metric(self.logger.run_id, sanitized_name, value.item(), step=self.current_epoch)
                # keeping original name for our metrics history
                self.current_epoch_metrics[name].append(value.detach().cpu())

        return losses["averaged_train_total_loss"]

    def on_train_epoch_end(self):
        print("\nTraining metrics at epoch end:", self.current_epoch_metrics.keys())
        pass

    def validation_step(self, batch_obj, _):
        batch, targets = batch_obj
        predictions = self(batch)

        losses, metrics = self._compute_metrics(predictions=predictions, targets=targets, prefix="val_")

        # logging the monitored metric for ModelCheckpoint
        self.log("averaged_val_total_loss", losses["averaged_val_total_loss"], prog_bar=True)
        print(
            f"Logger info: {self.logger.experiment.get_run(self.logger.run_id)}"
            f"\nRun ID: {self.logger.run_id}"
            f"\nStep: {self.current_epoch}"
            f"\n Logger experiment: {self.logger.experiment}"
        )

        # MLFlow logging
        for name, value in {**losses, **metrics}.items():
            if isinstance(value, torch.Tensor):
                sanitized_name = name.replace("(", "_").replace(")", "_")
                self.logger.experiment.log_metric(self.logger.run_id, sanitized_name, value.item(), step=self.current_epoch)
                self.current_epoch_metrics[name].append(value.detach().cpu())

        return losses["averaged_val_total_loss"]

    def on_validation_epoch_end(self):
        print("\nValidation metrics at epoch end:", self.current_epoch_metrics.keys())

        for metric_name, values in self.current_epoch_metrics.items():
            if "train_" in metric_name:
                epoch_avg = torch.stack(values).mean()
                self.train_metrics_history[metric_name].append(epoch_avg.item())
            elif "val_" in metric_name:
                epoch_avg = torch.stack(values).mean()
                self.val_metrics_history[metric_name].append(epoch_avg.item())

        self.current_epoch_metrics.clear()

    def test_step(self, batch_obj, _):
        batch, targets = batch_obj
        predictions = self(batch)

        losses, metrics = self._compute_metrics(predictions=predictions, targets=targets, prefix="test_")

        self.log_dict(losses, prog_bar=True)
        self.log_dict(metrics, prog_bar=True)

        output = {"predictions": predictions, "targets": targets}
        self.test_step_outputs.append(output)
        return output

    def on_test_epoch_end(self):
        outputs = self.test_step_outputs

        test_metrics = {}
        for name in self.feature_names["ground_truth"]:
            predictions = torch.cat([x["predictions"][name] for x in outputs])
            targets = torch.cat([x["targets"][name] for x in outputs])

            test_metrics[f"test_{name}_mse"] = F.mse_loss(predictions, targets)
            test_metrics[f"test_{name}_mae"] = F.l1_loss(predictions, targets)
        self.log_dict(test_metrics)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer, mode="min", factor=0.2, patience=3, verbose=True
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "monitor": "averaged_val_total_loss",
            },
        }

    def get_metrics_history(self):
        return {
            **{f"epoch_{k}": np.array(v) for k, v in self.train_metrics_history.items()},
            **{f"epoch_{k}": np.array(v) for k, v in self.val_metrics_history.items()},
        }


def main():
    # logging
    current_time = datetime.now()
    remote_server_uri = "http://0.0.0.0:5000"  # default MLFlow port
    mlf_logger = MLFlowLogger(experiment_name="AQFM_logs", run_name=f"AQFM_{current_time}", tracking_uri=remote_server_uri)

    # making a data set just as in encoder.py, data_set.py, decoder.py, and now here as well
    data_params = {
        "xlsx_path": Path(__file__).parent.parent / "data/AirQuality.xlsx",
        "sequence_length": 24,
        "prediction_horizon": 1,
        "feature_groups": {
            "sensor": ["PT08.S1(CO)", "PT08.S2(NMHC)", "PT08.S3(NOx)", "PT08.S4(NO2)", "PT08.S5(O3)"],
            "ground_truth": ["CO(GT)", "NMHC(GT)", "C6H6(GT)", "NOx(GT)", "NO2(GT)"],
            "physical": ["T", "RH", "AH"],
        },
    }

    train_dataset = AirQualityDataset(**data_params, mode="train")
    val_dataset = AirQualityDataset(**data_params, mode="val", scalers=train_dataset.get_scalers())
    test_dataset = AirQualityDataset(**data_params, mode="test", scalers=train_dataset.get_scalers())

    # setup data loading
    batch_size = 32
    train_loader = DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_aq_batches, num_workers=16
    )

    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, collate_fn=collate_aq_batches, num_workers=16)

    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, collate_fn=collate_aq_batches, num_workers=16)  # noqa

    # create model instance
    checkpoint_path = None
    da_model = AQFMPredictor(
        feature_names=data_params["feature_groups"],
        embed_dim=512,
        num_latent_tokens=8,
        backbone_type="swin",
        max_history_size=24,
        checkpoint_path=checkpoint_path,
    )

    # checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath="./checkpoints",
        filename="aqfm-{epoch:02d}-{val_total_loss:.2f}",
        monitor="averaged_val_total_loss",
        mode="min",
        save_top_k=3,
        save_last=True,  # also save latest model
        auto_insert_metric_name=False,
    )

    trainer = pl.Trainer(
        max_epochs=10,
        devices=1 if torch.cuda.is_available() else None,
        accelerator="gpu" if torch.cuda.is_available() else None,
        callbacks=[checkpoint_callback],
        gradient_clip_val=0.5,
        log_every_n_steps=10,
        enable_progress_bar=True,
        enable_model_summary=True,
        logger=mlf_logger,
    )
    # train / test
    trainer.fit(da_model, train_loader, val_loader)
    # trainer.test(da_model, test_loader)  # not testing for now - just training and validation tracking

    from test.test_bfm_alternate_version.src.visualization import TimeSeriesVisualizer

    visualizer = TimeSeriesVisualizer(
        model=da_model,
        dataset=val_dataset,
        sequence_length=data_params["sequence_length"],
        prediction_horizon=data_params["prediction_horizon"],
    )

    save_dir = Path("visualization_results")
    visualizer.plot_variable_predictions(save_dir)
    visualizer.plot_feature_correlations(save_dir)

    training_history = da_model.get_metrics_history()

    print("Available metrics:", list(training_history.keys()))

    if training_history:
        visualizer.plot_training_metrics(training_history, save_dir)
    else:
        print("No training metrics available for plotting :(((")


if __name__ == "__main__":
    main()
