from pathlib import Path
import mlflow
import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import MLFlowLogger
import pandas as pd
import numpy as np
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.metrics import MAE, RMSE, MAPE, MultiHorizonMetric
from pytorch_forecasting.models.temporal_fusion_transformer.tuning import optimize_hyperparameters
from collections import defaultdict
from typing import Optional, Literal
import torch.nn.functional as F

from test.test_bfm_alternate_version.src.data_set import AirQualityDataset

class MSELoss(MultiHorizonMetric):
    """mean squared error loss for tft predictions"""
    def __init__(self, reduction="mean"):
        super().__init__(reduction=reduction)

    def loss(self, y_pred, target):
        if isinstance(y_pred, (tuple, list)):
            y_pred = y_pred[0]
        if isinstance(target, (tuple, list)):
            target = target[0]
            
        y_pred = torch.as_tensor(y_pred)
        target = torch.as_tensor(target)
        
        loss = torch.pow(y_pred - target, 2)
        return loss

class TFTPredictor(pl.LightningModule):
    def __init__(self, feature_names, **kwargs):
        super().__init__()
        self.save_hyperparameters(ignore=['loss', 'logging_metrics'])
        self.feature_names = feature_names
        self.model_params = kwargs
        self.model = None
        self.current_epoch_metrics = defaultdict(list)
        self.train_metrics_history = defaultdict(list)
        self.val_metrics_history = defaultdict(list)
        self.test_step_outputs = []

    def _compute_metrics(self, predictions, targets, prefix: Optional[Literal["train_", "val_", "test_"]] = ""):
        """compute metrics for tft predictions"""
        losses = defaultdict(list)
        metrics = defaultdict(list)
        total_loss = 0

        if isinstance(predictions, list):
            predictions = torch.stack(predictions)
        if isinstance(targets, list):
            targets = torch.stack(targets)

        if isinstance(predictions, torch.Tensor):
            predictions = predictions.squeeze()
        if isinstance(targets, torch.Tensor):
            targets = targets.squeeze()
        
        mse_loss = F.mse_loss(predictions, targets)
        mae_loss = F.l1_loss(predictions, targets)
        
        losses[f"{prefix}mse"] = mse_loss
        metrics[f"{prefix}mae"] = mae_loss
        total_loss = mse_loss
        
        if prefix == "test_":
            rmse = torch.sqrt(mse_loss)
            r2 = 1 - mse_loss / torch.var(targets)
            metrics[f"{prefix}rmse"] = rmse
            metrics[f"{prefix}r2"] = r2

        losses[f"averaged_{prefix}total_loss"] = total_loss
        return losses, metrics

    def setup(self, stage=None):
        """initialize datasets and model"""
        if self.model is None:
            self.prepare_data()
            
            self.model = TemporalFusionTransformer.from_dataset(
                self.train_dataset,
                learning_rate=self.model_params.get("learning_rate", 0.001),
                hidden_size=self.model_params.get("hidden_size", 16),
                attention_head_size=self.model_params.get("attention_head_size", 4),
                dropout=self.model_params.get("dropout", 0.1),
                hidden_continuous_size=self.model_params.get("hidden_continuous_size", 8),
                loss=MSELoss(),
                lstm_layers=2,
                reduce_on_plateau_patience=3,
                static_categoricals=[],
                static_reals=[],
                time_varying_categoricals_encoder=[],
                time_varying_categoricals_decoder=[],
                time_varying_reals_encoder=[
                    f"truth_{name}" for name in self.feature_names["ground_truth"]
                ],
                time_varying_reals_decoder=[
                    f"truth_{name}" for name in self.feature_names["ground_truth"]
                ],
            )
            
            if self.trainer is not None:
                self.model.trainer = self.trainer

    def on_fit_start(self):
        if self.trainer is not None:
            self.model.trainer = self.trainer

    def on_test_start(self):
        if self.trainer is not None:
            self.model.trainer = self.trainer

    def on_validation_start(self):
        if self.trainer is not None:
            self.model.trainer = self.trainer

    def prepare_data(self):
        """convert airqualitydataset to timeseriesset format"""
        aq_dataset = AirQualityDataset(
            xlsx_path=Path(__file__).parent.parent / "data/AirQuality.xlsx",
            sequence_length=48,
            prediction_horizon=1
        )
        
        data_list = []
        
        def sanitize_name(name):
            return name.replace(".", "_").replace("(", "_").replace(")", "_")
        
        for i, (seq, _) in enumerate(aq_dataset):
            row_dict = {
                "time_idx": i,
                "group": 0
            }
            for name, values in seq.ground_truth_vars.items():
                row_dict[f"truth_{sanitize_name(name)}"] = values.numpy()[-1]
            data_list.append(row_dict)
        
        df = pd.DataFrame(data_list)
        
        self.train_dataset = TimeSeriesDataSet(
            df[:int(0.8 * len(df))],
            time_idx="time_idx",
            target=[f"truth_{sanitize_name(name)}" for name in self.feature_names["ground_truth"]],
            group_ids=["group"],
            max_encoder_length=48,
            max_prediction_length=1,
            static_categoricals=[],
            static_reals=[],
            time_varying_known_categoricals=[],
            time_varying_known_reals=[],
            time_varying_unknown_categoricals=[],
            time_varying_unknown_reals=[
                f"truth_{sanitize_name(name)}" for name in self.feature_names["ground_truth"]
            ],
        )
        
        self.val_dataset = TimeSeriesDataSet.from_dataset(
            self.train_dataset,
            df[int(0.8 * len(df)):int(0.9 * len(df))],
            predict=True
        )
        
        self.test_dataset = TimeSeriesDataSet.from_dataset(
            self.train_dataset,
            df[int(0.9 * len(df)):],
            predict=True
        )

    def train_dataloader(self):
        return self.train_dataset.to_dataloader(
            batch_size=8,
            num_workers=16
        )

    def val_dataloader(self):
        return self.val_dataset.to_dataloader(
            batch_size=8,
            num_workers=16
        )

    def test_dataloader(self):
        return self.test_dataset.to_dataloader(
            batch_size=8,
            num_workers=16
        )

    def configure_optimizers(self):
        """Match AQFM's optimizer configuration"""
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.model_params.get("learning_rate", 0.001),
            weight_decay=0.01
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.1,
            patience=10,
            verbose=True
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "averaged_val_total_loss"
            }
        }

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        out = self.model(x)
        predictions = out.prediction
        
        losses, metrics = self._compute_metrics(predictions=predictions, targets=y[0], prefix="train_")
        
        for name, value in {**losses, **metrics}.items():
            if isinstance(value, torch.Tensor):
                self.log(name, value, prog_bar=True if "averaged_train_total_loss" in name else False)
        
        return losses["averaged_train_total_loss"]

    def validation_step(self, batch, batch_idx):
        x, y = batch
        out = self.model(x)
        predictions = out.prediction
        losses, metrics = self._compute_metrics(predictions=predictions, targets=y[0], prefix="val_")
        
        for name, value in {**losses, **metrics}.items():
            if isinstance(value, torch.Tensor):
                self.log(name, value, prog_bar=True if "averaged_val_total_loss" in name else False)
        
        return losses["averaged_val_total_loss"]

    def test_step(self, batch, batch_idx):
        x, y = batch
        out = self.model(x)
        predictions = out.prediction
        
        losses, metrics = self._compute_metrics(predictions=predictions, targets=y[0], prefix="test_")
        
        self.log_dict(losses, prog_bar=True)
        self.log_dict(metrics, prog_bar=True)
        
        if isinstance(predictions, list):
            predictions = torch.stack(predictions)
        if isinstance(y[0], list):
            targets = torch.stack(y[0])
        else:
            targets = y[0]
        
        output = {"predictions": predictions, "targets": targets}
        self.test_step_outputs.append(output)
        return output

    def on_validation_epoch_end(self):
        for metric_name, values in self.current_epoch_metrics.items():
            if "train_" in metric_name:
                epoch_avg = torch.stack(values).mean()
                self.train_metrics_history[metric_name].append(epoch_avg.item())
            elif "val_" in metric_name:
                epoch_avg = torch.stack(values).mean()
                self.val_metrics_history[metric_name].append(epoch_avg.item())
        
        self.current_epoch_metrics.clear()

    def on_test_epoch_end(self):
        outputs = self.test_step_outputs
        test_metrics = {}
        
        try:
            all_predictions = torch.cat([x["predictions"] for x in outputs])
            all_targets = torch.cat([x["targets"] for x in outputs])
            
            all_predictions = all_predictions.squeeze()
            all_targets = all_targets.squeeze()
            
            for idx, name in enumerate(self.feature_names["ground_truth"]):
                if len(all_predictions.shape) > 1:
                    predictions = all_predictions[:, idx].squeeze()
                    targets = all_targets[:, idx].squeeze()
                else:
                    predictions = all_predictions
                    targets = all_targets
                
                test_metrics[f"test_{name}_mse"] = F.mse_loss(predictions, targets)
                test_metrics[f"test_{name}_mae"] = F.l1_loss(predictions, targets)
            
            self.log_dict(test_metrics)
        except Exception as e:
            print(f"Error in on_test_epoch_end: {str(e)}")
            raise
        finally:
            self.test_step_outputs.clear()

    def get_metrics_history(self):
        return {
            **{f"epoch_{k}": np.array(v) for k, v in self.train_metrics_history.items()},
            **{f"epoch_{k}": np.array(v) for k, v in self.val_metrics_history.items()},
        }


def sanitize_name(name):
    return name.replace(".", "_").replace("(", "_").replace(")", "_")

def main():
    remote_server_uri = "http://0.0.0.0:8082"
    experiment_name = "AQFM_TFT_comparison"

    mlflow.set_tracking_uri(remote_server_uri)
    try:
        experiment_id = mlflow.create_experiment(experiment_name)
    except Exception as e:
        experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id

    mlf_logger = MLFlowLogger(
        experiment_name=experiment_name,
        tracking_uri=remote_server_uri,
        run_name="tft_baseline",
    )

    model = TFTPredictor(
        feature_names={
            "sensor": [
                sanitize_name("PT08.S1(CO)"),
                sanitize_name("PT08.S2(NMHC)"),
                sanitize_name("PT08.S3(NOx)"),
                sanitize_name("PT08.S4(NO2)"),
                sanitize_name("PT08.S5(O3)")
            ],
            "ground_truth": [
                sanitize_name("CO(GT)"),
                sanitize_name("NMHC(GT)"),
                sanitize_name("C6H6(GT)"),
                sanitize_name("NOx(GT)"),
                sanitize_name("NO2(GT)")
            ],
            "physical": ["T", "RH", "AH"],
        }
    )
    checkpoint_callback = ModelCheckpoint(
        dirpath="./checkpoints/tft",
        filename="tft_best",
        monitor="averaged_val_total_loss",
        mode="min",
        save_top_k=1,
        save_last=True,
    )
    early_stopping_callback = EarlyStopping(
        monitor="averaged_val_total_loss",
        min_delta=0.001,
        patience=7,
        verbose=True,
        mode="min",
    )

    trainer = pl.Trainer(
        max_epochs=35,
        devices=1,
        accelerator="cuda" if torch.cuda.is_available() else "cpu",
        callbacks=[checkpoint_callback, early_stopping_callback],
        gradient_clip_val=0.1798532667725321,
        logger=mlf_logger,
        enable_progress_bar=True,
        enable_model_summary=True,
        log_every_n_steps=10,
        min_epochs=7
    )

    trainer.fit(model)
    test_results = trainer.test(model)[0]

    print("\nLogging results to MLFlow...aaaaaaaaaa")
    with mlflow.start_run(run_id=mlf_logger.run_id):
        mlflow.log_metrics(test_results)
        if checkpoint_callback.best_model_path:
            print(f"Logging best model from: {checkpoint_callback.best_model_path}")
            mlflow.log_artifact(checkpoint_callback.best_model_path)

    print("\nFinal Results:")
    print("==============")
    print(f"Best model saved at: {checkpoint_callback.best_model_path}")
    print("Test metrics:")
    for metric, value in test_results.items():
        print(f"  {metric}: {value:.4f}")


if __name__ == "__main__":
    main()
