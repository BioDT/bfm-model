import json
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from test.test_bfm_alternate_version.src.aqfm.aqfm import AQFM
from test.test_bfm_alternate_version.src.data_set import (
    AirQualityDataset,
    collate_aq_batches,
)
from typing import Literal, Optional

import lightning.pytorch as pl
import mlflow
import numpy as np
import optuna
import torch
import torch.nn.functional as F
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from lightning.pytorch.loggers import MLFlowLogger
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader

# sequence lengths to try during optimization
SEQUENCE_LENGTHS = [48, 72, 96, 120, 144]


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
        # encoder params
        encoder_num_heads: int = 16,
        encoder_head_dim: int = 64,
        encoder_depth: int = 2,
        encoder_drop_rate: float = 0.1,
        encoder_mlp_ratio: float = 4.0,
        # backbone params
        backbone_depth: int = 4,
        backbone_num_heads: int = 1,
        backbone_mlp_ratio: float = 4.0,
        backbone_drop_rate: float = 0.1,
        # decoder params
        decoder_num_heads: int = 16,
        decoder_head_dim: int = 64,
        decoder_depth: int = 2,
        decoder_drop_rate: float = 0.1,
        decoder_mlp_ratio: float = 4.0,
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
                encoder_num_heads=encoder_num_heads,
                encoder_head_dim=encoder_head_dim,
                encoder_depth=encoder_depth,
                encoder_drop_rate=encoder_drop_rate,
                encoder_mlp_ratio=encoder_mlp_ratio,
                backbone_depth=backbone_depth,
                backbone_num_heads=backbone_num_heads,
                backbone_mlp_ratio=backbone_mlp_ratio,
                backbone_drop_rate=backbone_drop_rate,
                decoder_num_heads=decoder_num_heads,
                decoder_head_dim=decoder_head_dim,
                decoder_depth=decoder_depth,
                decoder_drop_rate=decoder_drop_rate,
                decoder_mlp_ratio=decoder_mlp_ratio,
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
        """compute mse, mae and other metrics for predictions vs targets"""
        losses = defaultdict(list)
        metrics = defaultdict(list)
        total_loss = 0

        for name in targets.keys():
            prediction = predictions[name].cpu()
            true = targets[name].cpu()

            # Clean metric names for MLFlow
            clean_name = name.replace("(", "_").replace(")", "_")

            mse_loss = F.mse_loss(prediction, true)
            mae_loss = F.l1_loss(prediction, true)

            losses[f"{prefix}{clean_name}_mse"] = mse_loss
            metrics[f"{prefix}{clean_name}_mae"] = mae_loss

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

        batch_size = targets[list(targets.keys())[0]].size(0)
        for name, value in {**losses, **metrics}.items():
            if isinstance(value, torch.Tensor):
                sanitized_name = name.replace("(", "_").replace(")", "_")
                self.log(sanitized_name, value, batch_size=batch_size)

        return losses["averaged_train_total_loss"]

    def validation_step(self, batch_obj, _):
        batch, targets = batch_obj
        predictions = self(batch)

        losses, metrics = self._compute_metrics(predictions=predictions, targets=targets, prefix="val_")

        batch_size = targets[list(targets.keys())[0]].size(0)

        for name, value in {**losses, **metrics}.items():
            if isinstance(value, torch.Tensor):
                sanitized_name = name.replace("(", "_").replace(")", "_")
                self.log(
                    sanitized_name, value, batch_size=batch_size, prog_bar=True if "averaged_val_total_loss" in name else False
                )

        return losses["averaged_val_total_loss"]

    def on_validation_epoch_end(self):
        # compute epoch averages for metrics
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


class SimpleModelPruning(Callback):
    """callback for early stopping trials that aren't promising"""

    def __init__(self, trial, monitor="averaged_val_total_loss"):
        super().__init__()
        self._trial = trial
        self.monitor = monitor
        self._epoch = 0

    def on_validation_end(self, trainer, pl_module):
        self._epoch += 1
        metrics = trainer.callback_metrics
        current_score = metrics.get(self.monitor)
        if current_score is None:
            return

        self._trial.report(current_score.item(), step=self._epoch)
        if self._trial.should_prune():
            raise optuna.TrialPruned()


def generate_trial_name(trial):
    """generate descriptive name for the trial"""
    return (
        f"trial_{trial.number}_"
        f"backbone={trial.params['backbone_type']}_"
        f"seq{trial.params['sequence_length']}_"
        f"dim{trial.params['embed_dim']}_"
        f"tokens{trial.params['num_latent_tokens']}"
    )


def save_best_hyperparameters(study, save_dir):
    """save best hyperparameters from study to json"""
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)

    best_params = study.best_trial.params
    best_value = study.best_trial.value

    backbone_type = best_params.get("backbone_type")

    save_dict = {
        "best_parameters": best_params,
        "best_validation_loss": best_value,
        "parameters_amount": study.best_trial.user_attrs["n_parameters"],
        "optimization_finished": datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        "n_trials": len(study.trials),
        "study_name": study.study_name,
        "backbone_type": backbone_type,
    }

    save_path = save_dir / "best_hyperparameters.json"
    with open(save_path, "w") as f:
        json.dump(save_dict, f, indent=4)

    print(f"\nBest hyperparameters saved to: {save_path}")
    return save_path


def objective(trial, data_params, remote_server_uri):
    """optimization objective function"""
    # select sequence length for this trial
    sequence_length = trial.suggest_categorical("sequence_length", SEQUENCE_LENGTHS)
    data_params["sequence_length"] = sequence_length

    # Select backbone type FIRST
    backbone_type = trial.suggest_categorical("backbone_type", ["swin", "mvit"])

    # model hyperparameters to optimize
    params = {
        # basic model params
        "embed_dim": trial.suggest_int("embed_dim", 128, 512, step=64),
        "num_latent_tokens": trial.suggest_int("num_latent_tokens", 4, 16, step=2),
        "learning_rate": trial.suggest_float("learning_rate", 1e-6, 1e-3, log=True),
        "batch_size": trial.suggest_int("batch_size", 8, 128, step=8),
        "max_history_size": sequence_length,
        "backbone_type": backbone_type,
        "gradient_clip_val": trial.suggest_float("gradient_clip_val", 0.1, 1.0),
        # encoder params
        "encoder_num_heads": trial.suggest_int("encoder_num_heads", 4, 16, step=4),
        "encoder_head_dim": trial.suggest_int("encoder_head_dim", 32, 128, step=32),
        "encoder_depth": trial.suggest_int("encoder_depth", 1, 4),
        "encoder_drop_rate": trial.suggest_float("encoder_drop_rate", 0.0, 0.5),
        "encoder_mlp_ratio": trial.suggest_float("encoder_mlp_ratio", 2.0, 6.0),
        # decoder params
        "decoder_num_heads": trial.suggest_int("decoder_num_heads", 4, 16, step=4),
        "decoder_head_dim": trial.suggest_int("decoder_head_dim", 32, 128, step=32),
        "decoder_depth": trial.suggest_int("decoder_depth", 1, 4),
        "decoder_drop_rate": trial.suggest_float("decoder_drop_rate", 0.0, 0.5),
        "decoder_mlp_ratio": trial.suggest_float("decoder_mlp_ratio", 2.0, 6.0),
    }

    # Add backbone-specific parameters
    if backbone_type == "swin":
        backbone_params = {
            "backbone_depth": trial.suggest_int("backbone_depth", 2, 6),
            "backbone_num_heads": trial.suggest_int("backbone_num_heads", 1, 4),
            "backbone_mlp_ratio": trial.suggest_float("backbone_mlp_ratio", 2.0, 6.0),
            "backbone_drop_rate": trial.suggest_float("backbone_drop_rate", 0.0, 0.5),
        }
    else:  # mvit
        backbone_params = {
            "backbone_depth": trial.suggest_int("backbone_depth", 2, 6),
            "backbone_num_heads": trial.suggest_int("backbone_num_heads", 4, 16, step=4),
            "backbone_mlp_ratio": trial.suggest_float("backbone_mlp_ratio", 2.0, 6.0),
            "backbone_drop_rate": trial.suggest_float("backbone_drop_rate", 0.0, 0.5),
            "backbone_pool_size": trial.suggest_int("backbone_pool_size", 2, 8, step=2),
            "backbone_kernel_qkv": trial.suggest_int("backbone_kernel_qkv", 3, 7, step=2),
            "backbone_stride_q": trial.suggest_int("backbone_stride_q", 1, 3),
            "backbone_stride_kv": trial.suggest_int("backbone_stride_kv", 1, 3),
        }

    params.update(backbone_params)

    # Generate trial name AFTER all parameters are set
    trial_name = generate_trial_name(trial)

    # setup mlflow logging
    mlf_logger = MLFlowLogger(
        experiment_name="AQFM_optimization",
        tracking_uri=remote_server_uri,
        run_name=trial_name,
        tags={
            "trial_number": str(trial.number),
            "sequence_length": str(sequence_length),
            "embed_dim": str(params["embed_dim"]),
            "num_latent_tokens": str(params["num_latent_tokens"]),
        },
        log_model=False,
    )

    print(f"\nStarting {trial_name}")
    print(f"Sequence Length: {sequence_length} hours")
    print("Key parameters:")
    print(f"  Embedding dim: {params['embed_dim']}")
    print(f"  Latent tokens: {params['num_latent_tokens']}")
    print(f"  Learning rate: {params['learning_rate']:.2e}")
    print(f"  Batch size: {params['batch_size']}")

    # setup data
    train_dataset = AirQualityDataset(**data_params, mode="train")
    val_dataset = AirQualityDataset(**data_params, mode="val", scalers=train_dataset.get_scalers())

    train_loader = DataLoader(
        dataset=train_dataset, batch_size=params["batch_size"], shuffle=True, collate_fn=collate_aq_batches, num_workers=16
    )
    val_loader = DataLoader(dataset=val_dataset, batch_size=params["batch_size"], collate_fn=collate_aq_batches, num_workers=16)

    # get sample batch for model setup
    sample_batch, sample_targets = next(iter(train_loader))

    # create model
    model = AQFMPredictor(
        feature_names=sample_batch.metadata.feature_names,
        embed_dim=params["embed_dim"],
        num_latent_tokens=params["num_latent_tokens"],
        backbone_type=params["backbone_type"],
        max_history_size=sequence_length,
        learning_rate=params["learning_rate"],
    )

    # setup training
    callbacks = [
        ModelCheckpoint(
            dirpath=f"./checkpoints/trial_{trial.number}",
            filename="aqfm-{epoch:02d}-{averaged_val_total_loss:.2f}",
            monitor="averaged_val_total_loss",
            mode="min",
            save_top_k=2,
            save_last=True,
        ),
        SimpleModelPruning(trial, monitor="averaged_val_total_loss"),
    ]

    if torch.cuda.is_available():
        torch.cuda.init()

    trainer = pl.Trainer(
        max_epochs=100,
        devices=1,
        accelerator="cuda" if torch.cuda.is_available() else "cpu",
        callbacks=callbacks,
        gradient_clip_val=params["gradient_clip_val"],
        logger=mlf_logger,
        enable_progress_bar=True,
        enable_model_summary=True,
        log_every_n_steps=10,
    )

    try:
        # track model size
        n_params = sum(p.numel() for p in model.parameters())
        trial.set_user_attr("n_parameters", n_params)

        # train model
        trainer.fit(model, train_loader, val_loader)
        final_loss = trainer.callback_metrics["averaged_val_total_loss"].item()

        print(f"Trial {trial.number} completed successfully with validation loss: {final_loss}")
        return final_loss

    except Exception as e:
        error_type = type(e).__name__
        error_msg = str(e)
        print(f"Trial {trial.number} failed with {error_type}: {error_msg}")
        print(f"Parameters that caused failure: {params}")
        trial.set_user_attr("failure_reason", f"{error_type}: {error_msg}")
        return float("inf")


def main():
    """main optimization routine"""
    remote_server_uri = "http://0.0.0.0:8082"

    # setup directories for results and models
    results_dir = Path("optimization_results")
    model_dir = results_dir / "models"
    results_dir.mkdir(exist_ok=True)
    model_dir.mkdir(exist_ok=True)

    # setup mlflow
    mlflow.set_tracking_uri(remote_server_uri)
    experiment_name = "AQFM_optimization"
    try:
        experiment_id = mlflow.create_experiment(experiment_name)
    except Exception:
        experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id  # noqa

    mlflow.set_experiment(experiment_name)

    # data parameters
    data_params = {
        "xlsx_path": Path(__file__).parent.parent / "data/AirQuality.xlsx",
        "prediction_horizon": 1,
        "feature_groups": {
            "sensor": ["PT08.S1(CO)", "PT08.S2(NMHC)", "PT08.S3(NOx)", "PT08.S4(NO2)", "PT08.S5(O3)"],
            "ground_truth": ["CO(GT)", "NMHC(GT)", "C6H6(GT)", "NOx(GT)", "NO2(GT)"],
            "physical": ["T", "RH", "AH"],
        },
    }

    # setup optimization study
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=10,  # don't prune first 10 trials
        n_warmup_steps=5,  # don't prune before 5 epochs
        interval_steps=1,  # check for pruning every 1 epoch
    )

    study = optuna.create_study(
        study_name="aqfm_optimization",
        direction="minimize",
        pruner=pruner,
        storage="sqlite:///optuna_study.db",
        load_if_exists=True,
    )

    # run optimization
    study.optimize(
        lambda trial: objective(trial, data_params, remote_server_uri),
        n_trials=100,
        timeout=None,
        catch=(Exception,),
        show_progress_bar=True,
    )

    # save results
    best_params_path = save_best_hyperparameters(study, results_dir)

    # log final results to MLFlow
    with mlflow.start_run(experiment_id=experiment_id, run_name="best_model_summary"):
        mlflow.log_params(study.best_trial.params)
        mlflow.log_metric("best_validation_loss", study.best_trial.value)
        mlflow.log_artifact(str(best_params_path))

    # create visualizations
    try:
        fig1 = optuna.visualization.plot_optimization_history(study)
        fig2 = optuna.visualization.plot_param_importances(study)
        fig3 = optuna.visualization.plot_parallel_coordinate(study)

        fig1.write_html(str(results_dir / "optimization_history.html"))
        fig2.write_html(str(results_dir / "param_importances.html"))
        fig3.write_html(str(results_dir / "parallel_coordinate.html"))

        with mlflow.start_run(experiment_id=experiment_id, run_name="optimization_visualizations"):
            mlflow.log_artifact(str(results_dir / "optimization_history.html"))
            mlflow.log_artifact(str(results_dir / "param_importances.html"))
            mlflow.log_artifact(str(results_dir / "parallel_coordinate.html"))

    except Exception as e:
        print(f"Error creating visualization: {e}")

    print("\nOptimization completed!")
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best validation loss: {study.best_trial.value:.4f}")
    print("\nResults saved in:")
    print(f"  - Hyperparameters: {best_params_path}")
    print(f"  - Visualizations: {results_dir}")


if __name__ == "__main__":
    main()
