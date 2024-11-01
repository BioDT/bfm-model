from collections import defaultdict
from pathlib import Path
from test.test_bfm_alternate_version.src.data_set import AQBatch
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch


class TimeSeriesVisualizer:
    def __init__(self, model, dataset, sequence_length: int, prediction_horizon: int):
        self.model = model
        self.dataset = dataset
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.device = next(model.parameters()).device

    def generate_autoregressive_predictions(self, feature_name: str, window_size: int = None):
        """Generate predictions usinga rolling window approach"""
        if window_size is None:
            window_size = self.sequence_length

        sample_batch, sample_targets = self.dataset[0]
        all_data = []
        for i in range(len(self.dataset)):
            batch, targets = self.dataset[i]
            all_data.append(targets[feature_name])
        true_values = torch.cat(all_data).numpy()

        predictions = np.full(len(true_values), np.nan)
        current_window = true_values[:window_size]

        # starting the predictions after the initial window
        for i in range(window_size, len(true_values)):
            sensor_vars = {name: torch.zeros(1, window_size, device=self.device) for name in sample_batch.sensor_vars.keys()}

            ground_truth_vars = {
                name: (
                    torch.FloatTensor(current_window).unsqueeze(0).to(self.device)
                    if name == feature_name
                    else torch.zeros(1, window_size, device=self.device)
                )
                for name in sample_batch.ground_truth_vars.keys()
            }

            physical_vars = {name: torch.zeros(1, window_size, device=self.device) for name in sample_batch.physical_vars.keys()}

            batch = AQBatch(
                sensor_vars=sensor_vars,
                ground_truth_vars=ground_truth_vars,
                physical_vars=physical_vars,
                metadata=sample_batch.metadata,
            )

            with torch.no_grad():
                pred = self.model(batch)
                pred_value = pred[feature_name].cpu().numpy()[0]

            predictions[i] = pred_value
            current_window = np.roll(current_window, -1)
            current_window[-1] = pred_value

        return true_values, predictions

    def plot_variable_predictions(self, save_dir: Path):
        """Plot predictions for each ground truth variable"""
        save_dir.mkdir(exist_ok=True, parents=True)

        for feature_name in self.dataset.ground_truth_features:
            true_values, predictions = self.generate_autoregressive_predictions(feature_name)

            plt.figure(figsize=(15, 6))
            plt.plot(true_values, label="True values", alpha=0.7)
            plt.plot(predictions, label="Predictions", alpha=0.7)
            plt.title(f"Autoregressive predictions for {feature_name}")
            plt.xlabel("Time steps")
            plt.ylabel("Normalized value")
            plt.legend()
            plt.grid(True)
            plt.savefig(save_dir / f"{feature_name}_predictions.png")
            plt.close()

    def plot_training_metrics(self, training_history: Dict[str, np.ndarray], save_dir: Path):
        if not training_history:
            print("No training metrics available to plot :(((")
            return

        # Group metrics by variable and type (MSE/MAE)
        variable_metrics = defaultdict(lambda: defaultdict(dict))
        for metric_name, values in training_history.items():
            if isinstance(values, (np.ndarray, list)):
                parts = metric_name.split("_")
                if len(parts) >= 4:  # ensure we have enough parts
                    phase = parts[1]  # 'val' or 'train'
                    var_name = parts[2]  # e.g., 'CO(GT)'
                    metric_type = parts[-1]  # 'mse' or 'mae'

                    variable_metrics[var_name][metric_type][phase] = values

        # a plot for each variable and metric type
        for var_name, metric_types in variable_metrics.items():
            for metric_type, phases in metric_types.items():
                plt.figure(figsize=(12, 6))
                for phase, values in phases.items():
                    plt.plot(values, label=f"{phase} {metric_type.upper()}")

                plt.title(f"{var_name} - {metric_type.upper()} History")
                plt.xlabel("Epoch")
                plt.ylabel(metric_type.upper())
                plt.legend()
                plt.grid(True)
                plt.savefig(save_dir / f"{var_name}_{metric_type}_history.png")
                plt.close()

    def plot_feature_correlations(self, save_dir: Path):
        # making a correlation matrix between features, might be useful for further analysis
        all_features = {}
        for i in range(len(self.dataset)):
            batch, targets = self.dataset[i]
            for name, value in targets.items():
                if name not in all_features:
                    all_features[name] = []
                all_features[name].append(value.numpy())

        feature_data = {name: np.concatenate(values) for name, values in all_features.items()}
        df = pd.DataFrame(feature_data)

        plt.figure(figsize=(10, 8))
        sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
        plt.title("Feature correlations")
        plt.savefig(save_dir / "feature_correlations.png")
        plt.close()
