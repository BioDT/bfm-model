from pathlib import Path
from test.test_bfm_alternate_version.src.data_set import AirQualityDataset, AQBatch
from test.test_bfm_alternate_version.src.hyperparameter_search import AQFMPredictor
from test.test_bfm_alternate_version.src.train_baseline import TFTPredictor

import lightning as pl
import matplotlib.pyplot as plt
import pandas as pd
import torch


def load_models():
    """load all three models from their checkpoints"""
    tft_model = TFTPredictor.load_from_checkpoint(
        "checkpoints/tft/tft_best.ckpt",
        feature_names={
            "sensor": ["PT08_S1_CO_", "PT08_S2_NMHC_", "PT08_S3_NOx_", "PT08_S4_NO2_", "PT08_S5_O3_"],
            "ground_truth": ["CO_GT_", "NMHC_GT_", "C6H6_GT_", "NOx_GT_", "NO2_GT_"],
            "physical": ["T", "RH", "AH"],
        },
        map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        strict=False,
    )

    # temporary trainer for TFT model, to avoid some checkpoint importing issues
    temp_trainer = pl.Trainer(
        enable_checkpointing=False,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    tft_model.trainer = temp_trainer

    tft_model.setup()
    tft_model.eval()

    # AQFM models
    mvit_model = AQFMPredictor.load_from_checkpoint(
        "checkpoints/mvit/aqfm_mvit_best.ckpt",
        backbone_type="mvit",
        map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )
    mvit_model.eval()

    swin_model = AQFMPredictor.load_from_checkpoint(
        "checkpoints/swin/aqfm_swin_best.ckpt",
        backbone_type="swin",
        map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )
    swin_model.eval()

    return tft_model, mvit_model, swin_model


def prepare_data(n_percent: float = 0.02):
    """prepare dataset with first N % of data"""
    dataset = AirQualityDataset(
        xlsx_path=Path("test/test_bfm_alternate_version/data/AirQuality.xlsx"),
        sequence_length=48,  # using 48 hours as sequence length
        prediction_horizon=1,
    )

    total_len = len(dataset)
    cutoff_idx = int(n_percent * total_len)
    print(f"Using {cutoff_idx} sequences out of {total_len} total sequences ({(cutoff_idx/total_len)*100:.1f}%)")

    return dataset, cutoff_idx


def plot_predictions_for_variable(variable_name, actual_data, predictions_dict, future_predictions_dict):
    """make a plot for a single variable showing actual data and predictions from all models"""
    print(f"\nCreating plot for {variable_name}...")
    print(f"Data lengths: Actual={len(actual_data)}, Predictions={[f'{k}={len(v)}' for k,v in predictions_dict.items()]}")

    plt.figure(figsize=(15, 8))

    # actual data
    plt.plot(actual_data, label="Actual", color="black", linewidth=2)

    # model predictions on known data
    colors = {"TFT": "blue", "AQFM-MViT": "red", "AQFM-Swin": "green"}
    for model_name, preds in predictions_dict.items():
        plt.plot(preds, label=f"{model_name} Predictions", color=colors[model_name], linestyle="--")

    # future predictions
    for model_name, future_preds in future_predictions_dict.items():
        plt.plot(
            range(len(actual_data), len(actual_data) + len(future_preds)),
            future_preds,
            label=f"{model_name} Future",
            color=colors[model_name],
            linestyle=":",
        )

    plt.title(f"Predictions for {variable_name}")
    plt.xlabel("Time (hours)")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)

    save_path = f'predictions_{variable_name.replace("(", "_").replace(")", "_")}.png'
    print(f"Saving plot to {save_path}")
    plt.savefig(save_path)
    plt.close()


def get_actual_data(dataset, cutoff_idx, variable_name):
    """getting actual data for a specific variable up to some cutoff"""
    actual_values = []
    scalers = dataset.get_scalers()
    scaler_mean = scalers[variable_name]["mean"]
    scaler_std = scalers[variable_name]["std"]

    for i in range(cutoff_idx):
        batch, targets = dataset[i]
        value = targets[variable_name].item() * scaler_std + scaler_mean
        actual_values.append(value)
    return actual_values


def get_model_predictions(model, dataset, cutoff_idx, variable_name, model_type="tft"):
    predictions = []
    scalers = dataset.get_scalers()
    scaler_mean = scalers[variable_name]["mean"]
    scaler_std = scalers[variable_name]["std"]

    # mapping for TFT
    ground_truth_name_map = {
        "CO_GT_": "CO(GT)",
        "NMHC_GT_": "NMHC(GT)",
        "C6H6_GT_": "C6H6(GT)",
        "NOx_GT_": "NOx(GT)",
        "NO2_GT_": "NO2(GT)",
    }
    reverse_name_map = {v: k for k, v in ground_truth_name_map.items()}

    with torch.no_grad():
        for i in range(cutoff_idx):
            batch, _ = dataset[i]
            if model_type == "tft":
                try:
                    # tft want spcial batches of course
                    ground_truth_features = torch.stack(
                        [batch.ground_truth_vars[ground_truth_name_map[name]] for name in model.feature_names["ground_truth"]]
                    )

                    tft_batch = {
                        "encoder_lengths": torch.tensor([48]),
                        "decoder_lengths": torch.tensor([1]),
                        "encoder_cat": torch.empty((1, 48, 0)),
                        "decoder_cat": torch.empty((1, 1, 0)),
                        "encoder_cont": ground_truth_features.transpose(0, 1).unsqueeze(0),
                        "decoder_cont": ground_truth_features[:, -1:].transpose(0, 1).unsqueeze(0),
                        "decoder_target": torch.zeros((1, 1, len(model.feature_names["ground_truth"]))),
                        "target_scale": torch.ones((1, len(model.feature_names["ground_truth"]))),
                    }

                    output = model(tft_batch)
                    sanitized_name = reverse_name_map[variable_name]
                    var_idx = model.feature_names["ground_truth"].index(sanitized_name)

                    if isinstance(output.prediction, list):
                        # tft returns a list of predictions, so we take just take the first one
                        pred = output.prediction[0][0, 0, 0].item()
                    else:
                        pred = output.prediction[0, 0, 0].item()

                except Exception as e:
                    print(f"Error processing TFT batch {i}: {str(e)}")
                    print(
                        f"Output shape: {output.prediction.shape if not isinstance(output.prediction, list) else [x.shape for x in output.prediction]}"
                    )
                    print(f"Variable index: {var_idx}")
                    print(f"Variable name: {variable_name} -> {sanitized_name}")
                    raise e
            else:  # the AQFM variants
                print(f"\nProcessing batch {i}:")

                try:
                    # new batch with the same structure but reshaped tensors, because
                    new_batch = AQBatch(
                        sensor_vars={name: value.unsqueeze(0) for name, value in batch.sensor_vars.items()},
                        ground_truth_vars={name: value.unsqueeze(0) for name, value in batch.ground_truth_vars.items()},
                        physical_vars={name: value.unsqueeze(0) for name, value in batch.physical_vars.items()},
                        metadata=batch.metadata,
                    )

                    # imma let the AQFMPredictor wrapper handle the timedelta and model call
                    output = model(new_batch)
                    pred = output[variable_name].squeeze().item()

                except Exception as e:
                    print(f"Error in model prediction: {str(e)}")
                    print(f"Original batch sensor_vars shape: {[v.shape for v in batch.sensor_vars.values()]}")
                    print(f"Original batch ground_truth_vars shape: {[v.shape for v in batch.ground_truth_vars.values()]}")
                    print(f"Original batch physical_vars shape: {[v.shape for v in batch.physical_vars.values()]}")
                    raise e

            # denormalizing the prediction for these ones as well
            pred = pred * scaler_std + scaler_mean
            predictions.append(pred)

    return predictions


def get_future_predictions(model, dataset, cutoff_idx, variable_name, model_type="tft", n_steps=48):
    """Get iterative future predictions"""
    future_preds = []
    scalers = dataset.get_scalers()
    scaler_mean = scalers[variable_name]["mean"]
    scaler_std = scalers[variable_name]["std"]

    # last known sequence
    last_batch, _ = dataset[cutoff_idx - 1]

    ground_truth_name_map = {
        "CO_GT_": "CO(GT)",
        "NMHC_GT_": "NMHC(GT)",
        "C6H6_GT_": "C6H6(GT)",
        "NOx_GT_": "NOx(GT)",
        "NO2_GT_": "NO2(GT)",
    }
    # reverse_name_map = {v: k for k, v in ground_truth_name_map.items()}

    with torch.no_grad():
        for _ in range(n_steps):
            # prediction for next step
            if model_type == "tft":
                # tft wants special batches again
                ground_truth_features = torch.stack(
                    [last_batch.ground_truth_vars[ground_truth_name_map[name]] for name in model.feature_names["ground_truth"]]
                )

                tft_batch = {
                    "encoder_lengths": torch.tensor([48]),
                    "decoder_lengths": torch.tensor([1]),
                    "encoder_cat": torch.empty((1, 48, 0)),
                    "decoder_cat": torch.empty((1, 1, 0)),
                    "encoder_cont": ground_truth_features.transpose(0, 1).unsqueeze(0),
                    "decoder_cont": ground_truth_features[:, -1:].transpose(0, 1).unsqueeze(0),
                    "decoder_target": torch.zeros((1, 1, len(model.feature_names["ground_truth"]))),
                    "target_scale": torch.ones((1, len(model.feature_names["ground_truth"]))),
                }

                output = model(tft_batch)
                # sanitized_name = reverse_name_map[variable_name]
                # var_idx = model.feature_names["ground_truth"].index(sanitized_name)

                if isinstance(output.prediction, list):
                    pred = output.prediction[0][0, 0, 0].item()
                else:
                    pred = output.prediction[0, 0, 0].item()
            else:  # we have an AQFM variant
                new_batch = AQBatch(
                    sensor_vars={name: value.unsqueeze(0) for name, value in last_batch.sensor_vars.items()},
                    ground_truth_vars={name: value.unsqueeze(0) for name, value in last_batch.ground_truth_vars.items()},
                    physical_vars={name: value.unsqueeze(0) for name, value in last_batch.physical_vars.items()},
                    metadata=last_batch.metadata,
                )
                pred = model(new_batch)[variable_name].squeeze().item()

            # denormalizing the prediction
            pred = pred * scaler_std + scaler_mean
            future_preds.append(pred)

            # updating the batch with the new prediction
            if model_type == "tft":
                # update the ground truth variables
                for gt_var in last_batch.ground_truth_vars:
                    if gt_var == variable_name:
                        # shift the sequence and add new prediction
                        last_batch.ground_truth_vars[gt_var] = torch.cat(
                            [last_batch.ground_truth_vars[gt_var][1:], torch.tensor([pred]).float()]
                        )
            else:  # AQFM models
                # update the ground truth variables
                last_batch.ground_truth_vars[variable_name] = torch.cat(
                    [last_batch.ground_truth_vars[variable_name][1:], torch.tensor([pred]).float()]
                )

            # update other variables by shifting
            for sensor_var in last_batch.sensor_vars:
                last_batch.sensor_vars[sensor_var] = torch.cat(
                    [last_batch.sensor_vars[sensor_var][1:], last_batch.sensor_vars[sensor_var][-1:].clone()]
                )

            for phys_var in last_batch.physical_vars:
                last_batch.physical_vars[phys_var] = torch.cat(
                    [last_batch.physical_vars[phys_var][1:], last_batch.physical_vars[phys_var][-1:].clone()]
                )

    return future_preds


def main():
    print("Loading models...")
    tft_model, mvit_model, swin_model = load_models()

    print("Preparing data...")
    dataset, cutoff_idx = prepare_data()

    # we will get predictions for all ground truth variables
    ground_truth_vars = ["CO(GT)", "NMHC(GT)", "C6H6(GT)", "NOx(GT)", "NO2(GT)"]

    for variable in ground_truth_vars:
        print(f"\nProcessing {variable}...")

        print("Getting actual data...")
        actual_data = get_actual_data(dataset, cutoff_idx, variable)
        print(f"Got {len(actual_data)} actual data points")

        print("Getting model predictions...")
        predictions = {
            "TFT": get_model_predictions(tft_model, dataset, cutoff_idx, variable, "tft"),
            "AQFM-MViT": get_model_predictions(mvit_model, dataset, cutoff_idx, variable, "aqfm"),
            "AQFM-Swin": get_model_predictions(swin_model, dataset, cutoff_idx, variable, "aqfm"),
        }

        print("Getting future predictions...")
        future_predictions = {
            "TFT": get_future_predictions(tft_model, dataset, cutoff_idx, variable, "tft"),
            "AQFM-MViT": get_future_predictions(mvit_model, dataset, cutoff_idx, variable, "aqfm"),
            "AQFM-Swin": get_future_predictions(swin_model, dataset, cutoff_idx, variable, "aqfm"),
        }

        plot_predictions_for_variable(variable, actual_data, predictions, future_predictions)


if __name__ == "__main__":
    main()
