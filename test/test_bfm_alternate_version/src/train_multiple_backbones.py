import json
from pathlib import Path
from test.test_bfm_alternate_version.src.data_set import (
    AirQualityDataset,
    collate_aq_batches,
)
from test.test_bfm_alternate_version.src.hyperparameter_search import AQFMPredictor

import lightning.pytorch as pl
import mlflow
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import MLFlowLogger
from torch.utils.data import DataLoader


def load_best_hyperparameters(backbone_type: str) -> dict:
    """
    Load best hyperparameters for a given backbone type from a JSON file.

    Args:
        backbone_type (str): Type of backbone model ('mvit' or 'swin')

    Returns:
        dict: Dictionary containing best hyperparameters for the specified backbone
    """
    file_path = Path(__file__).parent.parent / f"backbone_hyperparameters/best_hyperparameters_{backbone_type}.json"
    with open(file_path, "r") as f:
        return json.load(f)["best_parameters"]


def train_and_evaluate_model(backbone_type: str, params: dict, data_params: dict, mlf_logger: MLFlowLogger):
    """
    Train and evaluate a model with specified backbone type and parameters.

    Args:
        backbone_type (str): Type of backbone model ('mvit' or 'swin')
        params (dict): Model hyperparameters
        data_params (dict): Dataset parameters
        mlf_logger (MLFlowLogger): MLflow logger instance

    Returns:
        tuple: (test_results, best_model_path)
            - test_results (dict): Dictionary containing test metrics
            - best_model_path (str): Path to the best saved model checkpoint
    """
    print(f"\nTraining {backbone_type.upper()} backbone model...")

    train_dataset = AirQualityDataset(**data_params, mode="train")
    val_dataset = AirQualityDataset(**data_params, mode="val", scalers=train_dataset.get_scalers())
    test_dataset = AirQualityDataset(**data_params, mode="test", scalers=train_dataset.get_scalers())

    dataloader_params = {"batch_size": params["batch_size"], "collate_fn": collate_aq_batches, "num_workers": 16}

    # setup
    train_loader = DataLoader(dataset=train_dataset, shuffle=True, **dataloader_params)
    val_loader = DataLoader(dataset=val_dataset, **dataloader_params)
    test_loader = DataLoader(dataset=test_dataset, **dataloader_params)

    # get sample batch
    sample_batch, _ = next(iter(train_loader))

    # model-specific parameters (excluding training params)
    model_params = {  # noqa
        "feature_names": sample_batch.metadata.feature_names,
        "embed_dim": params["embed_dim"],
        "num_latent_tokens": params["num_latent_tokens"],
        "backbone_type": backbone_type,
        "max_history_size": params["sequence_length"],
        "learning_rate": params["learning_rate"],
        # encoder params
        "encoder_num_heads": params["encoder_num_heads"],
        "encoder_head_dim": params["encoder_head_dim"],
        "encoder_depth": params["encoder_depth"],
        "encoder_drop_rate": params["encoder_drop_rate"],
        "encoder_mlp_ratio": params["encoder_mlp_ratio"],
        # decoder params
        "decoder_num_heads": params["decoder_num_heads"],
        "decoder_head_dim": params["decoder_head_dim"],
        "decoder_depth": params["decoder_depth"],
        "decoder_drop_rate": params["decoder_drop_rate"],
        "decoder_mlp_ratio": params["decoder_mlp_ratio"],
        # backbone params
        "backbone_depth": params["backbone_depth"],
        "backbone_num_heads": params["backbone_num_heads"],
        "backbone_mlp_ratio": params["backbone_mlp_ratio"],
        "backbone_drop_rate": params["backbone_drop_rate"],
    }

    model = AQFMPredictor(
        feature_names=sample_batch.metadata.feature_names,
        embed_dim=params["embed_dim"],
        num_latent_tokens=params["num_latent_tokens"],
        backbone_type=params["backbone_type"],
        max_history_size=params["sequence_length"],
        learning_rate=params["learning_rate"],
        backbone_num_heads=params["backbone_num_heads"],
    )

    print("\nAQFMPredictor Parameters:")
    print("========================")
    for param_name, param_value in model.hparams.items():
        print(f"{param_name}: {param_value}")

    checkpoint_callback = ModelCheckpoint(
        dirpath=f"./checkpoints/{backbone_type}",
        filename=f"aqfm_{backbone_type}_best",
        monitor="averaged_val_total_loss",
        mode="min",
        save_top_k=1,  # only the best of the best
        save_last=True,
    )

    early_stopping_callback = pl.callbacks.EarlyStopping(
        monitor="averaged_val_total_loss",
        min_delta=0.001,
        patience=10,
        verbose=True,
        mode="min",
    )

    trainer = pl.Trainer(
        max_epochs=100,
        devices=1,
        accelerator="cuda" if torch.cuda.is_available() else "cpu",
        callbacks=[checkpoint_callback, early_stopping_callback],
        gradient_clip_val=params["gradient_clip_val"],
        logger=mlf_logger,
        enable_progress_bar=True,
        enable_model_summary=True,
        log_every_n_steps=10,
        min_epochs=45,
    )

    trainer.fit(model, train_loader, val_loader)

    print(f"\nTesting {backbone_type.upper()} model...")
    test_results = trainer.test(model, test_loader)[0]

    # log tesst metrics
    cleaned_metrics = {}
    for k, v in test_results.items():
        clean_key = k.replace("(", "_").replace(")", "_")
        cleaned_metrics[clean_key] = v

    with mlflow.start_run(run_id=mlf_logger.run_id):
        mlflow.log_metrics(cleaned_metrics)
        best_model_path = checkpoint_callback.best_model_path
        if best_model_path and Path(best_model_path).exists():
            mlflow.log_artifact(best_model_path)

    return test_results, best_model_path


def main():
    """
    Main function to train and evaluate multiple backbone models.

    This function:
    1. Sets up MLflow tracking
    2. Loads data parameters and best hyperparameters for each backbone type
    3. Trains and evaluates models with different backbones (MVit and Swin)
    4. Logs results and saves best model checkpoints
    """
    remote_server_uri = "http://0.0.0.0:8082"
    experiment_name = "AQFM_best_models_comparison"

    # setup mlflow
    mlflow.set_tracking_uri(remote_server_uri)
    try:
        experiment_id = mlflow.create_experiment(experiment_name)
    except Exception:
        experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id  # noqa

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

    results = {}
    best_model_paths = {}

    # train and evaluate both models
    for backbone_type in ["mvit", "swin"]:
        params = load_best_hyperparameters(backbone_type)
        data_params["sequence_length"] = params["sequence_length"]

        # separate logger for each model
        mlf_logger = MLFlowLogger(
            experiment_name=experiment_name,
            tracking_uri=remote_server_uri,
            run_name=f"best_{backbone_type}_model",
            tags={"backbone_type": backbone_type},
        )

        # train and evaluate
        test_results, best_model_path = train_and_evaluate_model(
            backbone_type=backbone_type, params=params, data_params=data_params, mlf_logger=mlf_logger
        )

        results[backbone_type] = test_results
        best_model_paths[backbone_type] = best_model_path

    print("\nFinal Results:")
    print("==============")
    for backbone_type in ["mvit", "swin"]:
        print(f"\n{backbone_type.upper()} Model:")
        print(f"Best model saved at: {best_model_paths[backbone_type]}")
        print("Test metrics:")
        for metric, value in results[backbone_type].items():
            print(f"  {metric}: {value:.4f}")


if __name__ == "__main__":
    main()
