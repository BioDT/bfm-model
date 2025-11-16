from pathlib import Path

import hydra
import torch
from torch.utils.data import Subset, DataLoader
from hydra.core.hydra_config import HydraConfig
from lightning.pytorch import seed_everything
from omegaconf import DictConfig, OmegaConf

from bfm_model.bfm.attention_hooks import (
    AttentionWeightCapture,
    aggregate_attention_by_modality,
    aggregate_spatial_maps_all_modalities,
    build_modality_mapping,
    compute_cross_modality_correlation,
)
from bfm_model.bfm.dataloader_helpers import get_val_dataloader
from bfm_model.bfm.dataloader_monthly import (
    LargeClimateDataset,
    _convert,
)
from bfm_model.bfm.model_helpers import get_mlflow_logger, get_trainer, setup_bfm_model


@hydra.main(version_base=None, config_path="configs", config_name="train_config")
def main(cfg: DictConfig):
    """Process windows one at a time to avoid memory issues."""

    print(OmegaConf.to_yaml(cfg))
    torch.set_float32_matmul_precision(cfg.training.precision_in)
    seed_everything(42, workers=True)

    output_dir = HydraConfig.get().runtime.output_dir
    print(f"output dir: {output_dir}")

    # setup
    test_dataset = LargeClimateDataset(
        data_dir=cfg.data.test_data_path,
        scaling_settings=cfg.data.scaling,
        num_species=cfg.data.species_number,
        atmos_levels=cfg.data.atmos_levels,
        model_patch_size=cfg.model.patch_size,
    )
    test_dataloader = get_val_dataloader(cfg, batch_size_override=cfg.evaluation.batch_size)

    bfm_model = setup_bfm_model(cfg, mode="test")
    checkpoint_path = cfg.evaluation.checkpoint_path

    # initialize attention capture
    attention_capture = AttentionWeightCapture()
    if hasattr(bfm_model, 'encoder') and hasattr(bfm_model.encoder, 'perceiver_io'):
        attention_capture.enable_capture(bfm_model.encoder.perceiver_io)
    else:
        print("warning: could not find encoder.perceiver_io module!")

    bfm_model.eval()

    # setup trainer for single-batch prediction
    experiment_name = "BFM-attention-iterative"
    mlflow_logger = get_mlflow_logger(output_dir, experiment_name=experiment_name)
    loggers = [l for l in [mlflow_logger] if l]

    trainer = get_trainer(cfg, mlflow_logger=loggers, callbacks=[])

    # output directory
    SAVE_DIR = Path("standardize_with_annealed_mask")
    SAVE_DIR.mkdir(exist_ok=True, parents=True)

    print("\n processing windows one at a time ...")
    total_windows = 0
    attn_saved = 0

    # process each batch individually
    for batch_idx in range(len(test_dataloader)):
        print(f"\n[{batch_idx + 1}/{len(test_dataloader)}] Processing window...")

        # create a single-batch dataloader
        single_batch_dataset = Subset(test_dataloader.dataset, [batch_idx])
        single_batch_loader = DataLoader(
            single_batch_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0  # no workers to avoid multiprocessing issues
        )

        # run prediction on this single batch
        predictions = trainer.predict(
            model=bfm_model,
            ckpt_path=checkpoint_path,
            dataloaders=single_batch_loader
        )

        # process the result
        for batch in predictions:
            for rec in batch:
                idx = batch_idx  # use batch_idx directly for correct indexing
                total_windows += 1

                # save prediction
                pred_scaled = test_dataset.scale_batch(rec["pred"], direction="original")
                gt_scaled = test_dataset.scale_batch(rec["gt"], direction="original")

                window_data = {
                    "pred": _convert(pred_scaled),
                    "gt": _convert(gt_scaled, move_cpu=True),
                }

                torch.save(window_data, SAVE_DIR / f"window_{idx:05d}.pt")
                print(f"Saved prediction: window_{idx:05d}.pt")

                # get attention
                raw_batch = rec.get("raw_batch", rec["gt"])
                modality_mapping = build_modality_mapping(raw_batch, patch_size=cfg.model.patch_size)
                attention_weights = attention_capture.get_attention_weights()

                if attention_weights:
                    # 1. aggregate attention by modality (scores per layer)
                    modality_contributions = {}
                    for layer_name, attn_tensor in attention_weights.items():
                        modality_contributions[layer_name] = aggregate_attention_by_modality(
                            attn_tensor, modality_mapping
                        )

                    # 2. compute spatial attention maps for all modalities (use first layer)
                    first_layer_attn = list(attention_weights.values())[0]

                    # get grid dimensions from metadata
                    if hasattr(raw_batch, 'batch_metadata'):
                        H = len(raw_batch.batch_metadata.latitudes[0])
                        W = len(raw_batch.batch_metadata.longitudes[0])
                    else:
                        # fallback to default grid size
                        H, W = 160, 280

                    spatial_maps = aggregate_spatial_maps_all_modalities(
                        first_layer_attn,
                        modality_mapping,
                        H=H,
                        W=W,
                        patch_size=cfg.model.patch_size
                    )

                    # 3. compute cross-modality correlation matrix (use first layer)
                    correlation_matrix = compute_cross_modality_correlation(
                        first_layer_attn,
                        modality_mapping
                    )

                    # save hybrid data: aggregated scores + spatial maps + correlations
                    # file size: ~67KB
                    attn_payload = {
                        "modality_contributions": modality_contributions,
                        "spatial_attention_maps": spatial_maps,
                        "modality_correlation": correlation_matrix,
                        "modality_mapping": modality_mapping,
                        "metadata": {
                            "timestamp": raw_batch.batch_metadata.timestamp[0] if hasattr(raw_batch, 'batch_metadata') else None,
                            "latitudes": raw_batch.batch_metadata.latitudes[0] if hasattr(raw_batch, 'batch_metadata') else None,
                            "longitudes": raw_batch.batch_metadata.longitudes[0] if hasattr(raw_batch, 'batch_metadata') else None,
                        },
                    }

                    torch.save(attn_payload, SAVE_DIR / f"attention_window_{idx:05d}.pt")
                    print(f"Saved attention: attention_window_{idx:05d}.pt (hybrid format: ~67KB)")
                    attn_saved += 1
                else:
                    print(f"No attention captured")

                # clear memory
                attention_capture.reset()
                del pred_scaled, gt_scaled, window_data
                if attention_weights:
                    del attention_weights, attn_payload, spatial_maps, correlation_matrix
                torch.cuda.empty_cache()

    print(f"\n summary ...")
    print(f"windows processed: {total_windows}")
    print(f"attention files saved: {attn_saved}")

    attention_capture.disable_capture()


if __name__ == "__main__":
    main()