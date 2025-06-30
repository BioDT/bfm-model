"""
Copyright (C) 2025 TNO, The Netherlands. All rights reserved.
"""

from torch.utils.data import DataLoader, Dataset

from bfm_model.bfm.dataloader_monthly import LargeClimateDataset, custom_collate


def get_train_dataloader(cfg):
    dataset = LargeClimateDataset(
        data_dir=cfg.data.data_path,
        scaling_settings=cfg.data.scaling,
        num_species=cfg.data.species_number,
        atmos_levels=cfg.data.atmos_levels,
        model_patch_size=cfg.model.patch_size,
    )
    train_dataloader = DataLoader(
        dataset,
        shuffle=True,  # keep shuffle=True here
        batch_size=cfg.training.batch_size,
        num_workers=cfg.training.workers,
        collate_fn=custom_collate,
        drop_last=True,
        pin_memory=True,
    )
    print(f"Dataloader train: {len(train_dataloader)}")
    return train_dataloader


def get_val_dataloader(cfg, batch_size_override: int | None = None):
    batch_size = cfg.training.batch_size
    if batch_size_override:
        batch_size = batch_size_override
    test_dataset = LargeClimateDataset(
        data_dir=cfg.data.test_data_path,
        scaling_settings=cfg.data.scaling,
        num_species=cfg.data.species_number,
        atmos_levels=cfg.data.atmos_levels,
        model_patch_size=cfg.model.patch_size,
    )

    val_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=cfg.training.workers,
        collate_fn=custom_collate,
        drop_last=True,
        shuffle=False,
    )
    print(f"Validation train: {len(val_dataloader)}")
    return val_dataloader


class SequentialWindowDataset(Dataset):
    """
    Wrap an underlying single-sample dataset so __getitem__(i) returns a list
    [sample_i, sample_{i+1}, … sample_{i+steps}] where steps is user-defined.
    """

    def __init__(self, base_ds: Dataset, steps: int):
        assert steps >= 1, "steps must be ≥ 1"
        self.base = base_ds  # yields one Batch per index
        self.steps = steps

    def __len__(self):
        # last valid start idx is len(base) - steps - 1
        return len(self.base) - self.steps

    def __getitem__(self, idx):
        return [self.base[idx + k] for k in range(self.steps + 1)]  # list length = steps+1
