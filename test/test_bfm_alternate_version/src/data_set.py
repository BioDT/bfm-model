from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


@dataclass
class AQMetadata:
    """Metadata for air quality data, just like climate data"""

    time: List[datetime]
    feature_names: Dict[str, List[str]]
    sequence_length: int
    prediction_horizon: int


@dataclass
class AQBatch:
    """Batch structure for air quality data, again just like climate data"""

    sensor_vars: Dict[str, torch.Tensor]  # PT08.S* sensor readings
    ground_truth_vars: Dict[str, torch.Tensor]  # ground truth ones (*GT)
    physical_vars: Dict[str, torch.Tensor]  # any other things like T, RH, AH
    metadata: AQMetadata


class AirQualityDataset(Dataset):
    def __init__(
        self,
        xlsx_path: str,
        sequence_length: int = 24,
        prediction_horizon: int = 1,
        mode: Literal["train", "val", "test"] = "train",
        scalers: Dict[str, Dict[str, float]] = None,
        validation_split: float = 0.1,
        feature_groups: Optional[Dict[str, List[str]]] = None,
    ):
        df = pd.read_excel(xlsx_path)
        print(f"Data shape: {df.shape}")

        # combining date and time
        df["DateTime"] = df.apply(lambda row: pd.Timestamp.combine(row["Date"].date(), row["Time"]), axis=1)
        print(f"Date + time example: {df['DateTime'].iloc[0]}")

        if feature_groups is None:
            self.sensor_features = ["PT08.S1(CO)", "PT08.S2(NMHC)", "PT08.S3(NOx)", "PT08.S4(NO2)", "PT08.S5(O3)"]
            self.ground_truth_features = ["CO(GT)", "NMHC(GT)", "C6H6(GT)", "NOx(GT)", "NO2(GT)"]
            self.physical_features = ["T", "RH", "AH"]
        else:  # if there have been any provided, that was just added for convenience
            self.sensor_features = feature_groups["sensor"]
            self.ground_truth_features = feature_groups["ground_truth"]
            self.physical_features = feature_groups["physical"]

        # checking that all the features are actually there
        all_features = self.sensor_features + self.ground_truth_features + self.physical_features
        for feature in all_features:
            assert feature in df.columns, f"feat {feature} not found"
        # the dataset is fairly convenient for real life demonstrations, because we also have empty values - labeled with -200
        for col in df.columns:
            if col not in ["Date", "Time", "DateTime"]:
                df[col] = df[col].replace(-200, float("nan"))
        df = df.ffill().bfill()  # forward/backward filling

        # making sequences from the dataset - bascially our training/testing/validation examples
        sequences = []
        total_len = len(df)

        for i in range(total_len - sequence_length - prediction_horizon + 1):
            current_sequence = df.iloc[i : i + sequence_length]
            target_sequence = df.iloc[i + sequence_length : i + sequence_length + prediction_horizon]

            # ensuring that the sequences are continuous
            time_diffs = current_sequence["DateTime"].diff()[1:].dt.total_seconds() / 3600
            # if the time difference between any two consecutive rows is between 0.9 and 1.1 hours (allowing for some maring of error y'know), we consider the sequence continuous
            if time_diffs.between(0.9, 1.1).all():
                sequences.append((current_sequence, target_sequence))

        print(f"Created {len(sequences)} sequences")

        total_sequences = len(sequences)
        test_idx = int(total_sequences * 0.8)
        val_idx = int(test_idx * (1 - validation_split))

        # doing it professionally B)
        if mode == "train":
            self.sequences = sequences[:val_idx]
        elif mode == "val":
            self.sequences = sequences[val_idx:test_idx]
        else:
            self.sequences = sequences[test_idx:]

        if mode == "train":  # don't want to have data leaking, so using just the trining scalers
            self.scalers = self._compute_scalers(sequences[:test_idx])
        else:
            assert scalers is not None, f"Must provide scalers for {mode} set"
            self.scalers = scalers

        self.sequences = self._normalize_sequences(self.sequences)
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon

        print(f"Created {mode} dataset with {len(self.sequences)} sequences")
        print("Feature groups:")
        print(f"  Sensor features: {self.sensor_features}")
        print(f"  Ground truth features: {self.ground_truth_features}")
        print(f"  Physical features: {self.physical_features}")

    def _compute_scalers(self, sequences: List[Tuple[pd.DataFrame, pd.DataFrame]]) -> Dict[str, Dict[str, float]]:
        all_data = pd.concat([seq[0] for seq in sequences])

        scalers = {}
        all_features = self.sensor_features + self.ground_truth_features + self.physical_features
        for feature in all_features:  # using mean and std should do the trick
            mean = all_data[feature].mean()
            std = all_data[feature].std()
            scalers[feature] = {"mean": mean, "std": std}

        return scalers

    def _normalize_sequences(self, sequences: List[Tuple[pd.DataFrame, pd.DataFrame]]) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """Normalize sequenecs using computed/provided scalers"""
        normalized_sequences = []
        all_features = self.sensor_features + self.ground_truth_features + self.physical_features

        for input_seq, target_seq in sequences:
            input_normalized = input_seq.copy()
            target_normalized = target_seq.copy()

            for feature in all_features:
                mean = self.scalers[feature]["mean"]
                std = self.scalers[feature]["std"]
                input_normalized[feature] = (input_seq[feature] - mean) / std
                target_normalized[feature] = (target_seq[feature] - mean) / std

            normalized_sequences.append((input_normalized, target_normalized))

        return normalized_sequences

    def get_scalers(self) -> Dict[str, Dict[str, float]]:
        return self.scalers

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        input_seq, target_seq = self.sequences[idx]

        # tensors for each group of variables
        sensor_vars = {name: torch.FloatTensor(input_seq[name].values) for name in self.sensor_features}
        ground_truth_vars = {name: torch.FloatTensor(input_seq[name].values) for name in self.ground_truth_features}
        physical_vars = {name: torch.FloatTensor(input_seq[name].values) for name in self.physical_features}

        # and also target tensors
        targets = {name: torch.FloatTensor(target_seq[name].values) for name in self.ground_truth_features}

        metadata = AQMetadata(
            time=input_seq["DateTime"].tolist(),
            feature_names={
                "sensor": self.sensor_features,
                "ground_truth": self.ground_truth_features,
                "physical": self.physical_features,
            },
            sequence_length=self.sequence_length,
            prediction_horizon=self.prediction_horizon,
        )

        batch = AQBatch(sensor_vars, ground_truth_vars, physical_vars, metadata)
        return batch, targets


def collate_aq_batches(batch_list):
    """Custom collate function for creating batches"""
    inputs = [item[0] for item in batch_list]
    targets = [item[1] for item in batch_list]

    sensor_vars = {name: torch.stack([b.sensor_vars[name] for b in inputs]) for name in inputs[0].sensor_vars.keys()}
    ground_truth_vars = {
        name: torch.stack([b.ground_truth_vars[name] for b in inputs]) for name in inputs[0].ground_truth_vars.keys()
    }
    physical_vars = {name: torch.stack([b.physical_vars[name] for b in inputs]) for name in inputs[0].physical_vars.keys()}

    target_vars = {name: torch.stack([t[name] for t in targets]) for name in targets[0].keys()}

    metadata = inputs[0].metadata
    batch = AQBatch(sensor_vars, ground_truth_vars, physical_vars, metadata)

    return batch, target_vars


def main():
    data_params = {
        "xlsx_path": Path(__file__).parent.parent / "data/AirQuality.xlsx",
        "sequence_length": 24,  # using a day
        "prediction_horizon": 1,  # to predict an hour
        "feature_groups": {
            "sensor": ["PT08.S1(CO)", "PT08.S2(NMHC)", "PT08.S3(NOx)", "PT08.S4(NO2)", "PT08.S5(O3)"],
            "ground_truth": ["CO(GT)", "NMHC(GT)", "C6H6(GT)", "NOx(GT)", "NO2(GT)"],
            "physical": ["T", "RH", "AH"],
        },
    }

    # making train, validation, and test sets and their corresponding dataloaders
    train_dataset = AirQualityDataset(**data_params, mode="train")
    val_dataset = AirQualityDataset(**data_params, mode="val", scalers=train_dataset.scalers)
    test_dataset = AirQualityDataset(**data_params, mode="test", scalers=train_dataset.scalers)

    batch_size = 4
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_aq_batches)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_aq_batches)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_aq_batches)

    print("Dataset splits:")
    print(f"Train sequences: {len(train_dataset)}")
    print(f"Validation sequences: {len(val_dataset)}")
    print(f"Test sequences: {len(test_dataset)}")

    # getting just one batch from each loader and printing shapes
    for name, loader in [("Train", train_loader), ("Validation", val_loader), ("Test", test_loader)]:
        print(f"\n{name} batch shapes:")
        batch, targets = next(iter(loader))
        print("Sensor vars:")
        for var_name, tensor in batch.sensor_vars.items():
            print(f"  {var_name}: {tensor.shape}")
        print("Target vars:")
        for var_name, tensor in targets.items():
            print(f"  {var_name}: {tensor.shape}")


if __name__ == "__main__":
    main()
