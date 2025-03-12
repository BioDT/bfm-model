from datetime import timedelta
from pathlib import Path
from test.test_bfm_alternate_version.src.data_set import (
    AirQualityDataset,
    collate_aq_batches,
)

import torch
import torch.nn as nn
from einops import rearrange, repeat
from torch.utils.data import DataLoader

from bfm_model.perceiver_components.helpers_io import dropout_seq
from bfm_model.perceiver_components.pos_encoder import build_position_encoding
from bfm_model.perceiver_core.perceiver_io import PerceiverIO


class AQEncoder(nn.Module):
    """
    Encoder module for processing air quality sensor data using a Perceiver IO architecture.

    This encoder handles multiple types of variables (sensor readings, ground truth measurements,
    physical measurements) and processes them through embeddings and structured latent transformations.

    Args:
        feature_names (dict[str, list[str]]): Dictionary mapping feature categories to lists of feature names
        patch_size (int, optional): Size of patches for tokenization. Defaults to 4
        latent_tokens (int, optional): Number of latent tokens. Defaults to 8
        embed_dim (int, optional): Embedding dimension. Defaults to 512
        num_heads (int, optional): Number of attention heads. Defaults to 16
        head_dim (int, optional): Dimension of each attention head. Defaults to 64
        drop_rate (float, optional): Dropout rate. Defaults to 0.1
        depth (int, optional): Number of transformer layers. Defaults to 2
        mlp_ratio (float, optional): MLP expansion ratio. Defaults to 4.0
        max_history_size (int, optional): Maximum history window size. Defaults to 24
        perceiver_ln_eps (float, optional): Layer norm epsilon. Defaults to 1e-5

    Attributes:
        latent_tokens (int): Number of latent tokens
        drop_rate (float): Dropout rate
        embed_dim (int): Embedding dimension
        max_history_size (int): Maximum history window size
        feature_names (dict): Original feature names
        sanitized_names (dict): Cleaned feature names for internal use
        sensor_embeds (nn.ModuleDict): Embeddings for sensor variables
        ground_truth_embeds (nn.ModuleDict): Embeddings for ground truth variables
        physical_embeds (nn.ModuleDict): Embeddings for physical variables
        hour_proj (nn.Linear): Hour of day embedding
        temporal_encoding (nn.Parameter): Temporal position encoding
        lead_time_proj (nn.Linear): Lead time embedding
        latents (nn.Parameter): Learnable latent tokens
        perceiver (PerceiverIO): Main Perceiver IO module
        pos_drop (nn.Dropout): Position dropout layer
    """

    def __init__(
        self,
        feature_names: dict[str, list[str]],
        patch_size: int = 4,
        latent_tokens: int = 8,
        embed_dim: int = 512,
        num_heads: int = 16,
        head_dim: int = 64,
        drop_rate: float = 0.1,
        depth: int = 2,
        mlp_ratio: float = 4.0,
        max_history_size: int = 24,  # time steps used for prediction
        perceiver_ln_eps: float = 1e-5,
    ):
        super().__init__()
        self.latent_tokens = latent_tokens
        self.drop_rate = drop_rate
        self.embed_dim = embed_dim
        self.max_history_size = max_history_size

        self.feature_names = feature_names
        self.sanitized_names = {  # making a mapping for the feature names - to keep it clean
            "sensor": {name: f"sensor_{i}" for i, name in enumerate(feature_names["sensor"])},
            "ground_truth": {name: f"gt_{i}" for i, name in enumerate(feature_names["ground_truth"])},
            "physical": {name: f"phys_{i}" for i, name in enumerate(feature_names["physical"])},
        }

        # creating random embeddings for the sanitized names
        self.sensor_embeds = nn.ModuleDict(
            {self.sanitized_names["sensor"][name]: nn.Linear(1, embed_dim) for name in feature_names["sensor"]}
        )
        self.ground_truth_embeds = nn.ModuleDict(
            {self.sanitized_names["ground_truth"][name]: nn.Linear(1, embed_dim) for name in feature_names["ground_truth"]}
        )
        self.physical_embeds = nn.ModuleDict(
            {self.sanitized_names["physical"][name]: nn.Linear(1, embed_dim) for name in feature_names["physical"]}
        )

        # adding also some random embeddings for the hour of day and the temporal encoding
        self.hour_proj = nn.Linear(1, embed_dim)
        self.temporal_encoding = nn.Parameter(torch.randn(max_history_size, embed_dim))
        self.lead_time_proj = nn.Linear(1, embed_dim)
        self.latents = nn.Parameter(torch.randn(latent_tokens, embed_dim))  # latents for perceiver io

        self.perceiver = PerceiverIO(
            num_layers=depth,
            dim=embed_dim,
            queries_dim=embed_dim,
            logits_dimension=None,
            num_latent_tokens=latent_tokens,
            latent_dimension=embed_dim,
            cross_attention_heads=num_heads,
            latent_attention_heads=num_heads,
            cross_attention_head_dim=head_dim,
            latent_attention_head_dim=head_dim,
            sequence_dropout_prob=drop_rate,
            num_fourier_bands=32,
            max_frequency=max_history_size,
            num_input_axes=1,
            position_encoding_type="fourier",
        )

        self.pos_drop = nn.Dropout(p=drop_rate)

    def _embed_features(
        self, batch_vars: dict[str, torch.Tensor], embed_dict: nn.ModuleDict, sanitized_mapping: dict[str, str]
    ) -> torch.Tensor:
        """
        Embed a group of features using their cleaned up names.

        Args:
            batch_vars (dict[str, torch.Tensor]): Dictionary of input variables
            embed_dict (nn.ModuleDict): Dictionary of embedding layers
            sanitized_mapping (dict[str, str]): Mapping from original to clean names

        Returns:
            torch.Tensor: Embedded features
        """
        embeddings = []
        for name, tensor in batch_vars.items():
            safe_name = sanitized_mapping[name]
            x = tensor.unsqueeze(-1)
            embed = embed_dict[safe_name](x)
            embeddings.append(embed)
        return torch.stack(embeddings, dim=1)

    def _validate_dimensions(self, x: torch.Tensor, T: int) -> None:
        """
        Validate tensor dimensions.

        Args:
            x (torch.Tensor): Input tensor
            T (int): Expected sequence length

        Raises:
            AssertionError: If dimensions don't match expectations
        """
        B, N, D = x.shape
        assert N % T == 0, f"Number of features ({N}) must be divisible by sequence length ({T})"
        assert D == self.embed_dim, f"Feature dimension ({D}) must match embed_dim ({self.embed_dim})"

    def forward(self, batch, lead_time: timedelta) -> torch.Tensor:
        """
        Forward pass of the encoder.

        Args:
            batch: Batch object containing input variables and metadata
            lead_time (timedelta): Time difference between input and target

        Returns:
            torch.Tensor: Encoded representation
            Shape: [batch_size, num_latent_tokens, embed_dim]
        """
        B = next(iter(batch.sensor_vars.values())).shape[0]
        T = self.max_history_size
        device = next(iter(batch.sensor_vars.values())).device

        sensor_x = self._embed_features(batch.sensor_vars, self.sensor_embeds, self.sanitized_names["sensor"])
        ground_truth_x = self._embed_features(
            batch.ground_truth_vars, self.ground_truth_embeds, self.sanitized_names["ground_truth"]
        )
        physical_x = self._embed_features(batch.physical_vars, self.physical_embeds, self.sanitized_names["physical"])

        x = torch.cat([sensor_x, ground_truth_x, physical_x], dim=1)
        x = rearrange(x, "b n t d -> b (n t) d")  # reshaping the tensor to the input of the perceiver io

        self._validate_dimensions(x, T)
        pos_encoding = self.temporal_encoding[:T].unsqueeze(
            0
        )  # since we have timeseries, we will use simple temporal encodings as the positional encoding
        pos_encoding = repeat(pos_encoding, "1 t d -> b (n t) d", b=B, n=x.size(1) // T)
        x = x + pos_encoding

        hours = torch.tensor([t.hour for t in batch.metadata.time], device=device).float()
        hour_encoding = self.hour_proj(hours.unsqueeze(-1))
        hour_encoding = repeat(hour_encoding, "t d -> b (n t) d", b=B, n=x.size(1) // T)
        x = x + hour_encoding

        # adding the lead time encoding
        lead_hours = torch.tensor([[lead_time.total_seconds() / 3600]], device=device, dtype=torch.float)
        lead_time_encoding = self.lead_time_proj(lead_hours)
        x = x + lead_time_encoding.unsqueeze(1)

        latents = repeat(self.latents, "n d -> b n d", b=B)  # same latents for all the batch

        x = self.perceiver(x, queries=latents)
        x = self.pos_drop(x)

        return x


def main():
    # same as in the exemplar used in data_set.py (look one level up)
    data_params = {
        "xlsx_path": Path(__file__).parent.parent.parent / "data/AirQuality.xlsx",
        "sequence_length": 24,
        "prediction_horizon": 1,
        "feature_groups": {
            "sensor": ["PT08.S1(CO)", "PT08.S2(NMHC)", "PT08.S3(NOx)", "PT08.S4(NO2)", "PT08.S5(O3)"],
            "ground_truth": ["CO(GT)", "NMHC(GT)", "C6H6(GT)", "NOx(GT)", "NO2(GT)"],
            "physical": ["T", "RH", "AH"],
        },
    }

    train_dataset = AirQualityDataset(**data_params, mode="train")
    loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_aq_batches)

    batch, targets = next(iter(loader))
    print("Input shapes:")
    print("Sensor variables:")
    for name, tensor in batch.sensor_vars.items():
        print(f"{name}: {tensor.shape}")
    print("Ground truth variables:")
    for name, tensor in batch.ground_truth_vars.items():
        print(f"{name}: {tensor.shape}")
    print("Physical variables:")
    for name, tensor in batch.physical_vars.items():
        print(f"{name}: {tensor.shape}")

    encoder = AQEncoder(
        feature_names=batch.metadata.feature_names, embed_dim=512, num_heads=8, latent_tokens=8, max_history_size=24
    )

    lead_time = timedelta(hours=1)
    encoded = encoder(batch, lead_time)

    print("\nEncoder output:")
    print(f"Shape: {encoded.shape}")
    print("Expected: [batch_size=4, latent_tokens=8, embed_dim=512]")


if __name__ == "__main__":
    main()
