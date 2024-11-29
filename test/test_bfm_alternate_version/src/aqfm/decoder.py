from datetime import timedelta
from pathlib import Path
from test.test_bfm_alternate_version.src.aqfm.encoder import AQEncoder
from test.test_bfm_alternate_version.src.data_set import (
    AirQualityDataset,
    collate_aq_batches,
)

import torch
import torch.nn as nn
from einops import rearrange, repeat
from torch.utils.data import DataLoader

from src.perceiver_components.helpers_io import dropout_seq
from src.perceiver_core.perceiver_io import PerceiverIO


class AQDecoder(nn.Module):
    """
    Air Quality Foundation Model Decoder.

    This decoder takes encoded representations from the AQFM encoder and transforms them back into
    predictions for air quality ground truth variables using a Perceiver IO architecture.

    Args:
        feature_names (dict[str, list[str]]): Dictionary mapping feature categories to lists of feature names
        embed_dim (int, optional): Embedding dimension. Defaults to 512.
        num_heads (int, optional): Number of attention heads. Defaults to 16.
        head_dim (int, optional): Dimension of each attention head. Defaults to 64.
        depth (int, optional): Number of transformer layers. Defaults to 2.
        mlp_ratio (float, optional): Ratio for MLP hidden dimension. Defaults to 4.0.
        drop_rate (float, optional): Dropout rate. Defaults to 0.1.
        perceiver_ln_eps (float, optional): Layer norm epsilon. Defaults to 1e-5.

    Attributes:
        feature_names (dict): Original feature names mapping
        embed_dim (int): Embedding dimension
        target_names (dict): Mapping from original ground truth names to safe internal names
        perceiver (PerceiverIO): Main Perceiver IO model
        prediction_heads (nn.ModuleDict): Prediction heads for each target variable
        query_tokens (nn.Parameter): Learnable query tokens for each target
        lead_time_embed (nn.Linear): Lead time embedding layer
        pos_drop (nn.Dropout): Position dropout layer
    """

    def __init__(
        self,
        feature_names: dict[str, list[str]],
        embed_dim: int = 512,
        num_heads: int = 16,
        head_dim: int = 64,
        depth: int = 2,
        mlp_ratio: float = 4.0,
        drop_rate: float = 0.1,
        perceiver_ln_eps: float = 1e-5,
    ):
        super().__init__()
        self.feature_names = feature_names
        self.embed_dim = embed_dim

        # making a mapping for the feature names - to keep it clean, just as in the encoder, but only for the targets now
        self.target_names = {name: f"target_{i}" for i, name in enumerate(feature_names["ground_truth"])}

        self.perceiver = PerceiverIO(
            num_layers=depth,
            dim=embed_dim,
            queries_dim=embed_dim,
            logits_dimension=None,
            num_latent_tokens=len(feature_names["ground_truth"]),  # One token per target variable
            latent_dimension=embed_dim,
            cross_attention_heads=num_heads,
            latent_attention_heads=num_heads,
            cross_attention_head_dim=head_dim,
            latent_attention_head_dim=head_dim,
            sequence_dropout_prob=drop_rate,
            num_fourier_bands=32,
            max_frequency=24,  # For 24 hours of history
            num_input_axes=1,
            position_encoding_type="fourier",
        )

        # making heads for each ground truth variable (can be improved by leveraging perceiver IO queries but for now this is fine)
        self.prediction_heads = nn.ModuleDict(
            {
                self.target_names[name]: nn.Sequential(
                    nn.Linear(embed_dim, embed_dim // 2),
                    nn.ReLU(),
                    nn.Dropout(drop_rate),
                    nn.Linear(embed_dim // 2, embed_dim // 4),
                    nn.ReLU(),
                    nn.Dropout(drop_rate),
                    nn.Linear(embed_dim // 4, 1),  # single value for next time step
                )
                for name in feature_names["ground_truth"]
            }
        )

        # query tokens for each target variable
        self.query_tokens = nn.Parameter(torch.randn(len(feature_names["ground_truth"]), embed_dim))

        self.lead_time_embed = nn.Linear(1, embed_dim)
        self.pos_drop = nn.Dropout(p=drop_rate)

    def forward(self, x: torch.Tensor, batch, lead_time: timedelta) -> dict[str, torch.Tensor]:
        """
        Forward pass of the AQFM decoder.

        Args:
            x (torch.Tensor): Encoded representation from encoder [B, L, D]
            batch: Original batch data containing input variables and metadata
            lead_time (timedelta): Time difference between input and target

        Returns:
            dict[str, torch.Tensor]: Dictionary mapping ground truth variable names to their predictions.
                                   Each prediction has shape [B, 1] for one-step-ahead forecasting.
        """
        B = x.shape[0]
        device = x.device

        queries = repeat(self.query_tokens, "n d -> b n d", b=B)

        # lead time info to queries
        lead_hours = torch.tensor([[lead_time.total_seconds() / 3600]], device=device, dtype=torch.float)
        lead_time_encoding = self.lead_time_embed(lead_hours)
        queries = queries + lead_time_encoding.unsqueeze(1)

        decoded = self.perceiver(x, queries=queries)
        decoded = self.pos_drop(decoded)

        # make predictions predictions for each target variable
        predictions = {}
        for idx, (orig_name, safe_name) in enumerate(self.target_names.items()):
            # Use the corresponding token from decoded output
            token_embedding = decoded[:, idx]  # shape: [B, D]
            pred = self.prediction_heads[safe_name](token_embedding)  # shape: [B, 1]
            predictions[orig_name] = pred
        return predictions


def main():
    # same as in encoder.py or data_set.py
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
    batch, targets = next(iter(loader))  # and get a batch out of it with its targets

    print("Target shapes:")
    for name, tensor in targets.items():
        print(f"  {name}: {tensor.shape}")

    encoder = AQEncoder(
        feature_names=batch.metadata.feature_names, embed_dim=512, num_heads=8, latent_tokens=8, max_history_size=24
    )

    decoder = AQDecoder(feature_names=batch.metadata.feature_names, embed_dim=512, num_heads=8, head_dim=64, depth=2)

    # test!
    lead_time = timedelta(hours=1)
    encoded = encoder(batch, lead_time)
    print(f"Encoder output shape: {encoded.shape}")
    predictions = decoder(encoded, batch, lead_time)

    print("Prediction shapes:")
    for name, pred in predictions.items():
        print(f"  {name}: {pred.shape}")

    print("Verifying shapes match:")
    print(f"All predictions should be [batch_size={batch.batch_size}, prediction_horizon={data_params['prediction_horizon']}]")
    for name in predictions.keys():
        assert (
            predictions[name].shape == targets[name].shape
        ), f"Shape mismatch for {name}: prediction {predictions[name].shape} != target {targets[name].shape}"
    print("All shapes verified correctly!")


if __name__ == "__main__":
    main()
