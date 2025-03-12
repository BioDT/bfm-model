"""
AQFM (Air Quality Foundation Model) Main Module.

This module contains the main AQFM architecture, combining encoder, backbone and decoder components
to process air quality sensor data and related environmental variables.

The model uses either a Swin or MViT backbone architecture to process encoded representations
before decoding back to the original variable space.

Key Components:
    - Variable preprocessing and encoding
    - Encoder for initial representation learning
    - Backbone (Swin or MViT) for temporal processing
    - Decoder for reconstructing variables
    - Multi-category variable handling (sensor readings, ground truth, physical measurements)

Example usage:
    model = AQFM(
        feature_names={
            'sensor': ['PT08.S1(CO)', 'PT08.S2(NMHC)'],
            'ground_truth': ['CO(GT)', 'NMHC(GT)'],
            'physical': ['T', 'RH', 'AH']
        },
        embed_dim=512,
        backbone_type='mvit'
    )
    predictions = model(batch, lead_time)
"""

from datetime import timedelta
from test.test_bfm_alternate_version.src.aqfm.decoder import AQDecoder
from test.test_bfm_alternate_version.src.aqfm.encoder import AQEncoder
from typing import Literal

import torch
import torch.nn as nn

from bfm_model.mvit.mvit_model import MViT
from bfm_model.swin_transformer.core.swim_core_v2 import Swin3DTransformer


class AQFM(nn.Module):
    """
    Air Quality Foundation Model.

    This model combines encoder, backbone and decoder components to process air quality sensor data
    and related environmental variables.

    Can be technically be called as a sibbling of the BFM, as it is a simplified version of it, and for a simpler dataset/task.

    Args:
        feature_names (dict[str, list[str]]): Dictionary mapping feature categories to lists of feature names
        embed_dim (int, optional): Embedding dimension. Defaults to 512.
        num_latent_tokens (int, optional): Number of latent tokens. Defaults to 8.
        backbone_type (Literal["swin", "mvit"], optional): Type of backbone architecture. Defaults to "mvit".
        max_history_size (int, optional): Maximum number of historical timesteps. Defaults to 24.
        encoder_num_heads (int, optional): Number of attention heads in encoder. Defaults to 16.
        encoder_head_dim (int, optional): Dimension of each encoder attention head. Defaults to 64.
        encoder_depth (int, optional): Number of encoder layers. Defaults to 2.
        encoder_drop_rate (float, optional): Dropout rate in encoder. Defaults to 0.1.
        encoder_mlp_ratio (float, optional): MLP ratio in encoder. Defaults to 4.0.
        backbone_depth (int, optional): Number of backbone layers. Defaults to 4.
        backbone_num_heads (int, optional): Number of attention heads in backbone. Defaults to 1.
        backbone_mlp_ratio (float, optional): MLP ratio in backbone. Defaults to 4.0.
        backbone_drop_rate (float, optional): Dropout rate in backbone. Defaults to 0.1.
        mvit_attn_mode (str, optional): Attention mode for MViT backbone. Defaults to "conv".
        mvit_pool_first (bool, optional): Whether to pool before attention in MViT. Defaults to False.
        mvit_rel_pos (bool, optional): Whether to use relative positional encoding in MViT. Defaults to False.
        mvit_res_pool (bool, optional): Whether to use residual pooling in MViT. Defaults to True.
        mvit_dim_mul_attn (bool, optional): Whether to multiply dimensions in MViT attention. Defaults to False.
        decoder_num_heads (int, optional): Number of attention heads in decoder. Defaults to 16.
        decoder_head_dim (int, optional): Dimension of each decoder attention head. Defaults to 64.
        decoder_depth (int, optional): Number of decoder layers. Defaults to 2.
        decoder_drop_rate (float, optional): Dropout rate in decoder. Defaults to 0.1.
        decoder_mlp_ratio (float, optional): MLP ratio in decoder. Defaults to 4.0.
        **kwargs: Additional arguments passed to components

    Attributes:
        encoder (AQEncoder): Encoder component
        backbone (nn.Module): Backbone network (Swin or MViT)
        decoder (AQDecoder): Decoder component
        backbone_type (str): Type of backbone being used
    """

    def __init__(
        self,
        feature_names: dict[str, list[str]],
        embed_dim: int = 512,
        num_latent_tokens: int = 8,
        backbone_type: Literal["swin", "mvit"] = "mvit",
        max_history_size: int = 24,
        # Encoder params
        encoder_num_heads: int = 16,
        encoder_head_dim: int = 64,
        encoder_depth: int = 2,
        encoder_drop_rate: float = 0.1,
        encoder_mlp_ratio: float = 4.0,
        # Backbone params
        backbone_depth: int = 4,
        backbone_num_heads: int = 1,
        backbone_mlp_ratio: float = 4.0,
        backbone_drop_rate: float = 0.1,
        # MViT specific params
        mvit_attn_mode: str = "conv",
        mvit_pool_first: bool = False,
        mvit_rel_pos: bool = False,
        mvit_res_pool: bool = True,
        mvit_dim_mul_attn: bool = False,
        # Decoder params
        decoder_num_heads: int = 16,
        decoder_head_dim: int = 64,
        decoder_depth: int = 2,
        decoder_drop_rate: float = 0.1,
        decoder_mlp_ratio: float = 4.0,
        **kwargs,
    ):
        super().__init__()
        self.encoder = AQEncoder(
            feature_names=feature_names,
            latent_tokens=num_latent_tokens,
            embed_dim=embed_dim,
            max_history_size=max_history_size,
            num_heads=encoder_num_heads,
            head_dim=encoder_head_dim,
            depth=encoder_depth,
            drop_rate=encoder_drop_rate,
            mlp_ratio=encoder_mlp_ratio,
        )
        if backbone_type == "swin":
            self.backbone = Swin3DTransformer(
                embed_dim=embed_dim,
                encoder_depths=(backbone_depth,),
                encoder_num_heads=(backbone_num_heads,),
                decoder_depths=(backbone_depth,),
                decoder_num_heads=(backbone_num_heads,),
                window_size=(2, 1, 1),
                mlp_ratio=backbone_mlp_ratio,
                qkv_bias=True,
                drop_rate=backbone_drop_rate,
                attn_drop_rate=backbone_drop_rate,
                drop_path_rate=backbone_drop_rate,
                use_lora=False,
                skip_connections=False,
            )
        elif backbone_type == "mvit":
            self.backbone = MViT(
                patch_shape=[num_latent_tokens, 1, 1],
                embed_dim=embed_dim,
                depth=backbone_depth,
                num_heads=backbone_num_heads,
                mlp_ratio=backbone_mlp_ratio,
                qkv_bias=True,
                path_drop_rate=backbone_drop_rate,
                attn_mode=mvit_attn_mode,
                pool_first=mvit_pool_first,
                rel_pos=mvit_rel_pos,
                zero_init_rel=False,
                res_pool=mvit_res_pool,
                dim_mul_attn=mvit_dim_mul_attn,
                dim_scales=[(i, 1.0) for i in range(4)],  # no dimension scaling - timeseries are flat, sorta
                head_scales=[(1, 2.0), (2, 2.0)],  # scale only heads
                pool_kernel=[1, 1, 1],  # aand no pooling
                kv_stride=[1, 1, 1],
                q_stride=[(0, [1, 1, 1]), (1, [1, 1, 1]), (2, [1, 1, 1])],
            )
        else:
            raise ValueError(f"Unknown backbone type: {backbone_type}")

        self.backbone_type = backbone_type

        self.decoder = AQDecoder(
            feature_names=feature_names,
            embed_dim=embed_dim,
            num_heads=decoder_num_heads,
            head_dim=decoder_head_dim,
            depth=decoder_depth,
            drop_rate=decoder_drop_rate,
            mlp_ratio=decoder_mlp_ratio,
        )

    def forward(self, batch, lead_time: timedelta) -> dict[str, torch.Tensor]:
        """
        Forward pass of the AQFM model.

        Args:
            batch: Batch object containing input variables and metadata
            lead_time (timedelta): Time difference between input and target

        Returns:
            dict[str, torch.Tensor]: Dictionary mapping feature names to predicted values
        """
        encoded = self.encoder(batch, lead_time)

        patch_shape = [self.encoder.latent_tokens, 1, 1]

        backbone_output = self.backbone(encoded, lead_time=lead_time, rollout_step=0, patch_shape=patch_shape)

        predictions = self.decoder(backbone_output, batch, lead_time)

        return predictions


def main():
    """Main function for testing the AQFM implementation."""
    from datetime import timedelta
    from pathlib import Path
    from test.test_bfm_alternate_version.src.data_set import (
        AirQualityDataset,
        collate_aq_batches,
    )

    from torch.utils.data import DataLoader

    SEQUENCE_LENGTH = 48

    data_params = {
        "xlsx_path": Path(__file__).parent.parent.parent / "data/AirQuality.xlsx",
        "sequence_length": SEQUENCE_LENGTH,
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
    print("\nTarget shapes:")
    for name, tensor in targets.items():
        print(f"  {name}: {tensor.shape}")

    # testing both backbones
    for backbone in ["mvit", "swin"]:
        print(f"\nTesting {backbone.upper()} backbone:")
        model = AQFM(
            feature_names=batch.metadata.feature_names,
            embed_dim=512,
            num_latent_tokens=8,
            backbone_type=backbone,
            max_history_size=SEQUENCE_LENGTH,
        )
        lead_time = timedelta(hours=1)
        predictions = model(batch, lead_time)

        print("Prediction shapes:")
        for name, pred in predictions.items():
            print(f"  {name}: {pred.shape}")
        for name in predictions.keys():
            assert (
                predictions[name].shape == targets[name].shape
            ), f"Shape mismatch for {name}: prediction {predictions[name].shape} != target {targets[name].shape}"
        print(f"{backbone.upper()} shapes verified correctly!")


if __name__ == "__main__":
    main()
