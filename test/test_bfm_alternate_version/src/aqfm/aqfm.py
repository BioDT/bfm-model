from datetime import timedelta
from test.test_bfm_alternate_version.src.aqfm.decoder import AQDecoder
from test.test_bfm_alternate_version.src.aqfm.encoder import AQEncoder
from typing import Literal

import torch
import torch.nn as nn

from src.mvit.mvit_model import MViT
from src.swin_transformer.core.swim_core_v2 import Swin3DTransformer


class AQFM(nn.Module):
    def __init__(
        self,
        feature_names: dict[str, list[str]],
        embed_dim: int = 512,
        num_latent_tokens: int = 8,
        backbone_type: Literal["swin", "mvit"] = "mvit",
        max_history_size: int = 24,
        **kwargs,
    ):
        super().__init__()
        self.encoder = AQEncoder(
            feature_names=feature_names,
            latent_tokens=num_latent_tokens,
            embed_dim=embed_dim,
            max_history_size=max_history_size,
            **kwargs,
        )
        if backbone_type == "swin":
            self.backbone = Swin3DTransformer(
                embed_dim=embed_dim,
                encoder_depths=(2,),
                encoder_num_heads=(8,),  # barely any depth, for simplicity, and since we are operating on time series
                decoder_depths=(2,),
                decoder_num_heads=(8,),
                window_size=(2, 1, 1),
                mlp_ratio=4.0,
                qkv_bias=True,
                drop_rate=0.0,
                attn_drop_rate=0.0,
                drop_path_rate=0.1,
                use_lora=False,
                skip_connections=False,  # no skip connections - for a simple forward pass
            )
        elif backbone_type == "mvit":
            self.backbone = MViT(
                patch_shape=[num_latent_tokens, 1, 1],  # treating timeseries as 3D image, flat temporal dimension
                embed_dim=embed_dim,
                depth=4,
                num_heads=1,
                mlp_ratio=4.0,
                qkv_bias=True,
                path_drop_rate=0.1,
                attn_mode="conv",
                pool_first=False,
                rel_pos=False,
                zero_init_rel=False,
                res_pool=True,
                dim_mul_attn=False,
                dim_scales=[(i, 1.0) for i in range(4)],  # no dimension scaling - timeseries are flat, sorta
                head_scales=[(1, 2.0), (2, 2.0)],  # scale only heads
                pool_kernel=[1, 1, 1],  # aand no pooling
                kv_stride=[1, 1, 1],
                q_stride=[(0, [1, 1, 1]), (1, [1, 1, 1]), (2, [1, 1, 1])],
            )
        else:
            raise ValueError(f"Unknown backbone type: {backbone_type}")

        self.backbone_type = backbone_type

        self.decoder = AQDecoder(feature_names=feature_names, embed_dim=embed_dim, **kwargs)

    def forward(self, batch, lead_time: timedelta) -> dict[str, torch.Tensor]:
        encoded = self.encoder(batch, lead_time)

        patch_shape = [self.encoder.latent_tokens, 1, 1]

        backbone_output = self.backbone(encoded, lead_time=lead_time, rollout_step=0, patch_shape=patch_shape)

        predictions = self.decoder(backbone_output, batch, lead_time)

        return predictions


def main():
    from datetime import timedelta
    from pathlib import Path
    from test.test_bfm_alternate_version.src.data_set import (
        AirQualityDataset,
        collate_aq_batches,
    )

    from torch.utils.data import DataLoader

    # making a data set just as in encoder.py, data_set.py, decoder.py, and now here as well (:
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
            max_history_size=24,
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
