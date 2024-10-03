from datetime import timedelta

import torch
import torch.nn as nn
from timm.models.layers import to_3tuple

from src.swin_transformer.core.swin_components import (
    PatchMerging3D,
    PatchSplitting3D,
    SwinTransformer3DLayer,
)
from src.swin_transformer.helpers.fourier_expansion import lead_time_expansion
from src.swin_transformer.helpers.low_rank_adaptation import LoRAMode, LoRARollout
from src.swin_transformer.helpers.utilities import init_weights


class Basic3DEncoderLayer(SwinTransformer3DLayer):
    """
    A basic 3D Swin Transformer encoder layer.
    Used for Fully Sharded Data Parallelism (i.e., for training on multiple GPUs).
    This requires a subclass.
    """


class Basic3DDecoderLayer(SwinTransformer3DLayer):
    """
    A basic 3D Swin Transformer decoder layer.
    Used for Fully Sharded Data Parallelism (i.e., for training on multiple GPUs).
    This requires a subclass.
    """


class Swin3DTransformer(nn.Module):
    """Swin 3D Transformer backbone for processing 3D data with multi-scale feature extraction."""

    def __init__(
        self,
        embed_dim: int = 96,
        encoder_depths: tuple[int, ...] = (2, 2, 6, 2),
        encoder_num_heads: tuple[int, ...] = (3, 6, 12, 24),
        decoder_depths: tuple[int, ...] = (2, 6, 2, 2),
        decoder_num_heads: tuple[int, ...] = (24, 12, 6, 3),
        window_size: int | tuple[int, int, int] = 7,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.1,
        drop_path_rate: float = 0.1,
        lora_steps: int = 40,
        lora_mode: LoRAMode = "single",
        use_lora: bool = False,
    ) -> None:
        """
        Args:
            embed_dim (int): Patch embedding dimension. Default: 96
            encoder_depths (tuple[int, ...]): Number of blocks in each encoder layer. Default: [2, 2, 6, 2]
            encoder_num_heads (tuple[int, ...]): Number of attention heads in each encoder layer. Default: [3, 6, 12, 24]
            decoder_depths (tuple[int, ...]): Number of blocks in each decoder layer. Default: [2, 6, 2, 2]
            decoder_num_heads (tuple[int, ...]): Number of attention heads in each decoder layer. Default: [24, 12, 6, 3]
            window_size (int | tuple[int, int, int]): Window size for attention computation. Default: 7
            mlp_ratio (float): Ratio of MLP hidden dim to embedding dim. Default: 4.0
            qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
            drop_rate (float): Dropout rate. Default: 0.0
            attn_drop_rate (float): Attention dropout rate. Default: 0.1
            drop_path_rate (float): Stochastic depth rate. Default: 0.1
            lora_steps (int): Maximum number of LoRA roll-out steps. Default: 40
            lora_mode (LoRAMode): Mode for LoRA. Default: "single"
            use_lora (bool): Enable LoRA. Default: False
        """
        super().__init__()

        self.window_size = to_3tuple(window_size)
        self.num_encoder_layers = len(encoder_depths)
        self.num_decoder_layers = len(decoder_depths)
        self.embed_dim = embed_dim
        self.mlp_ratio = mlp_ratio

        # Time embedding MLP for temporal information processing
        self.time_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim, bias=True),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim, bias=True),
        )

        assert sum(encoder_depths) == sum(decoder_depths), "Encoder and decoder depths must sum to the same value"

        # Calculate drop path rate for each layer
        dpr = torch.linspace(0, drop_path_rate, sum(encoder_depths)).tolist()

        # Build encoder layers
        self.encoder_layers = nn.ModuleList()
        for i_layer in range(self.num_encoder_layers):
            layer = Basic3DEncoderLayer(
                input_dim=int(embed_dim * 2**i_layer),
                depth=encoder_depths[i_layer],
                num_heads=encoder_num_heads[i_layer],
                window_size=self.window_size,
                mlp_ratio=self.mlp_ratio,
                temporal_dim=embed_dim,
                qkv_bias=qkv_bias,
                dropout=drop_rate,
                attention_dropout_rate=attn_drop_rate,
                drop_path=dpr[sum(encoder_depths[:i_layer]) : sum(encoder_depths[: i_layer + 1])],
                downsample=(PatchMerging3D if (i_layer < self.num_encoder_layers - 1) else None),
                use_lora=use_lora,
                lora_steps=lora_steps,
                lora_mode=lora_mode,
            )
            self.encoder_layers.append(layer)

        # Build decoder layers
        self.decoder_layers = nn.ModuleList()
        for i_layer in range(self.num_decoder_layers):
            exponent = self.num_decoder_layers - i_layer - 1
            layer = Basic3DDecoderLayer(
                input_dim=int(embed_dim * 2**exponent),
                depth=decoder_depths[i_layer],
                num_heads=decoder_num_heads[i_layer],
                window_size=self.window_size,
                mlp_ratio=self.mlp_ratio,
                temporal_dim=embed_dim,
                qkv_bias=qkv_bias,
                dropout=drop_rate,
                attention_dropout_rate=attn_drop_rate,
                drop_path=dpr[sum(decoder_depths[:i_layer]) : sum(decoder_depths[: i_layer + 1])],
                upsample=(PatchSplitting3D if (i_layer < self.num_decoder_layers - 1) else None),
                use_lora=use_lora,
                lora_steps=lora_steps,
                lora_mode=lora_mode,
            )
            self.decoder_layers.append(layer)

        self.apply(init_weights)

        # Initialize post-normalization layers in the residual connections
        for layer in self.encoder_layers + self.decoder_layers:
            layer.init_respostnorm()

        # Final projection layer to match input dimensions
        self.final_proj = nn.Linear(2 * self.embed_dim, self.embed_dim)

    def get_encoder_specs(self, patch_res: tuple[int, int, int]) -> tuple[list[tuple[int, int, int]], list[tuple[int, int, int]]]:
        """
        Calculate the input resolution and output padding for each encoder layer.

        Args:
            patch_res (tuple[int, int, int]): Initial patch resolution (C, H, W)

        Returns:
            tuple[list[tuple[int, int, int]], list[tuple[int, int, int]]]:
                List of resolutions and list of paddings for each layer
        """
        print(f"Initial patch resolution: {patch_res}")
        all_res = [patch_res]
        padded_outs = []
        for i in range(1, self.num_encoder_layers):
            C, H, W = all_res[-1]
            pad_H, pad_W = H % 2, W % 2
            padded_outs.append((0, pad_H, pad_W))
            new_res = (C, (H + pad_H) // 2, (W + pad_W) // 2)
            all_res.append(new_res)
            print(f"Layer {i}: Input res: {all_res[-2]}, Padding: {padded_outs[-1]}, Output res: {new_res}")

        padded_outs.append((0, 0, 0))
        print(f"Final all_res: {all_res}")
        print(f"Final padded_outs: {padded_outs}")
        return all_res, padded_outs

    def forward(
        self,
        x: torch.Tensor,
        lead_time: timedelta,
        rollout_step: int,
        patch_res: tuple[int, int, int],
    ) -> torch.Tensor:
        """
        Forward pass of the Swin 3D Transformer backbone.

        Args:
            x (torch.Tensor): Input tokens. Shape: [B, L, D]
            lead_time (timedelta): Lead time for temporal information
            rollout_step (int): Current roll-out step
            patch_res (tuple[int, int, int]): Patch resolution (C, H, W)

        Returns:
            torch.Tensor: Processed tokens. Shape: [B, L, D]
        """
        B, L, D = x.shape
        print(f"Input shape: {x.shape}")
        print(f"Patch resolution: {patch_res}")
        print(f"L size: {L}")
        print(f"Patch res size: {patch_res[0] * patch_res[1] * patch_res[2]}")
        assert L == patch_res[0] * patch_res[1] * patch_res[2], "Input shape does not match patch size"
        assert (
            patch_res[1] % self.window_size[1] == 0
        ), f"Patch height ({patch_res[1]}) must be divisible by window size ({self.window_size[1]})"
        assert (
            patch_res[2] % self.window_size[2] == 0
        ), f"Patch width ({patch_res[2]}) must be divisible by window size ({self.window_size[2]})"

        all_enc_res, padded_outs = self.get_encoder_specs(patch_res)

        # Process lead time information
        lead_hours = lead_time.total_seconds() / 3600
        lead_times = torch.full((B,), lead_hours, dtype=torch.float32, device=x.device)
        c = self.time_mlp(lead_time_expansion(lead_times, self.embed_dim).to(dtype=x.dtype))

        # Encoder forward pass
        skips = []
        for i, layer in enumerate(self.encoder_layers):
            print(f"Encoder Layer {i}: Input shape: {x.shape}, Resolution: {all_enc_res[i]}")
            x, x_unscaled = layer(x, c, all_enc_res[i], rollout_step=rollout_step)
            print(f"Encoder Layer {i}: Output shape: {x.shape}")
            skips.append(x_unscaled)

        # Decoder forward pass with skip connections
        for i, layer in enumerate(self.decoder_layers):
            index = self.num_decoder_layers - i - 1
            print(
                f"Decoder Layer {i}: Input shape: {x.shape}, Resolution: {all_enc_res[index]}, Padding: {padded_outs[index - 1]}"
            )
            x, _ = layer(x, c, all_enc_res[index], padded_outs[index - 1], rollout_step=rollout_step)
            print(f"Decoder Layer {i}: Output shape: {x.shape}")

            if 0 < i < self.num_decoder_layers - 1:
                x = x + skips[index - 1]  # Additive skip connection
            elif i == self.num_decoder_layers - 1:
                x = torch.cat([x, skips[0]], dim=-1)  # Concatenation for the last stage

        # Final projection to match input dimensions
        x = self.final_proj(x)  # shape: [B, L, D]
        print(f"Final output shape: {x.shape}")

        return x


def test_swin_transformer_backbone():
    # Initialize the Swin3DTransformerBackbone with matching encoder and decoder depths
    import time

    start_time = time.time()
    backbone = Swin3DTransformer(
        embed_dim=96,
        encoder_depths=(2, 2, 6, 2),
        encoder_num_heads=(3, 6, 12, 24),
        decoder_depths=(2, 2, 6, 2),  # Match encoder_depths
        decoder_num_heads=(24, 12, 6, 3),
        window_size=(2, 7, 7),
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.1,
        drop_path_rate=0.1,
        use_lora=False,
    )

    # Set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone = backbone.to(device)

    # Create dummy input data
    batch_size = 2
    patch_res = (2, 56, 56)  # (C, H, W)
    x = torch.randn(batch_size, patch_res[0] * patch_res[1] * patch_res[2], 96).to(device)
    lead_time = timedelta(hours=6)
    rollout_step = 0

    # Run a forward pass
    try:
        output = backbone(x, lead_time, rollout_step, patch_res)
        print(f"Forward pass successful. Output shape: {output.shape}")
        assert output.shape == x.shape, f"Expected output shape {x.shape}, but got {output.shape}"
        print("Test passed successfully!")
    except Exception as e:
        print(f"Test failed with error: {str(e)}")

    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")


if __name__ == "__main__":
    test_swin_transformer_backbone()
