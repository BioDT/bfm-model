import torch
import torch.nn as nn

from src.mvit.attention import MultiScaleAttention, MultiScaleTransformerBlock
from src.mvit.common import MLPBlock


def test_mlp():
    print("Testing MLPBlock...")
    mlp = MLPBlock(input_dim=256, hidden_dim=1024, output_dim=256)
    dummy_input = torch.randn(32, 1024, 256)  # [B, L, D]
    output = mlp(dummy_input)
    print(f"MLPBlock test {'passed' if output.shape == dummy_input.shape else 'failed - wow, that was unexpected'}")
    print()


def test_multi_scale_attention():
    print("Testing MultiScaleAttention...")

    # Test configuration 1: no relative positional embeddings
    msa_basic = MultiScaleAttention(
        input_dim=256,
        output_dim=256,
        spatial_size=[32, 32],
        num_heads=8,
        qkv_bias=False,
        query_kernel_size=[1, 1],
        kv_kernel_size=[1, 1],
        query_stride=[1, 1],
        kv_stride=[1, 1],
        norm_layer=nn.LayerNorm,
        has_cls_token=True,  # legacy stuff, just checking if it does not break anything
        pool_mode="conv",
        pool_first=False,
        use_rel_pos=False,
        zero_init_rel_pos=False,
        residual_pool=True,
    )

    # configuration 2: with relative positional embeddings (reason of testing - had some issues with this)
    msa_rel_pos = MultiScaleAttention(
        input_dim=256,
        output_dim=256,
        spatial_size=[32, 32],
        num_heads=8,
        qkv_bias=True,
        query_kernel_size=[1, 1],
        kv_kernel_size=[1, 1],
        query_stride=[1, 1],
        kv_stride=[1, 1],
        norm_layer=nn.LayerNorm,
        has_cls_token=False,
        pool_mode="conv",
        pool_first=True,
        use_rel_pos=True,
        zero_init_rel_pos=True,
        residual_pool=True,
    )

    # both configurations
    dummy_input = torch.randn(32, 1024, 256)  # [B, L, D]
    spatial_shape = [8, 32, 32]  # [T, H, W]

    print("\nTesting basic configuration...")
    output1, new_shape1 = msa_basic(dummy_input, spatial_shape)
    print(f"Basic config - Output shape: {output1.shape}, new spatial shape: {new_shape1}")

    print("\nTesting relative position configuration...")
    output2, new_shape2 = msa_rel_pos(dummy_input, spatial_shape)
    print(f"Rel pos config - Output shape: {output2.shape}, new spatial shape: {new_shape2}")

    print(f"MultiScaleAttention tests {'passed' if output1.shape == output2.shape == dummy_input.shape else 'failed'}")
    print()


def test_multi_scale_transformer_block():
    print("Testing MultiScaleTransformerBlock...")
    mstb = MultiScaleTransformerBlock(
        input_dim=256,
        output_dim=256,
        num_heads=8,
        spatial_size=[32, 32],
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_path_rate=0.1,
        norm_layer=nn.LayerNorm,
        query_kernel_size=[1, 1],
        kv_kernel_size=[1, 1],
        query_stride=[1, 1],
        kv_stride=[1, 1],
        pool_mode="conv",
        has_cls_token=False,
        pool_first=True,
        use_rel_pos=True,
        zero_init_rel_pos=True,
        residual_pool=True,
        scale_dim_in_attn=True,
    )

    dummy_input = torch.randn(32, 1024, 256)  # [B, L, D]
    spatial_shape = [8, 32, 32]  # [T, H, W]
    print(f"Input shape: {dummy_input.shape}, spatial shape: {spatial_shape}")

    output, new_shape = mstb(dummy_input, spatial_shape)
    print(f"Output shape: {output.shape}, new spatial shape: {new_shape}")
    print(f"MultiScaleTransformerBlock test {'passed' if output.shape == dummy_input.shape else 'failed'}")


if __name__ == "__main__":
    test_mlp()
    test_multi_scale_attention()
    test_multi_scale_transformer_block()
