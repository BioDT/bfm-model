import traceback
from datetime import timedelta

import torch

from src.mvit.mvit_model import MViT


def test_mvit():
    print("Testing MViT Model...")

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        embed_dim = 1024
        patch_shape = [8, 8, 16]  # [T, H, W] format

        model = MViT(
            patch_shape=patch_shape,
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
            # fixed dimension scaling
            dim_scales=[(i, 1.0) for i in range(4)],
            head_scales=[(1, 2.0), (2, 2.0)],
            # no spatial pooling - this is the default
            pool_kernel=[1, 1, 1],
            kv_stride=[1, 1, 1],
            q_stride=[(0, [1, 1, 1]), (1, [1, 1, 1]), (2, [1, 1, 1])],
        )
        model = model.to(device)
        print(f"Model created with patch_shape={patch_shape}")

        # dummy input
        batch_size = 32
        num_tokens = patch_shape[0] * patch_shape[1] * patch_shape[2]  # total number of tokens
        x = torch.randn(batch_size, num_tokens, embed_dim).to(device)
        lead_time = timedelta(hours=6)
        rollout_step = 0

        print(f"\nInput tensor shape: {x.shape}")

        # up up and away
        output = model(x, lead_time=lead_time, rollout_step=rollout_step, patch_shape=patch_shape)
        print("\nForward pass successful!")
        print(f"Output shape: {output.shape}")

        # check that the output shape matches the input shape
        assert output.shape == x.shape, f"Shape mismatch: expected {x.shape}, got {output.shape}"
        print("Shape verification passed!")

    except Exception as e:
        print(f"Test failed with error: {str(e)}\n:c")
        traceback.print_exc()


if __name__ == "__main__":
    test_mvit()
