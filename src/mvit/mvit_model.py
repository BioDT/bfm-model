import math
from functools import partial

import torch
import torch.nn as nn
from fvcore.common.registry import Registry
from torch.nn.init import trunc_normal_

from src.mvit.attention import MultiScaleTransformerBlock

model_registry = Registry("MODEL")


class PatchEmbed(nn.Module):
    """
    Token embedding layer that maintains spatial information.
    Projects input tokens to the embedding dimension while preserving patch resolution.

    Args:
        input_dim: Input token dimension
        embed_dim: Output embedding dimension
        patch_shape: Patch resolution [T, H, W]
    """

    def __init__(
        self,
        input_dim: int,
        embed_dim: int,
        patch_shape: list[int],  # [T, H, W]
    ):
        super().__init__()
        print(f"Initializing PatchEmbed with input_dim={input_dim}, embed_dim={embed_dim}")
        self.patch_shape = patch_shape
        # Linear projection to embedding dimension
        self.proj = nn.Linear(input_dim, embed_dim)  # shape: [B, L, input_dim] -> [B, L, embed_dim]

    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, list[int]]:
        """Forward pass for token embedding, nothing too fancy"""
        tokens = self.proj(tokens)  # shape: [B, L, embed_dim]
        return tokens, self.patch_shape


@model_registry.register()
class MViT(nn.Module):
    """
    Multiscale Vision Transformer (MViT) implementation. MViT processes input tokens through multiple transformer blocks
    while progressively scaling attention heads, embedding dimensions, and spatial resolution.

    This implementation is based on the original MViT paper, but is modified to work, instead of raw image data, with latent varaibles.

    Args:
        patch_shape: Spatial dimensions of input patches in format [Latent levels, Height, Width].
                            Each dimension represents the number of patches in that axis.
        embed_dim: Initial dimension of token embeddings.
                        Will be scaled up through the network. Default: 1024
        depth: Number of transformer blocks in the network.
                    Each block contains attention and MLP layers. Default: 4
        num_heads: Initial number of attention heads.
                        Will be scaled up through the network based on `head_scales`. Default: 1
        mlp_ratio: Multiplier for hidden dimension in MLP layers relative to embedding dimension. Default: 4.0
        qkv_bias: Whether to include bias terms in Query, Key and Value projections. Default: True
        path_drop_rate: Stochcstic depth rate - probability of dropping residual paths.
                                Higher values mean more aggressive regularization. Default: 0.1
        attn_mode: Type of attention mechanism to use. Options include 'conv' for convolution-based attention. Default: 'conv'
        pool_first: If True, applies pooling before computing attention. If False, computes attention before pooling. Default: False
        rel_pos: Whether to use relative spatial positional embeddings in attention computation. Default: False
        zero_init_rel: If True, initializes relative position embeddings to zero. Only relevant if rel_pos=True. Default: False
        res_pool: Whether to add residual conncetions around pooling layers. Default: True
        dim_mul_attn: If True, applies dimension scaling inside attention blocks. If False, scales after attention. Default: False
        dim_scales: Schedule for scaling embedding dimensions. Each tuple contains (layer_index, scale_factor). Default: [(1, 2.0), (2, 2.0)]
        head_scales: Schedule for scaling number of attention heads. Each tuple contains (layer_index, scale_factor). Default: [(1, 2.0), (2, 2.0)]
        pool_kernel: Kernel sizes [kt, kh, kw] used for pooling operations in temporal and spatial dimensions. Default: [3, 3, 3]
        kv_stride: Stride sizes [st, sh, sw] for Key and Value tensors in attention computation. Default: [1, 2, 2]
        q_stride: Schedule for Query stride sizes. Each tuple contains (layer_index, [st, sh, sw]).
                        Used to control the spatial resolution of the attention computation.
                        Default: [(0, [1, 2, 2]), (1, [1, 2, 2]), (2, [1, 2, 2])]
    """

    def __init__(
        self,
        patch_shape: list[int],
        embed_dim: int = 1024,
        depth: int = 4,
        num_heads: int = 1,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        path_drop_rate: float = 0.1,
        attn_mode: str = "conv",
        pool_first: bool = False,
        rel_pos: bool = False,
        zero_init_rel: bool = False,
        res_pool: bool = True,
        dim_mul_attn: bool = False,
        dim_scales: list = [(1, 2.0), (2, 2.0)],
        head_scales: list = [(1, 2.0), (2, 2.0)],
        pool_kernel: list = [3, 3, 3],
        kv_stride: list = [1, 2, 2],
        q_stride: list = [(0, [1, 2, 2]), (1, [1, 2, 2]), (2, [1, 2, 2])],
    ):
        super().__init__()
        print(f"\nInitializing MViT with patch_shape={patch_shape}")
        print(f"Model parameters: embed_dim={embed_dim}, depth={depth}, num_heads={num_heads}")

        # initialize embeddings, scaling factors, and pooling configs
        self._init_embeddings(patch_shape, embed_dim)
        dim_mults, head_mults = self._init_scaling_factors(depth, dim_scales, head_scales)
        pool_q, pool_kv, stride_q, stride_kv = self._init_pooling_configs(depth, q_stride, kv_stride, pool_kernel)

        # and the core part - transformer blocks
        self._build_transformer_blocks(
            patch_shape,
            depth,
            embed_dim,
            num_heads,
            dim_mults,
            head_mults,
            pool_q,
            pool_kv,
            stride_q,
            stride_kv,
            mlp_ratio,
            qkv_bias,
            path_drop_rate,
            attn_mode,
            pool_first,
            rel_pos,
            zero_init_rel,
            res_pool,
            dim_mul_attn,
        )

        # Initialize weights
        self.apply(self._init_weights)
        trunc_normal_(self.pos_embed, std=0.02)

    def _init_embeddings(self, patch_shape: list[int], embed_dim: int):
        """Initialize patch and position embeddings"""
        self.patch_embed = PatchEmbed(embed_dim, embed_dim, patch_shape)
        num_patches = torch.prod(torch.tensor(patch_shape)).item()
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))

    def _init_scaling_factors(self, depth: int, dim_scales: list, head_scales: list) -> tuple[torch.Tensor, torch.Tensor]:
        """Initialize progressive scaling factors for dimensions and heads"""
        dim_mults = torch.ones(depth + 1)
        head_mults = torch.ones(depth + 1)

        if dim_scales:
            idx, scales = zip(*dim_scales)
            dim_mults[torch.tensor(idx)] = torch.tensor(scales)
        if head_scales:
            idx, scales = zip(*head_scales)
            head_mults[torch.tensor(idx)] = torch.tensor(scales)

        return dim_mults, head_mults

    def _init_pooling_configs(
        self, depth: int, q_stride: list, kv_stride: list, pool_kernel: list
    ) -> tuple[list, list, list, list]:
        """Initialize pooling and stride configurations"""
        pool_q = [[] for _ in range(depth)]
        pool_kv = [[] for _ in range(depth)]
        stride_q = [[] for _ in range(depth)]
        stride_kv = [[] for _ in range(depth)]

        # Configure Q pooling
        for idx, stride in q_stride:
            stride_q[idx] = stride
            pool_q[idx] = pool_kernel

        # Configure KV pooling
        if kv_stride:
            curr_stride = torch.tensor(kv_stride)
            for i in range(depth):
                if stride_q[i]:
                    curr_stride = torch.maximum(curr_stride // torch.tensor(stride_q[i]), torch.ones_like(curr_stride))
                    stride_kv[i] = curr_stride.tolist()
                    pool_kv[i] = pool_kernel

        return pool_q, pool_kv, stride_q, stride_kv

    def _build_transformer_blocks(
        self,
        patch_shape: list[int],
        depth: int,
        embed_dim: int,
        num_heads: int,
        dim_mults: torch.Tensor,
        head_mults: torch.Tensor,
        pool_q: list,
        pool_kv: list,
        stride_q: list,
        stride_kv: list,
        mlp_ratio: float,
        qkv_bias: bool,
        path_drop_rate: float,
        attn_mode: str,
        pool_first: bool,
        rel_pos: bool,
        zero_init_rel: bool,
        res_pool: bool,
        dim_mul_attn: bool,
    ):
        """Build transformer blocks with all configurations"""
        spatial_size = patch_shape[1:]  # [H, W]
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        drop_rates = torch.linspace(0, path_drop_rate, depth).tolist()

        self.blocks = nn.ModuleList()
        for i in range(depth):
            curr_heads = round(num_heads * head_mults[i].item())
            next_dim = round(embed_dim * dim_mults[i + 1].item())

            block = MultiScaleTransformerBlock(
                input_dim=embed_dim,
                output_dim=next_dim,
                num_heads=curr_heads,
                spatial_size=spatial_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop_path_rate=drop_rates[i],
                norm_layer=norm_layer,
                query_kernel_size=pool_q[i] if pool_q[i] else [1, 1],
                kv_kernel_size=pool_kv[i] if pool_kv[i] else [1, 1],
                query_stride=stride_q[i] if stride_q[i] else [1, 1],
                kv_stride=stride_kv[i] if stride_kv[i] else [1, 1],
                pool_mode=attn_mode,
                has_cls_token=False,  # left as legacy from original MViT v2, TODO: potentially remove
                pool_first=pool_first,
                use_rel_pos=rel_pos,
                zero_init_rel_pos=zero_init_rel,
                residual_pool=res_pool,
                scale_dim_in_attn=dim_mul_attn,
            )

            if stride_q[i]:
                spatial_size = [size // stride for size, stride in zip(spatial_size, stride_q[i])]

            self.blocks.append(block)
            embed_dim = next_dim

        # Final normalization and projection
        self.norm = norm_layer(embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)  # shape: [B, L, D] -> [B, L, D]

    def _init_weights(self, module: nn.Module):
        """Initialize network weights."""
        if isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)

    def forward(self, tokens: torch.Tensor, lead_time=None, rollout_step=None, patch_shape=None) -> torch.Tensor:
        """
        Forward pass through MViT.

        Args:
            tokens: Input tensor of shape [B, L, D]
            lead_time: Optional forecast lead time
            rollout_step: Optional rollout step
            patch_shape: Optional patch resolution override

        Returns:
            tokens: Output tensor of shape [B, L, D]
        """
        batch_size, seq_len, dim = tokens.shape
        print("\nMViT Forward Pass:")
        print(f"Input shape: [B={batch_size}, L={seq_len}, D={dim}]")
        print(f"Patch shape: {patch_shape}")
        print(f"Position embedding shape: {self.pos_embed.shape}")

        # Token embedding and position embedding in one forward pass
        tokens, spatial_shape = self.patch_embed(tokens)
        tokens = tokens + self.pos_embed

        print(f"After embeddings: {tokens.shape}")
        print(f"Spatial shape: {spatial_shape}")

        # Process through transformer blocks
        curr_shape = spatial_shape  # [T, H, W] spatial dimensions
        print(f"Initial spatial shape: {curr_shape}")

        # Forward through transformer blocks
        for i, block in enumerate(self.blocks):
            print(f"\nProcessing block {i}")
            tokens, curr_shape = block(tokens, curr_shape)  # shape: [B, L, D]
            print(f"After block {i}: shape={tokens.shape}, spatial_shape={curr_shape}")

        # Final normalization and projection
        tokens = self.proj(self.norm(tokens))
        print(f"Final output: {tokens.shape}")

        return tokens
