from typing import List, Tuple

import numpy
import torch
import torch.nn as nn
from torch.nn.init import trunc_normal_

from bfm_model.mvit.common import MLPBlock, StochasticDepth


def attention_pool(
    input_tensor: torch.Tensor,
    pool_layer: nn.Module,
    spatial_shape: list,
    has_cls_token: bool = True,
    norm_layer: nn.Module = None,
) -> tuple[torch.Tensor, list]:
    """Applies pooling to attention tensor while handling CLS token and dimensions.

    Args:
        input_tensor: Input tensor to be pooled
            Shape: [B, N, L, C] or [B, L, C]
        pool_layer: Pooling layer (nn.MaxPool2d or nn.AvgPool2d or nn.Conv2d)
        spatial_shape: Temporal and spatial dimensions [T, H, W]
        has_cls_token: Whether input has CLS token. Default: True
        norm_layer: Optional normalization layer. Default: None

    Returns:
        output_tensor: Pooled tensor with same number of dimensions as input
        output_shape: Updated [T, H, W] after pooling
    """
    if pool_layer is None:
        return input_tensor, spatial_shape

    # Handle different input dimensions
    tensor_dim = input_tensor.ndim
    if tensor_dim == 4:
        pass  # Already in correct format [B, N, L, C]
    elif tensor_dim == 3:
        input_tensor = input_tensor.unsqueeze(1)  # Add head dimension [B, 1, L, C]
    else:
        raise NotImplementedError(f"Unsupported input dimension {input_tensor.shape}")

    # Split CLS token if present
    if has_cls_token:
        cls_token, input_tensor = input_tensor[:, :, :1, :], input_tensor[:, :, 1:, :]  # shape: [B, N, 1, C], [B, N, L-1, C]

    # Reshape for pooling
    B, N, L, C = input_tensor.shape
    T, H, W = spatial_shape
    # Reshape to [B*N, C, T, H, W] for pooling
    input_tensor = input_tensor.reshape(B * N, T, H, W, C).permute(0, 4, 1, 2, 3).contiguous()

    # Apply pooling
    output_tensor = pool_layer(input_tensor)  # shape: [B*N, C, T', H', W']

    # Get new spatial dimensions and reshape back
    output_shape = [output_tensor.shape[2], output_tensor.shape[3], output_tensor.shape[4]]
    L_pooled = numpy.prod(output_shape)
    output_tensor = output_tensor.reshape(B, N, C, L_pooled).transpose(2, 3)  # shape: [B, N, L', C]

    # Restore CLS token and apply normalization if needed
    if has_cls_token:
        output_tensor = torch.cat((cls_token, output_tensor), dim=2)  # shape: [B, N, L'+1, C]
    if norm_layer is not None:
        output_tensor = norm_layer(output_tensor)

    # Restore original dimensions if needed
    if tensor_dim == 3:
        output_tensor = output_tensor.squeeze(1)  # Remove head dimension

    return output_tensor, output_shape


def compute_relative_position_attention(
    attention_scores: torch.Tensor,
    query_tensor: torch.Tensor,
    has_cls_token: bool,  # kept for future compatibility
    query_spatial_shape: tuple[int, int],  # [H, W]
    key_spatial_shape: tuple[int, int],  # [H, W]
    rel_pos_height: torch.Tensor,
    rel_pos_width: torch.Tensor,
) -> torch.Tensor:
    """Computes relative positional attention maintaining separate H/W contributions."""
    query_height, query_width = query_spatial_shape
    key_height, key_width = key_spatial_shape
    B, n_head, seq_len, head_dim = query_tensor.shape

    spatial_size = query_height * query_width
    temporal_size = seq_len // spatial_size

    # computing the relative distances for height and width
    dist_h = torch.arange(query_height, device=attention_scores.device)[:, None]
    dist_h = dist_h - torch.arange(key_height, device=attention_scores.device)[None, :]
    dist_h = dist_h + (key_height - 1)

    dist_w = torch.arange(query_width, device=attention_scores.device)[:, None]
    dist_w = dist_w - torch.arange(key_width, device=attention_scores.device)[None, :]
    dist_w = dist_w + (key_width - 1)

    # Getting the relative position embeddings for height and width
    Rh = rel_pos_height[dist_h.long()]  # [H, H, D]
    Rw = rel_pos_width[dist_w.long()]  # [W, W, D]

    # processing each temporal step (latent levels)
    for t in range(temporal_size):
        start = t * spatial_size
        end = (t + 1) * spatial_size

        # getting the query content for the current temporal step
        q_t = query_tensor[:, :, start:end].view(B, n_head, query_height, query_width, head_dim)

        # computing the content-dependent relative position attention
        rel_h = torch.einsum("byhwc,hkc->byhwk", q_t, Rh)  # [B, H, qh, qw, kh]
        rel_w = torch.einsum("byhwc,wkc->byhwk", q_t, Rw)  # [B, H, qh, qw, kw]

        # getting the attention scores for the current temporal step
        attn_t = attention_scores[:, :, start:end, start:end].view(B, n_head, query_height, query_width, key_height, key_width)

        # Add content-dependent relative position attention
        attn_t = attn_t + rel_h[..., :, None] + rel_w[..., None, :]  # add height relative position  # add width relative position

        # update attention scores
        attention_scores[:, :, start:end, start:end] = attn_t.view(B, n_head, spatial_size, spatial_size)

    # TODO: Add support for CLS tokens if needed in the future.
    # Current implementation assumes has_cls_token=False for simplicity and efficiency.
    return attention_scores


class MultiScaleAttention(nn.Module):
    """Multi-Scale Attention module for video/image processing.

    Implements attention mechanism with spatial pooling and relative positional embeddings.
    Supports different pooling modes (conv, avg, max) and optional class token handling.

    Args:
        input_dim: Input dimension [int]
        output_dim: Output dimension [int]
        spatial_size: Spatial input dimensions [H, W]
        num_heads: Number of attention heads. Default: 8
        qkv_bias: Whether to use bias in QKV projections. Default: False
        query_kernel_size: Query pooling kernel size. Default: [1, 1]
        kv_kernel_size: Key/Value pooling kernel size. Default: [1, 1]
        query_stride: Query pooling stride. Default: [1, 1]
        kv_stride: Key/Value pooling stride. Default: [1, 1]
        norm_layer: Normalization layer. Default: nn.LayerNorm
        has_cls_token: Whether input has class token. Default: False
        pool_mode: Pooling mode ('conv', 'avg', 'max'). Default: 'conv'
        pool_first: Whether to pool before QKV projections. Default: False
        use_rel_pos: Use relative positional embeddings. Default: False
        zero_init_rel_pos: Zero init relative pos embeddings. Default: False
        residual_pool: Add residual connection after pooling. Default: True
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        spatial_size: List[int],
        num_heads: int = 8,
        qkv_bias: bool = False,
        query_kernel_size: List[int] = [1, 1],
        kv_kernel_size: List[int] = [1, 1],
        query_stride: List[int] = [1, 1],
        kv_stride: List[int] = [1, 1],
        norm_layer: nn.Module = nn.LayerNorm,
        has_cls_token: bool = False,
        pool_mode: str = "conv",
        pool_first: bool = False,
        use_rel_pos: bool = False,
        zero_init_rel_pos: bool = False,
        residual_pool: bool = True,
    ):
        super().__init__()
        # just initializing basic parameters
        self._init_basic_params(input_dim, output_dim, num_heads, pool_first, has_cls_token, pool_mode, residual_pool)
        # initializing the projections for QKV and output
        self._init_projections(qkv_bias)
        # initializing the pooling layers - these are the layers that will be used to pool the input tensor
        self._init_pooling(query_kernel_size, kv_kernel_size, query_stride, kv_stride, norm_layer, pool_mode)
        # initializing the relative positional embeddings - these are the embeddings that will be used to add relative positional information to the attention scores (optional)
        self._init_rel_pos(use_rel_pos, spatial_size, query_stride, kv_stride, zero_init_rel_pos)
        # validating the dimensions (output_dim must be divisible by num_heads, and query_kernel_size and kv_kernel_size must be greater than 0)
        self._validate_dimensions(output_dim, num_heads, query_kernel_size, kv_kernel_size)

    def _init_basic_params(
        self,
        input_dim: int,
        output_dim: int,
        num_heads: int,
        pool_first: bool,
        has_cls_token: bool,
        pool_mode: str,
        residual_pool: bool,
    ):
        """Initialize basic parameters for attention module"""
        self.pool_first = pool_first
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.output_dim = output_dim
        self.head_dim = input_dim // num_heads
        self.scale = self.head_dim**-0.5
        self.has_cls_token = has_cls_token
        self.pool_mode = pool_mode
        self.residual_pool = residual_pool

    def _init_projections(self, qkv_bias: bool):
        """Initialize projection layers based on pooling order"""
        if self.pool_first:
            self.query_proj = nn.Linear(self.input_dim, self.output_dim, bias=qkv_bias)
            self.key_proj = nn.Linear(self.input_dim, self.output_dim, bias=qkv_bias)
            self.value_proj = nn.Linear(self.input_dim, self.output_dim, bias=qkv_bias)
        else:
            self.qkv_proj = nn.Linear(self.input_dim, self.input_dim * 3, bias=qkv_bias)

        self.output_proj = nn.Linear(self.input_dim, self.output_dim)

    def _init_pooling(
        self,
        query_kernel_size: List[int],
        kv_kernel_size: List[int],
        query_stride: List[int],
        kv_stride: List[int],
        norm_layer: nn.Module,
        pool_mode: str,
    ):
        """Initialize pooling layers based on mode"""
        # calculate padding sizes
        padding_query = [k // 2 for k in query_kernel_size]
        padding_kv = [k // 2 for k in kv_kernel_size]

        # skip trivial pooling operations
        if numpy.prod(query_kernel_size) == 1 and numpy.prod(query_stride) == 1:
            query_kernel_size = []
        if numpy.prod(kv_kernel_size) == 1 and numpy.prod(kv_stride) == 1:
            kv_kernel_size = []

        if pool_mode in ["avg", "max"]:
            self._init_pool_layers_basic(
                pool_mode, query_kernel_size, kv_kernel_size, query_stride, kv_stride, padding_query, padding_kv
            )
        elif pool_mode in ["conv", "conv_unshared"]:
            self._init_pool_layers_conv(
                pool_mode, query_kernel_size, kv_kernel_size, query_stride, kv_stride, padding_query, padding_kv, norm_layer
            )
        else:
            raise NotImplementedError(f"Unsupported pooling mode: {pool_mode}")

    def _init_pool_layers_basic(
        self,
        pool_mode: str,
        query_kernel_size: List[int],
        kv_kernel_size: List[int],
        query_stride: List[int],
        kv_stride: List[int],
        padding_query: List[int],
        padding_kv: List[int],
    ):
        """Initialize max/average pooling layers"""
        pool_op = nn.MaxPool2d if pool_mode == "max" else nn.AvgPool2d
        self.query_pool = pool_op(query_kernel_size, query_stride, padding_query, ceil_mode=False) if query_kernel_size else None
        self.key_pool = pool_op(kv_kernel_size, kv_stride, padding_kv, ceil_mode=False) if kv_kernel_size else None
        self.value_pool = pool_op(kv_kernel_size, kv_stride, padding_kv, ceil_mode=False) if kv_kernel_size else None

    def _init_pool_layers_conv(
        self,
        pool_mode: str,
        query_kernel_size: List[int],
        kv_kernel_size: List[int],
        query_stride: List[int],
        kv_stride: List[int],
        padding_query: List[int],
        padding_kv: List[int],
        norm_layer: nn.Module,
    ):
        """Initialize convolutional pooling layers - the layers responsible for pooling the input tensor"""
        conv_dim = (self.input_dim if self.pool_first else self.output_dim) // (self.num_heads if pool_mode == "conv" else 1)

        if query_kernel_size:
            self.query_pool = nn.Conv2d(
                conv_dim, conv_dim, query_kernel_size, stride=query_stride, padding=padding_query, groups=conv_dim, bias=False
            )
            self.query_norm = norm_layer(conv_dim)
        else:
            self.query_pool = self.query_norm = None

        if kv_kernel_size:
            self.key_pool = nn.Conv2d(
                conv_dim, conv_dim, kv_kernel_size, stride=kv_stride, padding=padding_kv, groups=conv_dim, bias=False
            )
            self.value_pool = nn.Conv2d(
                conv_dim, conv_dim, kv_kernel_size, stride=kv_stride, padding=padding_kv, groups=conv_dim, bias=False
            )
            self.key_norm = self.value_norm = norm_layer(conv_dim)
        else:
            self.key_pool = self.value_pool = self.key_norm = self.value_norm = None

    def _init_rel_pos(
        self, use_rel_pos: bool, spatial_size: List[int], query_stride: List[int], kv_stride: List[int], zero_init_rel_pos: bool
    ):
        """Initialize relative positional embeddings"""
        self.use_rel_pos = use_rel_pos
        if not use_rel_pos:
            return

        if spatial_size[0] != spatial_size[1]:
            raise ValueError(f"Spatial dimensions must be equal, got {spatial_size}")

        size = spatial_size[0]
        query_size = size // query_stride[1] if query_stride else size
        kv_size = size // kv_stride[1] if kv_stride else size
        rel_pos_dim = 2 * max(query_size, kv_size) - 1

        self.rel_pos_height = nn.Parameter(torch.zeros(rel_pos_dim, self.head_dim))
        self.rel_pos_width = nn.Parameter(torch.zeros(rel_pos_dim, self.head_dim))

        if not zero_init_rel_pos:
            trunc_normal_(self.rel_pos_height, std=0.02)
            trunc_normal_(self.rel_pos_width, std=0.02)

    def _validate_dimensions(self, output_dim: int, num_heads: int, query_kernel_size: List[int], kv_kernel_size: List[int]):
        """Validate input dimensions and kernel sizes"""
        if output_dim % num_heads != 0:
            raise ValueError(f"output_dim ({output_dim}) must be divisible by num_heads ({num_heads})")
        if not all(k > 0 for k in query_kernel_size + kv_kernel_size):
            raise ValueError(f"Invalid kernel sizes: query={query_kernel_size}, kv={kv_kernel_size}")

    def forward(self, input_tensor: torch.Tensor, spatial_shape: List[int]) -> Tuple[torch.Tensor, List[int]]:
        """Forward pass of multi-scale attention.

        Args:
            input_tensor: Input tensor [B, L, D] where L = T*H*W
            spatial_shape: Input spatial dimensions [T, H, W]

        Returns:
            output_tensor: Output tensor [B, L, output_dim]
            output_shape: Output spatial dimensions [T, H, W]
        """
        batch_size, seq_len, _ = input_tensor.shape
        T, H, W = spatial_shape

        # generating Q,K,V based on pooling order
        if self.pool_first:  # if pooling first: reshape for spatial pooling
            fold_dim = 1 if self.pool_mode == "conv_unshared" else self.num_heads
            input_tensor = input_tensor.reshape(batch_size, seq_len, fold_dim, -1).permute(
                0, 2, 1, 3
            )  # shape: [B, fold_dim, L, D//fold_dim]
            query = key = value = input_tensor
        else:  # if attention first: apply QKV projection
            qkv = self.qkv_proj(input_tensor)  # shape: [B, L, 3*input_dim]
            qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, -1)  # shape: [B, L, 3, num_heads, head_dim]
            qkv = qkv.permute(2, 0, 3, 1, 4)  # shape: [3, B, num_heads, L, head_dim]
            query, key, value = qkv  # Each shape: [B, num_heads, L, head_dim]

        # applying spatial pooling to Q,K,V
        query, query_shape = attention_pool(
            query, self.query_pool, spatial_shape, has_cls_token=self.has_cls_token, norm_layer=getattr(self, "query_norm", None)
        )
        key, key_shape = attention_pool(
            key, self.key_pool, spatial_shape, has_cls_token=self.has_cls_token, norm_layer=getattr(self, "key_norm", None)
        )
        value, value_shape = attention_pool(
            value, self.value_pool, spatial_shape, has_cls_token=self.has_cls_token, norm_layer=getattr(self, "value_norm", None)
        )

        # computing scaled dot-product attention
        attention_scores = (query * self.scale) @ key.transpose(-2, -1)  # shape: [B, num_heads, L_q, L_k]

        # Add relative positional embeddings if enabled
        if self.use_rel_pos:
            query_height, query_width = query_shape[1:]  # extract spatial dimensions
            key_height, key_width = key_shape[1:]

            attention_scores = compute_relative_position_attention(
                attention_scores,
                query,
                self.has_cls_token,
                [query_height, query_width],
                [key_height, key_width],
                self.rel_pos_height,
                self.rel_pos_width,
            )

        # normalize attention weights and compute weighted sum
        attention_weights = attention_scores.softmax(dim=-1)  # shape: [B, num_heads, L_q, L_k]
        output_tensor = attention_weights @ value  # shape: [B, num_heads, L_q, head_dim]

        # Add residual connection from queries if enabled
        if self.residual_pool:
            if self.has_cls_token:
                output_tensor[:, :, 1:, :] += query[:, :, 1:, :]  # Skip CLS token
            else:
                output_tensor = output_tensor + query

        # Reshape and project output
        output_tensor = output_tensor.transpose(1, 2).reshape(batch_size, -1, self.output_dim)  # shape: [B, L, output_dim]
        output_tensor = self.output_proj(output_tensor)  # shape: [B, L, output_dim]

        return output_tensor, spatial_shape


class MultiScaleTransformerBlock(nn.Module):
    """Implements a multi-scale transformer block with attention and MLP layers.

    This is a key component of the MViT architecture, combining multi-scale attention
    with feed-forward processing while supporting various spatial pooling operations.
    It processes input tokens through self-attention and feed-forward layers while allowing
    for dynamic spatial resolution changes.

    Args:
        input_dim: Number of channels in the input features.
                   Determines the dimensionality of the token representations processed by the transformer block.
        output_dim: Number of channels in the output features.
                        Can differ from input_dim to enable progressive scaling of the feature dimension through the network.
        num_heads: Number of parallel attention heads in each attention layer.
        spatial_size: Height and width dimensions [H, W] of the input feature map.
                      Used to properly handle spatial operations and relative positioning.
        mlp_ratio: Multiplier for the hidden dimension size in the MLP layer relative to input_dim.
                   A larger ratio means a larger intermediate representation. Default: 4.0
        qkv_bias: If True, adds learnable bias terms to the query, key and value projections in the attention layer. Default: False
        drop_path_rate: Probability of dropping each path (i.e., residual connection) during training.
                       Higher values give stronger regularization. Default: 0.0
        norm_layer: Type of normalization to use, typically LayerNorm. Applied before
                    both attention and MLP layers as per standard transformer architecture. Default: nn.LayerNorm
        query_kernel_size: Size of convolutional kernel [kh, kw] used for query pooling.
                   Larger kernels capture more spatial context when pooling. Default: [1, 1]
        kv_kernel_size: Size of convolutional kernel [kh, kw] used for key and value pooling.
                       Can differ from query kernel to enable asymmetric attention patterns. Default: [1, 1]
        query_stride: Stride values [sh, sw] for query pooling.
                      Controls how much the spatial dimensions are reduced for queries. Default: [1, 1]
        kv_stride: Stride values [sh, sw] for key and value pooling.
                   Controls how much the spatial dimensions are reduced for keys and values. Default: [1, 1]
        pool_mode: Type of pooling operation to use in attention. Options:\n
                   - 'conv' Standard convolution-based pooling shared across heads;\n
                   - 'conv_unshared' Separate pooling operations per attention head;\n
                   Default: 'conv'.
        has_cls_token: Whether the input includes a classification token that should be
                       handled specially during pooling operations. Default: True
        pool_first: If True, applies pooling before computing attention scores.
                   If False, computes attention scores before pooling.
                   Affects efficiency and feature hierarchy. Default: False
        use_rel_pos: Whether to use relative spatial positional embeddings in the attention
                     computation. Helps model better understand spatial relationships. Default: False
        zero_init_rel_pos: If True, initializes relative position embeddings to zero.
                           Only relevant when use_rel_pos=True. Can help stabilize training. Default: False
        residual_pool: If True, adds a residual connection around the pooling operation.
                       Helps maintain feature fidelity through pooling layers. Default: True
        scale_dim_in_attn: If True, performs dimension scaling inside attention block.
                           If False, scales dimensions after attention computation. Default: False
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_heads: int,
        spatial_size: list[int],
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        drop_path_rate: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
        query_kernel_size: list[int] = [1, 1],
        kv_kernel_size: list[int] = [1, 1],
        query_stride: list[int] = [1, 1],
        kv_stride: list[int] = [1, 1],
        pool_mode: str = "conv",
        has_cls_token: bool = True,
        pool_first: bool = False,
        use_rel_pos: bool = False,
        zero_init_rel_pos: bool = False,
        residual_pool: bool = True,
        scale_dim_in_attn: bool = False,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.scale_dim_in_attn = scale_dim_in_attn
        self.has_cls_token = has_cls_token

        # 1st normalization layer
        self.norm1 = norm_layer(input_dim)

        attn_dim = output_dim if scale_dim_in_attn else input_dim

        self.attn = MultiScaleAttention(
            input_dim=input_dim,
            output_dim=attn_dim,
            num_heads=num_heads,
            spatial_size=spatial_size,
            qkv_bias=qkv_bias,
            query_kernel_size=query_kernel_size,
            kv_kernel_size=kv_kernel_size,
            query_stride=query_stride,
            kv_stride=kv_stride,
            norm_layer=norm_layer,
            has_cls_token=has_cls_token,
            pool_mode=pool_mode,
            pool_first=pool_first,
            use_rel_pos=use_rel_pos,
            zero_init_rel_pos=zero_init_rel_pos,
            residual_pool=residual_pool,
        )

        # As described - stochastic depth for regularization
        self.drop_path = StochasticDepth(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
        # 2nd normalization layer
        self.norm2 = norm_layer(attn_dim)

        mlp_hidden_dim = int(attn_dim * mlp_ratio)
        mlp_output_dim = output_dim
        self.mlp = MLPBlock(
            input_dim=attn_dim,
            hidden_dim=mlp_hidden_dim,
            output_dim=mlp_output_dim,
        )

        # Optional projection for dimension matching
        self.dim_proj = nn.Linear(input_dim, output_dim) if input_dim != output_dim else None

        # configruing skip connection pooling if needed
        if len(query_stride) > 0 and numpy.prod(query_stride) > 1:
            # pooling parameters based on stride
            kernel_skip = [s + 1 if s > 1 else s for s in query_stride]  # changing kernel size if stride is greater than 1
            padding_skip = [int(k // 2) for k in kernel_skip]  # calculating padding for kernel
            self.pool_skip = nn.MaxPool2d(kernel_size=kernel_skip, stride=query_stride, padding=padding_skip, ceil_mode=False)
        else:
            self.pool_skip = None

    def forward(self, input_tensor: torch.Tensor, spatial_shape: list) -> tuple[torch.Tensor, list]:
        """Forward pass through the transformer block.

        Args:
            input_tensor: Input tensor with shape [B, L, D] where:
            - B = batch size
            - L = sequence length
            - D = embedding dimension
            spatial_shape: Current [T, H, W] dimensions - resolution of a patch

        Returns:
            output_tensor: Processed tensor with shape [B, L', D_out]
            output_shape: Updated [T, H, W] after pooling
        """
        # storing input for skip connections
        identity = input_tensor  # shape: [B, L, D]

        # 1st branch: multi-scale attention
        norm_tensor = self.norm1(input_tensor)  # shape: [B, L, D]
        attn_output, output_shape = self.attn(norm_tensor, spatial_shape)  # shape: [B, L', D_out]

        # Handling dimension matching and skip connections
        if self.scale_dim_in_attn and self.dim_proj is not None:
            identity = self.dim_proj(norm_tensor)  # Project if dims don't match

        # applying skip pooling if configured - that would be used for skip connections
        identity_pooled, _ = attention_pool(
            identity, self.pool_skip, spatial_shape, has_cls_token=self.has_cls_token
        )  # shape: [B, L', D]

        # combining attention output with skip connection
        output_tensor = identity_pooled + self.drop_path(attn_output)  # shape: [B, L', D_out]

        # and now - 2nd branch: MLP :)
        norm_tensor = self.norm2(output_tensor)  # shape: [B, L', D_out]
        mlp_output = self.mlp(norm_tensor)  # shape: [B, L', D_out]

        # projecting dimensions if needed and not done in attention
        if not self.scale_dim_in_attn and self.dim_proj is not None:
            output_tensor = self.dim_proj(norm_tensor)

        # adding MLP output with skip connection
        output_tensor = output_tensor + self.drop_path(mlp_output)  # shape: [B, L', D_out]

        return output_tensor, output_shape
