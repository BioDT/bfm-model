from datetime import timedelta
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.models.layers import DropPath, to_3tuple

from bfm_model.swin_transformer.helpers.adaptive_layer_norm import AdaptiveLayerNorm
from bfm_model.swin_transformer.helpers.fourier_expansion import lead_time_expansion
from bfm_model.swin_transformer.helpers.low_rank_adaptation import LoRAMode, LoRARollout
from bfm_model.swin_transformer.helpers.vera import VeRA, VeRAMode, VeRARollout

from bfm_model.swin_transformer.helpers.utilities import adjust_windows, init_weights
from bfm_model.swin_transformer.helpers.window_operations import (
    apply_or_remove_3d_padding,
    compute_3d_shifted_window_mask,
    window_partition_and_reverse_3d,
)


class MLP(nn.Module):
    """A simple MLP with a single hidden layer and dropout after the hidden layer at the end."""

    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        activation: type = nn.GELU,
        dropout: float = 0.0,
    ) -> None:
        """
        Args:
            in_features (int): Input dimensionality.
            hidden_features (int, optional): Hidden layer dimensionality. Defaults to the input
                dimensionality.
            out_features (int, optional): Output dimensionality. Defaults to the input
                dimensionality.
            activation (type, optional): Activation function to use. Will be instantiated as
                `activation()`. Defaults to `torch.nn.GELU`.
            dropout (float, optional): Drop-out rate. Defaults to no drop-out.
        """
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = activation()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """No need for documentation, hopefully :)"""
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class WindowMultiHeadSelfAttention(nn.Module):
    """Window-based multi-head self-attention (W-MSA) module.

    This module supports both shifted and non-shifted window attention mechanisms.
    It implements the core attention computation for the Swin Transformer architecture.
    """

    def __init__(
        self,
        dim: int,
        window_size: list[int, int, int],
        num_heads: int,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        attention_dropout_rate: float = 0.0,
        proj_dropout: float = 0.0,
        peft_r: int = 8,
        lora_alpha: int = 8,
        d_initial: float = 0.1,
        peft_dropout: float = 0.0,
        peft_steps: int = 40,
        peft_mode: LoRAMode = "single",
        use_lora: bool = False,
        use_vera: bool = False,
    ) -> None:
        """
        Initialize the WindowMultiHeadSelfAttention module.

        Args:
            dim (int): Number of input channels.
            window_size (list[int, int, int]): The size of the attention windows.
            num_heads (int): Number of attention heads.
            qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
            qk_scale (float, optional): Override default qk scale of head_dim ** -0.5 if set. Default: None
            attention_dropout_rate (float): Dropout rate for attention weights. Default: 0.0
            proj_dropout (float): Dropout rate for projection output. Default: 0.0
            peft_r (int): Rank for Parameter Efficient Fine Tuning: 
                a) Low-Rank Adaptation (LoRA). Default: 8 | 
                b) Vector-based Random-matrix Adaptation. Default: 256
            lora_alpha (int): Scaling factor for LoRA. Default: 8
            d_initial (float): Initialization factor for lamda vector. Default: 0.1
            peft_dropout (float): Dropout rate for PEFTs. Default: 0.0
            peft_steps (int): Maximum number of PEFTs roll-out steps. Default: 40
            lora_mode (LoRAMode): Mode for LoRA application. Default: "single" => Same with VeRA
            use_lora (bool): Whether to use LoRA. Default: False
            use_vera (bool): whether to use VeRA. Default: False
        """
        super().__init__()

        self.dim = dim
        self.window_size = window_size  # [Wc, Wh, Ww]
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        # Ensure input dimension is divisible by number of heads
        assert dim % num_heads == 0, f"dim ({dim}) should be divisible by num_heads ({num_heads})."

        # Initialize main components
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attention_dropout_rate)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_dropout)

        # Initialize LoRA if enabled
        if use_lora:
            self.peft_proj = LoRARollout(dim, dim, peft_r, lora_alpha, peft_dropout, peft_steps, peft_mode)
            self.peft_qkv = LoRARollout(dim, dim * 3, peft_r, lora_alpha, peft_dropout, peft_steps, peft_mode)
        elif use_vera:
            self.peft_qkv = VeRARollout(
                dim, dim * 3, r=peft_r, dropout=peft_dropout,
                d_initial=d_initial, max_steps=peft_steps, mode=peft_mode
            )
            self.peft_proj = VeRARollout(
                dim, dim, r=peft_r, dropout=peft_dropout,
                d_initial=d_initial, max_steps=peft_steps, mode=peft_mode
            )
        else:
            # Use lambda functions returning 0 for efficiency when LoRA is disabled
            self.peft_proj = lambda *args, **kwargs: 0
            self.peft_qkv = lambda *args, **kwargs: 0

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        rollout_step: int = 0,
    ) -> torch.Tensor:
        """
        Forward pass of the WindowMultiHeadSelfAttention module.

        Args:
            x (torch.Tensor): Input features. Shape: [num_windows*B, N, C]
                where num_windows is the number of windows, B is the batch size,
                N is the number of tokens per window, and C is the input channel dimension.
            mask (torch.Tensor, optional): Attention mask. Shape: [num_windows, Wh*Ww, Wh*Ww]
                Contains floating points in the range [-inf, 0.0]. Default: None
            rollout_step (int): Current rollout step for LoRA. Default: 0

        Returns:
            torch.Tensor: Processed features. Shape: [num_windows*B, N, C]
        """
        B_, N, C = x.shape  # shape: [num_windows*B, N, C]

        # Compute QKV matrices + lora values if enabled
        qkv = self.qkv(x) + self.peft_qkv(x, rollout_step)  # shape: [num_windows*B, N, 3*C]

        # Get ready for splitting into q, k, v
        qkv = qkv.reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(
            2, 0, 3, 1, 4
        )  # shape: [3, num_windows*B, num_heads, N, C//num_heads]

        # Separate Q, K, V from the unified QKV tensor
        q, k, v = qkv.unbind(0)  # Each has shape: [num_windows*B, num_heads, N, C//num_heads]

        # Compute attention scores
        # k is transposed from dimensions [num_windows*B, num_heads, N, C//num_heads] to [num_windows*B, num_heads, C//num_heads, N]
        attention_scores = (q @ k.transpose(-2, -1)) * self.scale  # shape: [num_windows*B, num_heads, N, N]

        # Apply attention mask if provided
        if mask is not None:
            num_windows = mask.shape[0]
            attention_scores = attention_scores.view(B_ // num_windows, num_windows, self.num_heads, N, N)
            attention_scores = attention_scores + mask.unsqueeze(1).unsqueeze(0)
            attention_scores = attention_scores.view(-1, self.num_heads, N, N)

        # Apply softmax to get attention weights
        attention_weights = F.softmax(attention_scores, dim=-1)

        # Apply attention dropout
        attention_weights = self.attn_drop(attention_weights)

        # Compute weighted sum of values
        x = (attention_weights @ v).transpose(1, 2).reshape(B_, N, C)  # shape: [num_windows*B, N, C]

        # Final linear projection and LoRA addition
        x = self.proj(x) + self.peft_proj(x, rollout_step)  # shape: [num_windows*B, N, C]

        # Apply projection dropout
        x = self.proj_drop(x)

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}"


class Swin3DTransformerBlock(nn.Module):
    """3D Swin Transformer block for processing 3D data."""

    def __init__(
        self,
        input_dim: int,
        num_heads: int,
        temporal_dim: int,
        window_size: tuple[int, int, int] = (2, 7, 7),
        shift_size: tuple[int, int, int] = (0, 0, 0),
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        dropout_rate: float = 0.0,
        attention_dropout_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        activation_fn: type = nn.GELU,
        scale_bias: float = 0.0,
        peft_r: int = 8,
        lora_alpha: int = 8,
        d_initial: float = 0.1,
        peft_dropout: float = 0.0,
        peft_steps: int = 40,
        peft_mode: LoRAMode = "single",
        use_lora: bool = False,
        use_vera: bool = False,
    ) -> None:
        """
        Args:
            input_dim (int): Number of input channels.
            num_heads (int): Number of attention heads.
            temporal_dim (int): Dimension of the lead time embedding for temporal information.
            window_size (tuple[int, int, int]): Window size for attention computation. Default: [2, 7, 7]
            shift_size (tuple[int, int, int]): Shift size for SW-MSA. Default: [0, 0, 0]
            mlp_ratio (float): Ratio of MLP hidden dim to embedding dim. Default: 4.0
            qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
            dropout_rate (float): Dropout rate. Default: 0.0
            attention_dropout_rate (float): Attention dropout rate. Default: 0.0
            drop_path_rate (float): Stochastic depth rate. Default: 0.0
            activation_fn (type): Activation layer type. Default: nn.GELU
            scale_bias (float): Scale bias for AdaptiveLayerNorm. Default: 0.0
            peft_r (int): Rank for Parameter Efficient Fine Tuning: 
                a) Low-Rank Adaptation (LoRA). Default: 8 | 
                b) Vector-based Random-matrix Adaptation. Default: 256
            lora_alpha (int): Scaling factor for LoRA. Default: 8
            d_initial (float): Initialization factor for lamda vector. Default: 0.1
            peft_dropout (float): Dropout rate for PEFTs. Default: 0.0
            peft_steps (int): Maximum number of PEFTs roll-out steps. Default: 40
            lora_mode (LoRAMode): Mode for LoRA application. Default: "single" => Same with VeRA
            use_lora (bool): Whether to use LoRA. Default: False
            use_vera (bool): whether to use VeRA. Default: False
        """
        super().__init__()

        self.input_dim = input_dim
        self.window_size = window_size
        self.shift_size = shift_size
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio

        # Adaptive Layer Normalization for improved performance, better than just layer norm trust me
        self.norm1 = AdaptiveLayerNorm(input_dim, temporal_dim, scale_bias=scale_bias)

        # Window Multi-Head Self-Attention
        self.attention = WindowMultiHeadSelfAttention(
            input_dim,
            window_size=self.window_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attention_dropout_rate=attention_dropout_rate,
            proj_dropout=dropout_rate,
            peft_r=peft_r,
            lora_alpha=lora_alpha,
            d_initial=d_initial,
            peft_dropout=peft_dropout,
            peft_steps=peft_steps,
            peft_mode=peft_mode,
            use_lora=use_lora,
            use_vera=use_vera,
        )


        # Stochastic Depth
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()

        # Second Adaptive Layer Normalization
        self.norm2 = AdaptiveLayerNorm(input_dim, temporal_dim, scale_bias=scale_bias)

        # MLP block
        mlp_hidden_dim = int(input_dim * mlp_ratio)
        self.mlp = MLP(
            in_features=input_dim,
            hidden_features=mlp_hidden_dim,
            activation=activation_fn,
            dropout=dropout_rate,
        )

    def forward(
        self,
        input_tensor: torch.Tensor,
        context_tensor: torch.Tensor,
        resolution: tuple[int, int, int],
        rollout_step: int,
        use_warped_attention: bool = True,
    ) -> torch.Tensor:
        """Forward pass of the Swin Transformer block.

        Args:
            input_tensor (torch.Tensor): Input tokens. Shape: [B, L, D]
            context_tensor (torch.Tensor): Conditioning context. Shape: [B, D]
            resolution (tuple[int, int, int]): Resolution of the input `input_tensor`. Shape: [C, H, W]
            rollout_step (int): Current step in the rollout process.
            use_warped_attention (bool): If True, connects left and right sides in shifted window attention. Default: True

        Returns:
            torch.Tensor: Output tokens. Shape: [B, L, D]
        """
        C, H, W = resolution
        B, L, D = input_tensor.shape
        assert L == C * H * W, f"Wrong feature size: {L} vs {C}x{H}x{W}={C*H*W}"

        # Adjust window size if larger than input resolution
        adjusted_window_size, adjusted_shift_size = adjust_windows(self.window_size, self.shift_size, resolution)

        shortcut = input_tensor
        input_tensor = input_tensor.view(B, C, H, W, D)  # shape: [B, C, H, W, D]

        # Perform cyclic shift if shift size is non-zero
        if any(s != 0 for s in adjusted_shift_size):
            shifted_input = torch.roll(
                input_tensor, shifts=(-adjusted_shift_size[0], -adjusted_shift_size[1], -adjusted_shift_size[2]), dims=(1, 2, 3)
            )  # shape: [B, C, H, W, D]
            attention_mask, _ = compute_3d_shifted_window_mask(
                C,
                H,
                W,
                adjusted_window_size,
                adjusted_shift_size,
                input_tensor.device,
                input_tensor.dtype,
                warped=use_warped_attention,
            )
        else:
            shifted_input = input_tensor
            attention_mask = None

        # Pad input to multiple of window size
        pad_size = tuple(-dim % w for dim, w in zip((C, H, W), adjusted_window_size))
        shifted_input = apply_or_remove_3d_padding(shifted_input, pad_size, mode="apply")  # shape: [B, C', H', W', D]

        # Partition patches/tokens into windows
        windows = window_partition_and_reverse_3d(
            shifted_input, adjusted_window_size, mode="partition"
        )  # shape: [nW*B, ws[0], ws[1], ws[2], D]
        windows = windows.view(
            -1, adjusted_window_size[0] * adjusted_window_size[1] * adjusted_window_size[2], D
        )  # shape: [nW*B, ws[0]*ws[1]*ws[2], D]

        # Apply Window Multi-Head Self-Attention
        attention_windows = self.attention(
            windows, mask=attention_mask, rollout_step=rollout_step
        )  # shape: [nW*B, ws[0]*ws[1]*ws[2], D]

        # Merge windows back to original resolution
        attention_windows = attention_windows.view(
            -1, adjusted_window_size[0], adjusted_window_size[1], adjusted_window_size[2], D
        )  # shape: [nW*B, ws[0], ws[1], ws[2], D]
        shifted_input = window_partition_and_reverse_3d(
            attention_windows, adjusted_window_size, mode="reverse", total_channels=C, total_height=H, total_width=W
        )  # shape: [B, C', H', W', D]

        # Remove padding
        shifted_input = apply_or_remove_3d_padding(shifted_input, pad_size, mode="remove")  # shape: [B, C, H, W, D]

        # Reverse cyclic shift if applied
        if any(s != 0 for s in adjusted_shift_size):
            output_tensor = torch.roll(
                shifted_input, shifts=(adjusted_shift_size[0], adjusted_shift_size[1], adjusted_shift_size[2]), dims=(1, 2, 3)
            )  # shape: [B, C, H, W, D]
        else:
            output_tensor = shifted_input

        output_tensor = output_tensor.reshape(B, C * H * W, D)  # shape: [B, L, D]

        # Apply first normalization, attention, and residual connection
        output_tensor = shortcut + self.drop_path(self.norm1(output_tensor, context_tensor))

        # Apply second normalization, MLP, and residual connection
        output_tensor = output_tensor + self.drop_path(self.norm2(self.mlp(output_tensor), context_tensor))

        return output_tensor  # shape: [B, L, D]


class PatchMerging3D(nn.Module):
    """
    This layer reduces spatial dimensions by 2 in height and witdh,
    while doubling the channel dimension.
    """

    def __init__(self, input_dim: int) -> None:
        """
        Initialize the PatchMerging3D layer.

        Args:
            dim (int): Number of input channels.
        """
        super().__init__()
        self.input_dim = input_dim
        self.reduction = nn.Linear(4 * input_dim, 2 * input_dim, bias=False)
        self.norm = nn.LayerNorm(4 * input_dim)

    def _merge_patches(self, x: torch.Tensor, res: tuple[int, int, int]) -> torch.Tensor:
        """
        Merge patches in the spatial dimensions.

        Args:
            x (torch.Tensor): Input tensor. Shape: [B, L, D]
            res (tuple[int, int, int]): Resolution of input. (C, H, W)

        Returns:
            torch.Tensor: Merged patches. Shape: [B, L/4, 4D]
        """
        C, H, W = res
        B, L, D = x.shape
        assert L == C * H * W, f"Input shape mismatch: {L} != {C}*{H}*{W}"
        # assert H > 1 and W > 1, f"Height ({H}) and Width ({W}) must be larger than 1"
        # Prevent merging if H or W is less than or equal to 1
        if H <= 1 or W <= 1:
            return x  # Skip merging if dimensions are too small

        # Reshape input to [B, C, H, W, D]
        x = x.view(B, C, H, W, D)

        # Apply custom padding to ensure height and width are even
        x = apply_or_remove_3d_padding(x, (0, H % 2, W % 2), mode="apply")  # shape: [B, C, H', W', D]

        new_H, new_W = x.shape[2], x.shape[3]
        assert new_H % 2 == 0 and new_W % 2 == 0, f"Padded dimensions must be even: H={new_H}, W={new_W}"

        # Merge patches using reshape and rearrange operations
        x = x.reshape(B, C, new_H // 2, 2, new_W // 2, 2, D)
        x = rearrange(x, "B C H h W w D -> B (C H W) (h w D)")  # shape: [B, C*(H//2)*(W//2), 4D]

        return x

    def forward(self, x: torch.Tensor, input_resolution: tuple[int, int, int]) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tokens. Shape: [B, C*H*W, D]
            input_resolution (tuple[int, int, int]): Resolution of input. (C, H, W)

        Returns:
            torch.Tensor: Merged tokens. Shape: [B, C*(H//2)*(W//2), 2D]
        """
        # Merge patches
        merged_patches = self._merge_patches(x, input_resolution)  # shape: [B, L/4, 4D]

        # Apply layer normalization
        normalized_patches = self.norm(merged_patches)  # shape: [B, L/4, 4D]

        # Reduce channel dimension
        reduced_patches = self.reduction(normalized_patches)  # shape: [B, L/4, 2D]

        return reduced_patches


class PatchSplitting3D(nn.Module):
    """Patch spliting layer for 3D data."""

    def __init__(self, input_dim: int) -> None:
        """
        Initialize the PatchSplitting3D module.

        Args:
            dim (int): Number of input channels.
        """
        super().__init__()
        self.input_dim = input_dim
        assert input_dim % 2 == 0, f"dim ({input_dim}) should be divisible by 2."
        self.expand = nn.Linear(input_dim, input_dim * 2, bias=False)  # Expand channels
        self.reduce = nn.Linear(input_dim // 2, input_dim // 2, bias=False)  # Reduce channels after splitting
        self.norm = nn.LayerNorm(input_dim // 2)

    def _split_patches(
        self,
        x: torch.Tensor,
        res: tuple[int, int, int],
        crop: tuple[int, int, int],
    ) -> torch.Tensor:
        """
        Split patches in the spatial dimensions.

        Args:
            x (torch.Tensor): Input tensor. Shape: [B, L, D*2]
            res (tuple[int, int, int]): Resolution of input. (C, H, W)
            crop (tuple[int, int, int]): Cropping for every dimension.

        Returns:
            torch.Tensor: Split patches. Shape: [B, C*4H*4W, D//4]
        """
        C, H, W = res
        B, L, D = x.shape
        assert L == C * H * W, f"Input shape mismatch: {L} != {C}*{H}*{W}"
        assert D % 8 == 0, f"Number of input features ({D}) is not a multiple of 8."

        # Reshape and rearrange to split patches
        x = x.view(B, C, H, W, 2, 2, D // 4)
        x = rearrange(x, "B C H W h w D -> B C (H h) (W w) D")  # shape: [B, C, 2H, 2W, D//4]

        # Remove padding if any
        x = apply_or_remove_3d_padding(x, crop, mode="remove")

        return x.reshape(B, -1, D // 4)  # shape: [B, C*4H*4W, D//4]

    def forward(
        self,
        x: torch.Tensor,
        input_resolution: tuple[int, int, int],
        crop: tuple[int, int, int] = (0, 0, 0),
    ) -> torch.Tensor:
        """
        Perform the patch splitting.

        Quadruples the number of patches by doubling in the H and W dimensions.

        Args:
            x (torch.Tensor): Input tokens. Shape: [B, C*H*W, D]
            input_resolution (tuple[int, int, int]): Resolution of x. (C, H, W)
            crop (tuple[int, int, int]): Cropping for every dimension. Default: (0, 0, 0)

        Returns:
            torch.Tensor: Split tokens. Shape: [B, C*4H*4W, D//2]
        """
        # Expand channel dimension
        x = self.expand(x)  # shape: [B, C*H*W, D*2]

        # Split patches
        x = self._split_patches(x, input_resolution, crop)  # shape: [B, C*4H*4W, D//4]

        # Normalize
        x = self.norm(x)

        # Reduce channel dimension
        x = self.reduce(x)  # shape: [B, C*4H*4W, D//2]

        return x


class SwinTransformer3DLayer(nn.Module):
    """A 3D Swin Transformer layer for one stage, capable of processing 3D data with multi-head self-attention and optional downsampling or upsampling."""

    def __init__(
        self,
        input_dim: int,
        depth: int,
        num_heads: int,
        window_size: tuple[int, int, int],
        temporal_dim: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        dropout: float = 0.0,
        attention_dropout_rate: float = 0.0,
        drop_path: float | list[float] = 0.0,
        downsample: type[PatchMerging3D] | None = None,
        upsample: type[PatchSplitting3D] | None = None,
        scale_bias: float = 0.0,
        peft_r: int = 8,
        lora_alpha: int = 8,
        d_initial: float = 0.1,
        peft_dropout: float = 0.0,
        peft_steps: int = 40,
        peft_mode: LoRAMode = "single",
        use_lora: bool = False,
        use_vera: bool = False,
    ) -> None:
        """
        Args:
            input_dim (int): Number of input channels.
            depth (int): Number of Transformer blocks in this layer.
            num_heads (int): Number of attention heads in each Transformer block.
            window_size (tuple[int, int, int]): Size of attention window for each dimension.
            temporal_dim (int): Dimension of the lead time embedding.
            mlp_ratio (float): Ratio of MLP hidden dim to embedding dim. Default: 4.0
            qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
            dropout (float): Dropout rate. Default: 0.0
            attention_dropout_rate (float): Attention dropout rate. Default: 0.0
            drop_path (float | list[float]): Stochastic depth rate. Default: 0.0
            downsample (PatchMerging3D | None): Downsampling layer at the end of the layer. Default: None
            upsample (PatchSplitting3D | None): Upsampling layer at the end of the layer. Default: None
            scale_bias (float): Scale bias for AdaptiveLayerNorm. Default: 0.0
            lora_steps (int): Maximum number of LoRA roll-out steps. Default: 40
            lora_mode (LoRAMode): LoRA mode. 'single' uses same LoRA for all steps, 'all' uses different LoRA per step. Default: "single"
            use_lora (bool): Whether to use LoRA. Default: False
        """
        super().__init__()

        if downsample is not None and upsample is not None:
            raise ValueError("Cannot set both `downsample` and `upsample`.")

        self.input_dim = input_dim
        self.depth = depth

        # Create a list of Swin Transformer blocks
        self.blocks = nn.ModuleList(
            [
                Swin3DTransformerBlock(
                    input_dim=input_dim,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=self._get_shift_size(i, window_size),
                    temporal_dim=temporal_dim,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    dropout_rate=dropout,
                    attention_dropout_rate=attention_dropout_rate,
                    drop_path_rate=self._get_drop_path(i, drop_path),
                    scale_bias=scale_bias,
                    peft_r=peft_r,
                    lora_alpha=lora_alpha,
                    d_initial=d_initial,
                    peft_dropout=peft_dropout,
                    peft_steps=peft_steps,
                    peft_mode=peft_mode,
                    use_lora=use_lora,
                    use_vera=use_vera,
                )
                for i in range(depth)
            ]
        )

        # Set up downsampling or upsampling layer if specified
        self.downsample = downsample(input_dim=input_dim) if downsample is not None else None
        self.upsample = upsample(input_dim=input_dim) if upsample is not None else None

    def _get_shift_size(self, block_index: int, window_size: tuple[int, int, int]) -> tuple[int, int, int]:
        """Determine the shift size for the current block.

        Args:
            block_index (int): Index of the current block.
            window_size (tuple[int, int, int]): Size of the attention window.

        Returns:
            tuple[int, int, int]: Shift size for each dimension.
        """
        # Alternate between no shift and half window shift
        return (0, 0, 0) if block_index % 2 == 0 else tuple(ws // 2 for ws in window_size)

    def _get_drop_path(self, block_index: int, drop_path: float | list[float]) -> float:
        """Get the drop path rate for the current block.

        Args:
            block_index (int): Index of the current block.
            drop_path (float | list[float]): Drop path rate(s).

        Returns:
            float: Drop path rate for the current block.
        """
        # If drop_path is a list, use the corresponding value; otherwise, use the single value
        return drop_path[block_index] if isinstance(drop_path, list) else drop_path

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        res: tuple[int, int, int],
        crop: tuple[int, int, int] = (0, 0, 0),
        rollout_step: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Forward pass!

        Args:
            x (torch.Tensor): Input tokens. Shape: [B, L, D]
            c (torch.Tensor): Conditioning context. Shape: [B, D]
            res (tuple[int, int, int]): Resolution of the input x.
            crop (tuple[int, int, int]): Cropping for every dimension. Default: [0, 0, 0]
            rollout_step (int): Current roll-out step. Default: 0

        Returns:
            tuple[torch.Tensor, torch.Tensor | None]: Processed tokens and optionally the input before down/upsampling.
        """
        # Process input through each Transformer block
        for block in self.blocks:
            x = block(x, c, res, rollout_step)  # shape: [B, L, D]

        # Store the input before scaling for skip connections
        x_before_scaling = x

        # Apply downsampling if specified
        if self.downsample is not None:
            x_scaled = self.downsample(x, res)  # shape: [B, L/8, D*2]
            return x_scaled, x_before_scaling

        # Apply upsampling if specified
        if self.upsample is not None:
            x_scaled = self.upsample(x, res, crop)  # shape: [B, L*8, D//2]
            return x_scaled, x_before_scaling

        # If no scaling, return the processed tokens and None
        return x, None

    def init_respostnorm(self):
        """Initialize the post-normalization layers in the residual connections of the windowed attention mechanism."""
        for block in self.blocks:
            block.norm1.init_weights()
            block.norm2.init_weights()
