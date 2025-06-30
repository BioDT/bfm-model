"""
Copyright 2025 (C) TNO. Licensed under the MIT license.
"""

import torch
from torch import nn


class AdaptiveLayerNorm(nn.Module):
    """
    Adaptive layer normalisation with scale and shift modulation.
    Basically, given a token (i.e. a vector) and some context (i.e. a vector),
    adjust the scale and shift of the token using the context.
    """

    def __init__(self, dim: int, context_dim: int, scale_bias: float = 0, activation: nn.Module = nn.SiLU()) -> None:
        """
        Args:
            dim (int): Dimension of the input.
            context_dim (int): Dimension of the conditioning signal.
            scale_bias (float, optional): Scale bias to add to the scaling factor. Default: 0.
        """
        super().__init__()

        self.layer_norm = nn.LayerNorm(dim, elementwise_affine=False)
        # using SiLU as the activation function because it is always positive,
        # the output dimension is dim * 2, because we need to output a shift and a scale
        self.ln_modulation = nn.Sequential(activation, nn.Linear(context_dim, dim * 2))
        self.scale_bias = (
            scale_bias  # scale bias is a float number, to be added to the scale, to ensure that the scale is always positive
        )
        self.init_weights()

    def init_weights(self) -> None:
        nn.init.zeros_(self.ln_modulation[-1].weight)  # initialize the weight to 0, using the last layer of the ln_modulation
        nn.init.zeros_(self.ln_modulation[-1].bias)  # initialize the bias also to 0

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor. Shape [B, L, D].
            c (torch.Tensor): Conditioning tensor. Shape [B, D].

        Returns:
            torch.Tensor: Output tensor. Shape [B, L, D].
        """
        # Generate modulation parameters from the condiitoning tensor (which is [B, D])
        modulation = self.ln_modulation(c)  # shape: [B, D*2]

        # Reshape modulation parameters to match input tensor dimensions (which is [B, L, D])
        modulation = modulation.unsqueeze(1)  # shape: [B, 1, D*2]

        # Split modulation into shift and scale
        shift, scale = modulation.chunk(2, dim=-1)  # each has shape: [B, 1, D]
        # Apply layer normalization to the input tensor
        normal_x = self.layer_norm(x)  # shape: [B, L, D]

        # Apply modulation (scale and shift) to the normalized input
        # add the scale bias to ensure scale is always positive
        modulated_x = normal_x * (self.scale_bias + scale) + shift

        return modulated_x
