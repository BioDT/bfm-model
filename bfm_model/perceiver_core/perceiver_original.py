import weakref
from collections import OrderedDict
from functools import wraps
from typing import Any, Callable, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Reduce

from bfm_model.perceiver_components.helpers import (
    Attention,
    FeedForward,
    PreNorm,
    cache_fn,
)
from bfm_model.perceiver_components.pos_encoder import build_position_encoding


class Perceiver(nn.Module):
    """
    Perceiver: A General Architecture for Structured Inputs to Predictions.

    This implementation is based on the paper "Perceiver: General Perception with Iterative Attention"
    by Jaegle et al. (2021). The Perceiver uses a combination of cross-attention and self-attention
    mechanisms to process high-dimensional inputs into a latent space, and then make predictions.
    """

    def __init__(
        self,
        num_fourier_bands: int,
        num_layers: int,
        max_frequency: float,
        input_channels: int = 3,
        num_input_axes: int = 2,
        num_latent_tokens: int = 512,
        latent_dimension: int = 512,
        cross_attention_heads: int = 1,
        self_attention_heads: int = 8,
        cross_attention_head_dim: int = 64,
        self_attention_head_dim: int = 64,
        num_classes: int = 1000,
        attention_dropout: float = 0.0,
        feedforward_dropout: float = 0.0,
        weight_tie_layers: bool = False,
        use_fourier_encoding: bool = True,
        self_attentions_per_cross: int = 1,
        include_classifier_head: bool = True,
    ):
        """
        Initializes the Perceiver model.

        Args:
        num_fourier_bands: Number of frequency bands for Fourier positional feature encodings
        num_layers: Number of cross-attention/self-attention layer blocks
        max_frequency: Maximum frequency for Fourier feature encoding
        input_channels: Number of channels in the input data (default: 3 for RGB images)
        num_input_axes: Number of axes in the input data (e.g., 2 for images, 3 for video)
        num_latent_tokens: Number of latent tokens (called latents/centroids etc.) to use
        latent_dimension: Dimensionality of latent tokens
        cross_attention_heads: Number of heads in cross-attention layers
        self_attention_heads: Number of heads in latent self-attention layers
        cross_attention_head_dim: Dimension of each cross-attention head
        self_attention_head_dim: Dimension of each latent self-attention head
        num_classes: Number of output classes for classification
        attention_dropout: Dropout rate for attention layers
        feedforward_dropout: Dropout rate for feedforward layers
        weight_tie_layers: Whether to tie weights across layers (optional)
        use_fourier_encoding: Whether to use Fourier encoding for input data, using the given input axes
        self_attentions_per_cross: Number of self-attention blocks per cross-attention
        include_classifier_head: Whether to include a final classification head
        """
        super().__init__()
        # Input-related parameters
        self.num_input_axes = num_input_axes
        self.max_frequency = max_frequency
        self.num_fourier_bands = num_fourier_bands
        self.use_fourier_encoding = use_fourier_encoding

        # Calculate total input dimension including Fourier features
        fourier_channels = self._calculate_fourier_channels()
        # TODO: Add extra 'if's, since fourier encodings could be e.g., only sinusoidal, not-concatenated with original positions etc.
        total_input_dim = fourier_channels + input_channels

        # Initialize latent tokens (what makes the Perceiver special)
        self.latent_tokens = nn.Parameter(torch.randn(num_latent_tokens, latent_dimension))

        # Store model dimensions and hyperparameters
        self.latent_dimension = latent_dimension
        self.total_input_dim = total_input_dim
        self.cross_attention_heads = cross_attention_heads
        self.cross_attention_head_dim = cross_attention_head_dim
        self.attention_dropout = attention_dropout
        self.feedforward_dropout = feedforward_dropout
        self.self_attention_heads = self_attention_heads
        self.self_attention_head_dim = self_attention_head_dim

        # Build the main layers of the model
        self.layers = self._build_layers(num_layers, self_attentions_per_cross, weight_tie_layers)
        # Build the classifier head if desired
        self.classifier = self._build_classifier(latent_dimension, num_classes) if include_classifier_head else nn.Identity()

    def _calculate_fourier_channels(self):
        """
        Calculates the number of Fourier channels based on the number of bands and input axes.

        Returns:
            Number of Fourier channels
        """
        # TODO: Make it more adapative for various position encoding types (+ various parameters for each type)
        return (self.num_input_axes * ((self.num_fourier_bands * 2) + 1)) if self.use_fourier_encoding else 0

    @cache_fn
    def _get_cross_attention(self):
        """
        Creates a cross-attention layer with normalization.

        Returns:
            PreNorm wrapped cross-attention layer
        """
        return PreNorm(
            self.latent_dimension,
            Attention(
                self.latent_dimension,
                self.total_input_dim,
                heads=self.cross_attention_heads,
                head_dim=self.cross_attention_head_dim,
                dropout=self.attention_dropout,
            ),
            context_dimension=self.total_input_dim,
        )

    @cache_fn
    def _get_cross_feedforward(self):
        """
        Creates a feedforward layer for cross-attention with normalization.

        Returns:
            PreNorm wrapped feedforward layer
        """
        return PreNorm(self.latent_dimension, FeedForward(self.latent_dimension, dropout=self.feedforward_dropout))

    @cache_fn
    def _get_self_attention(self):
        """
        Creates a self-attention layer with normalization.

        Returns:
            PreNorm wrapped self-attention layer
        """
        return PreNorm(
            self.latent_dimension,
            Attention(
                self.latent_dimension,
                heads=self.self_attention_heads,
                head_dim=self.self_attention_head_dim,
                dropout=self.attention_dropout,
            ),
        )

    @cache_fn
    def _get_self_feedforward(self):
        """
        Creates a feedforward layer for self-attention with normalization.

        Returns:
            PreNorm wrapped feedforward layer
        """
        return PreNorm(self.latent_dimension, FeedForward(self.latent_dimension, dropout=self.feedforward_dropout))

    def _build_layers(self, num_layers: int, self_attentions_per_cross: int, weight_tie_layers: bool):
        """
        Builds the main layers of the Perceiver model.

        Args:
            num_layers: Number of cross-attention/self-attention layer blocks
            self_attentions_per_cross: Number of self-attention blocks per cross-attention
            weight_tie_layers: Whether to tie weights across layers

        Returns:
            ModuleList of model layers
        """
        layers = nn.ModuleList([])
        for i in range(num_layers):
            should_cache = i > 0 and weight_tie_layers
            cache_args = {"_cache": should_cache}

            self_attention_blocks = nn.ModuleList([])

            for block_index in range(self_attentions_per_cross):
                self_attention_blocks.append(
                    nn.ModuleList(
                        [
                            self._get_self_attention(**cache_args, key=block_index),
                            self._get_self_feedforward(**cache_args, key=block_index),
                        ]
                    )
                )

            layers.append(
                nn.ModuleList(
                    [self._get_cross_attention(**cache_args), self._get_cross_feedforward(**cache_args), self_attention_blocks]
                )
            )
        return layers

    def _build_classifier(self, latent_dimension: int, num_classes: int):
        """
        Builds the final classifier head.

        Args:
            latent_dimension: Dimensionality of latent tokens
            num_classes: Number of output classes

        Returns:
            Sequential model for classification
        """
        return nn.Sequential(
            Reduce("b n d -> b d", "mean"), nn.LayerNorm(latent_dimension), nn.Linear(latent_dimension, num_classes)
        )

    def _apply_fourier_encode(self, input_data: torch.Tensor, concat_pos: bool = True, sine_only: bool = False) -> torch.Tensor:
        """
        Applies Fourier encoding to the input data.

        Args:
            input_data: Input tensor
            concat_pos: Whether to concatenate the original positions with Fourier features
            sine_only: Whether to use only sine Fourier features

        Returns:
            Fourier encoded input tensor
        """
        if self.use_fourier_encoding:
            batch_size, *axes, _, device, dtype = *input_data.shape, input_data.device, input_data.dtype  # noqa
            assert (
                len(axes) == self.num_input_axes
            ), "Declared number of axes of the data is not equal to the actual number of axes present in the input data."

            # Dynamically create position encoder with correct index_dims
            position_encoder = build_position_encoding(
                position_encoding_type="fourier",
                index_dims=tuple(axes),
                fourier_position_encoding_kwargs={
                    "num_bands": self.num_fourier_bands,
                    "max_freq": self.max_frequency,
                    "concat_pos": concat_pos,
                    "sine_only": sine_only,
                },
            )

            # Generate encoded positions
            encoded_positions = position_encoder(batch_size=batch_size).to(device)

            # Concatenate encoded positions with input data
            input_data = torch.cat((input_data, encoded_positions), dim=-1)

        return input_data

    def forward(
        self, input_data: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, return_embeddings: bool = False
    ) -> torch.Tensor:
        """
        Forward pass of the Perceiver model.

        Args:
            input_data: Input tensor
            attention_mask: Optional attention mask
            return_embeddings: Whether to return embeddings instead of class logits

        Returns:
            Output tensor (logits or embeddings)
        """
        # Apply Fourier encoding to input data (if enabled)
        encoded_input = self._apply_fourier_encode(input_data)

        # Flatten input axes
        flattened_input = rearrange(encoded_input, "b ... d -> b (...) d")

        # Repeat latent tokens for each item in the batch
        latent_representation = repeat(self.latent_tokens, "n d -> b n d", b=flattened_input.shape[0])

        # Process through layers
        for cross_attention, cross_feedforward, self_attention_blocks in self.layers:
            # apply cross-attention
            latent_representation = (
                cross_attention(latent_representation, context=flattened_input, mask=attention_mask) + latent_representation
            )
            latent_representation = cross_feedforward(latent_representation) + latent_representation

            # apply self-attention
            for self_attention, self_feedforward in self_attention_blocks:
                latent_representation = self_attention(latent_representation) + latent_representation
                latent_representation = self_feedforward(latent_representation) + latent_representation

        if return_embeddings:
            return latent_representation

        # Return logits from the classifier head, if we don't want just embeddings
        return self.classifier(latent_representation)


def main():
    # Example usage:#
    ################
    model = Perceiver(
        num_fourier_bands=6,
        num_layers=2,
        max_frequency=10.0,
        input_channels=3,
        num_input_axes=2,
        num_latent_tokens=256,
        latent_dimension=512,
        num_classes=1000,
    )

    # Generate a random batch of 4 224x224 RGB images
    batch_size = 4
    input_data = torch.randn(batch_size, 224, 224, 3)

    # Forward pass
    output = model(input_data)

    print(f"Input shape: {input_data.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output: {output}")


if __name__ == "__main__":
    main()

# TODO:
# 1) Improve flexibility with regards to applying positional encdoigns and passing parameters for these
# 2) (For later) Warmup learning rate scheduler?
