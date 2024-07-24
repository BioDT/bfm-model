from typing import Optional

import torch
import torch.nn as nn
from einops import rearrange, repeat

from src.perceiver_components.helpers_io import (
    Attention,
    FeedForward,
    PreNorm,
    cache_fn,
    dropout_seq,
)
from src.perceiver_components.pos_encoder import build_position_encoding


class PerceiverIO(nn.Module):
    def __init__(
        self,
        *,
        num_layers,
        dim,
        queries_dim,
        logits_dimension=None,
        num_latent_tokens=512,
        latent_dimension=512,
        cross_attention_heads=1,
        latent_attention_heads=8,
        cross_attention_head_dim=64,
        latent_attention_head_dim=64,
        weight_tie_layers=False,
        decoder_feedforward=False,
        sequence_dropout_prob=0.0,
        num_fourier_bands=64,
        max_frequency=224,
        num_input_axes=2,
        position_encoding_type=None,
    ):
        super().__init__()
        self.sequence_dropout_prob = sequence_dropout_prob
        self.num_input_axes = num_input_axes
        self.max_frequency = max_frequency
        self.num_fourier_bands = num_fourier_bands
        self.position_encoding_type = position_encoding_type

        self.fourier_channels = self._calculate_fourier_channels()
        self.total_input_dim = dim + self.fourier_channels

        self.latents = nn.Parameter(torch.randn(num_latent_tokens, latent_dimension))

        self.cross_attend_blocks = self._build_cross_attention_blocks(
            latent_dimension, cross_attention_heads, cross_attention_head_dim
        )
        self.layers = self._build_latent_transformer(
            num_layers, latent_dimension, latent_attention_heads, latent_attention_head_dim, weight_tie_layers
        )

        self.decoder_cross_attn = self._build_decoder_cross_attention(
            queries_dim, latent_dimension, cross_attention_heads, cross_attention_head_dim
        )
        self.decoder_feedforward = self._build_decoder_feedforward(queries_dim) if decoder_feedforward else None

        self.to_logits = nn.Linear(queries_dim, logits_dimension) if logits_dimension is not None else nn.Identity()

    def _calculate_fourier_channels(self):
        # TODO: Make it more adapative for various position encoding types (+ various parameters for each type)
        return (self.num_input_axes * ((self.num_fourier_bands * 2) + 1)) if self.position_encoding_type == "fourier" else 0

    def _build_cross_attention_blocks(self, latent_dimension, cross_attention_heads, cross_attention_head_dim):
        return nn.ModuleList(
            [
                PreNorm(
                    latent_dimension,
                    Attention(
                        latent_dimension, self.total_input_dim, heads=cross_attention_heads, head_dim=cross_attention_head_dim
                    ),
                    context_dimension=self.total_input_dim,
                ),
                PreNorm(latent_dimension, FeedForward(latent_dimension)),
            ]
        )

    @cache_fn
    def _get_latent_attention(self, latent_dimension, latent_attention_heads, latent_attention_head_dim, **kwargs):
        return PreNorm(
            latent_dimension, Attention(latent_dimension, heads=latent_attention_heads, head_dim=latent_attention_head_dim)
        )

    @cache_fn
    def _get_latent_feedforward(self, latent_dimension, **kwargs):
        return PreNorm(latent_dimension, FeedForward(latent_dimension))

    def _build_latent_transformer(
        self, num_layers, latent_dimension, latent_attention_heads, latent_attention_head_dim, weight_tie_layers
    ):
        cache_args = {"_cache": weight_tie_layers}

        layers = nn.ModuleList([])

        for _ in range(num_layers):
            latent_attn = self._get_latent_attention(
                latent_dimension, latent_attention_heads, latent_attention_head_dim, **cache_args
            )
            latent_ff = self._get_latent_feedforward(latent_dimension, **cache_args)

            layers.append(nn.ModuleList([latent_attn, latent_ff]))

        return layers

    def _build_decoder_cross_attention(self, queries_dim, latent_dimension, cross_attention_heads, cross_attention_head_dim):
        return PreNorm(
            queries_dim,
            Attention(queries_dim, latent_dimension, heads=cross_attention_heads, head_dim=cross_attention_head_dim),
            context_dimension=latent_dimension,
        )

    def _build_decoder_feedforward(self, queries_dim):
        return PreNorm(queries_dim, FeedForward(queries_dim))

    def _build_position_encoding(self, shape):
        if self.position_encoding_type is None:
            return None
        elif self.position_encoding_type == "fourier":
            return build_position_encoding(
                position_encoding_type=self.position_encoding_type,
                index_dims=shape[1:-1],
                fourier_position_encoding_kwargs={
                    "num_bands": self.num_fourier_bands,
                    "max_freq": self.max_frequency,
                    "concat_pos": True,
                    "sine_only": False,
                },
            )

    def _apply_position_encoding(self, input_data):
        batch_size, *_, _, device = *input_data.shape, input_data.device
        if self.position_encoding_type == "fourier":
            pos_encoder = self._build_position_encoding(input_data.shape)
            pos_encoding = pos_encoder(batch_size=batch_size).to(device)
            input_data = torch.cat((input_data, pos_encoding), dim=-1)
        return input_data

    def _process_latents(self, latent_representation, flattened_input, attention_mask):
        cross_attention, cross_feedforward = self.cross_attend_blocks

        latent_representation = (
            cross_attention(latent_representation, context=flattened_input, mask=attention_mask) + latent_representation
        )
        latent_representation = cross_feedforward(latent_representation) + latent_representation

        for self_attention, self_feedforward in self.layers:
            latent_representation = self_attention(latent_representation) + latent_representation
            latent_representation = self_feedforward(latent_representation) + latent_representation

        return latent_representation

    def _decode(self, latent_representation, queries):
        decoded_output = self.decoder_cross_attn(queries, context=latent_representation)

        if self.decoder_feedforward is not None:
            decoded_output = decoded_output + self.decoder_feedforward(decoded_output)

        return self.to_logits(decoded_output)

    def forward(
        self, input_data: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, queries: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch_size, *axis, _ = input_data.shape
        assert len(axis) == self.num_input_axes, f"Input data must have {self.num_input_axes} axes, got {len(axis)}"

        input_data = self._apply_position_encoding(input_data)
        flattened_input = rearrange(input_data, "b ... d -> b (...) d")

        latent_representation = repeat(self.latents, "n d -> b n d", b=batch_size)

        if self.training and self.sequence_dropout_prob > 0.0:
            flattened_input, attention_mask = dropout_seq(flattened_input, attention_mask, self.sequence_dropout_prob)

        latent_representation = self._process_latents(latent_representation, flattened_input, attention_mask)

        if queries is None:
            return latent_representation

        if queries.ndim == 2:
            queries = repeat(queries, "n d -> b n d", b=batch_size)

        return self._decode(latent_representation, queries)


if __name__ == "__main__":
    # Test case 1: Language-like data without positional encoding
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_lang = PerceiverIO(
        num_layers=6,
        dim=512,
        queries_dim=128,
        logits_dimension=1000,
        num_latent_tokens=256,
        latent_dimension=512,
        cross_attention_heads=1,
        latent_attention_heads=8,
        cross_attention_head_dim=64,
        latent_attention_head_dim=64,
        weight_tie_layers=False,
        decoder_feedforward=True,
        sequence_dropout_prob=0.1,
        num_input_axes=1,
        position_encoding_type=None,  # no positional encoding
    )

    model_lang.to(device)

    # dummy language-like input data
    batch_size = 32
    seq_len = 1024
    input_dim = 512

    data_lang = torch.randn(batch_size, seq_len, input_dim).to(device)
    mask_lang = torch.ones(batch_size, seq_len).bool().to(device)
    queries_lang = torch.randn(batch_size, 100, 128).to(device)

    # forward pass for language-like data
    output_lang = model_lang(data_lang, attention_mask=mask_lang, queries=queries_lang)
    print(f"Language-like output shape (no positional encoding): {output_lang.shape}")

    # Test case 2: Image-like data with positional encoding
    model_img = PerceiverIO(
        num_layers=6,
        dim=3,  # Input dimension for image data (RGB channels)
        queries_dim=256,
        logits_dimension=1000,
        num_latent_tokens=256,
        latent_dimension=512,
        cross_attention_heads=1,
        latent_attention_heads=8,
        cross_attention_head_dim=64,
        latent_attention_head_dim=64,
        weight_tie_layers=False,
        decoder_feedforward=True,
        sequence_dropout_prob=0.1,
        num_fourier_bands=64,
        max_frequency=224,
        num_input_axes=2,
        position_encoding_type="fourier",
    )

    #  dummy image-like input data
    batch_size = 32
    height, width = 224, 224
    channels = 3

    data_img = torch.randn(batch_size, height, width, channels)
    mask_img = torch.ones(batch_size, height * width).bool()
    queries_img = torch.randn(batch_size, 100, 256)

    # forward pass for image-like data
    output_img = model_img(data_img, attention_mask=mask_img, queries=queries_img)
    print(f"Image-like output shape (with positional encoding): {output_img.shape}")

    # Test case 3: Language-like data with positional encoding
    model_lang_pos = PerceiverIO(
        num_layers=6,
        dim=512,
        queries_dim=128,
        logits_dimension=1000,
        num_latent_tokens=256,
        latent_dimension=512,
        cross_attention_heads=1,
        latent_attention_heads=8,
        cross_attention_head_dim=64,
        latent_attention_head_dim=64,
        weight_tie_layers=False,
        decoder_feedforward=True,
        sequence_dropout_prob=0.1,
        num_fourier_bands=64,
        max_frequency=10000,
        num_input_axes=1,
        position_encoding_type="fourier",
    ).to(device)

    # forward pass for language-like data with positional encoding
    output_lang_pos = model_lang_pos(data_lang, attention_mask=mask_lang, queries=queries_lang)
    print(f"Language-like output shape (with positional encoding): {output_lang_pos.shape}")

    # just checking out the number of parameters for each of the models
    print("\nModel parameters:")
    print(f"Language model (no pos encoding): {sum(p.numel() for p in model_lang.parameters()):,}")
    print(f"Image model (with pos encoding): {sum(p.numel() for p in model_img.parameters()):,}")
    print(f"Language model (with pos encoding): {sum(p.numel() for p in model_lang_pos.parameters()):,}")
