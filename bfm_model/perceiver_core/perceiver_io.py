from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from einops import rearrange, repeat

from bfm_model.perceiver_components.helpers_io import (
    Attention,
    BuiltinGQAttention,
    FeedForward,
    GQAttention,
    PreNorm,
    cache_fn,
    dropout_seq,
)
from bfm_model.perceiver_components.pos_encoder import build_position_encoding


class PerceiverIO(nn.Module):
    """
    A general architecture for processing structured inputs and outputs.

    This model uses a latent space to process inputs of various modalities and sizes,
    and can generate outputs of various structures through a flexible querying mechanism
    """

    def __init__(
        self,
        *,
        num_layers: int,
        dim: int,
        queries_dim: int,
        logits_dimension: Optional[int] = None,
        num_latent_tokens: int = 512,
        latent_dimension: int = 512,
        cross_attention_heads: int = 16,
        latent_attention_heads: int = 16,
        cross_attention_head_dim: int = 64,
        latent_attention_head_dim: int = 64,
        num_kv_heads: int = 8,
        weight_tie_layers: bool = False,
        decoder_feedforward: bool = False,
        sequence_dropout_prob: float = 0.0,
        num_fourier_bands: int = 64,
        max_frequency: int = 224,
        num_input_axes: int = 2,
        position_encoding_type: bool = "fourier",
        trainable_position_encoding_kwargs: Optional[dict] = None,
    ):
        """
        Initialize the PerceiverIO model.

        Args:
            num_layers (int): Number of self-attention layers in the latent transformer.
                              Each layer consists of a self-attention operation followed by a feedforward NN.

            dim (int): Dimension of the input features.
                       (e.g., for RGB images, this would be 3, for text, this would be the embedding dimension of the words, such as 512).

            queries_dim (int): Dimension of the decoder queries. Determines the size of the features used to quey the latent representation during decoding.

            logits_dimension (Optional[int]): Dimension of the output logits. For classification tasks, this would be the number of classes.
                                              If None, no final projection is applied and the output is the latent representation that can be combined with the custom query.

            num_latent_tokens (int): Number of latent tokens used in the model.
                                     Determines the size of the latent array that processes the input data.

            latent_dimension (int): Dimension of each latent token. Along with `nym_latent_tokens`, determines the overall capacity of the latent representation.

            cross_attention_heads (int): Number of heads in the cross-attention layers.
                                         These layers attend between the latent array and the input or output.

            latent_attention_heads (int): Number of heads in latent self-attention layers.
                                          These layers allow the latent okens to interact with each other.

            cross_attention_head_dim (int): Dimensions of each cross-attention head.
                                            The total dimension of cross-attention will be this value multiplied by cross_attention_heads.

            latent_attention_head_dim (int): Dimension of each latent self-attention head.
                                             The total dimension of latent self-attention will be this value multiplied by latent_attention_heads.

            weight_tie_layers (bool): If True, the weights of the self-attenttion layers will be shared across all layers.
                                      Reduces the number of parameters of course, but may affect performance.

            decoder_feedforward (bool): If True, includes a feedforward layer in the deocer after the cross-attention.
                                        This can increase the model's capacity to transform the attended information (i.e., the latent representation) into the final output.

            sequence_dropout_prob (float): Probability of sequence dropout during training.
                                           This is a form of strucutred dropout that can help with generalization in sequence models.

            num_fourier_bands (int): Number of Fourier bands for positional encoding.
                                     More bands allow for encoding of finer positional details but increase the input dimension.

            max_frequency (float): Maximum frequency for Fourier positional encoding.
                                   This determins the highest frequency componenet in the positional encoding.

            num_input_axes (int): Number of axes in the input data.
                                  For example: 1 for text (sequence length), 2 for images (height, width), 3 for videos (height, width, time).

            position_encoding_type (Optional[str]): Type of positional encoding to use.
                                                    If None, no positional encoding is applied. Current implementation supports 'fourier'.

        """
        super().__init__()
        self.dim = dim
        self.sequence_dropout_prob = sequence_dropout_prob
        self.num_input_axes = num_input_axes
        self.max_frequency = max_frequency
        self.num_fourier_bands = num_fourier_bands
        self.position_encoding_type = position_encoding_type
        self.trainable_position_encoding_kwargs = trainable_position_encoding_kwargs

        self.fourier_channels = self._calculate_fourier_channels()
        self.total_input_dim = self._calculate_total_input_dim()

        # Intiialize learnable latent tokens
        self.latents = nn.Parameter(torch.randn(num_latent_tokens, latent_dimension))

        # Build cross-attention blocks at the beginnging of the model
        self.cross_attend_blocks = self._build_cross_attention_blocks(
            latent_dimension, cross_attention_heads, cross_attention_head_dim, num_kv_heads
        )

        # Build the latent transformer layers
        self.layers = self._build_latent_transformer(
            num_layers, latent_dimension, latent_attention_heads, latent_attention_head_dim, num_kv_heads, weight_tie_layers
        )

        # Build the decoder cross-attention and feedforward (optional) layers
        self.decoder_cross_attn = self._build_decoder_cross_attention(
            queries_dim, latent_dimension, cross_attention_heads, cross_attention_head_dim, num_kv_heads
        )
        self.decoder_feedforward = self._build_decoder_feedforward(queries_dim) if decoder_feedforward else None

        # Build final logits projection (if logits_dimension is provided)
        self.to_logits = nn.Linear(queries_dim, logits_dimension) if logits_dimension is not None else nn.Identity()

    def _calculate_fourier_channels(self):
        """
        Calculate the number of Fourier channels based on the position encoding type.

        Returns:
            int: Number of Fourier channels.
        """
        # TODO: Make it more adapative for various position encoding types (+ various parameters for each type)
        return (self.num_input_axes * ((self.num_fourier_bands * 2) + 1)) if self.position_encoding_type == "fourier" else 0

    def _build_cross_attention_blocks(
        self,
        latent_dimension: int,
        cross_attention_heads: int,
        cross_attention_head_dim: int,
        num_kv_heads: int,
    ) -> nn.ModuleList:
        """
        Build the cross-attention blocks for the encoder.

        Args:
            latent_dimension (int): Dimension of the latent tokens.
            cross_attention_heads (int): Number of heads in cross-attention layers.
            cross_attention_head_dim (int): Dimension of each cross-attention head.

        Returns:
            nn.ModuleList: List containing the cross-attention and feedforward layers.
        """
        # return nn.ModuleList(
        #     [
        #         PreNorm(
        #             latent_dimension,
        #             Attention(
        #                 latent_dimension, self.total_input_dim, heads=cross_attention_heads, head_dim=cross_attention_head_dim
        #             ),
        #             context_dimension=self.total_input_dim,
        #         ),
        #         PreNorm(latent_dimension, FeedForward(latent_dimension)),
        #     ]
        # )
        return nn.ModuleList(
            [
                PreNorm(
                    latent_dimension,
                    BuiltinGQAttention(
                        latent_dimension,
                        self.total_input_dim,
                        n_q_heads=cross_attention_heads,
                        n_kv_heads=num_kv_heads,
                        head_dim=cross_attention_head_dim,
                    ),
                    context_dimension=self.total_input_dim,
                ),
                PreNorm(latent_dimension, FeedForward(latent_dimension)),
            ]
        )

    @cache_fn
    def _get_latent_attention(
        self, latent_dimension: int, latent_attention_heads: int, latent_attention_head_dim: int, num_kv_heads: int, **kwargs
    ) -> PreNorm:
        """
        Get a cached latent attention layer.

        Args:
            latent_dimension (int): Dimension of the latent tokens.
            latent_attention_heads (int): Number of heads in latent self-attention layers.
            latent_attention_head_dim (int): Dimension of each latent self-attention head.
            **kwargs: Additional keyword arguments for caching.

        Returns:
            PreNorm: Normalized latent attention layer.
        """
        # return PreNorm(
        #     latent_dimension, Attention(latent_dimension, heads=latent_attention_heads, head_dim=latent_attention_head_dim)
        # )
        return PreNorm(
            latent_dimension,
            BuiltinGQAttention(
                latent_dimension, n_q_heads=latent_attention_heads, n_kv_heads=num_kv_heads, head_dim=latent_attention_head_dim
            ),
        )

    @cache_fn
    def _get_latent_feedforward(self, latent_dimension: int, **kwargs) -> PreNorm:
        """
        Get a cached latent feedforward layer.

        Args:
            latent_dimension (int): Dimension of the latent tokens.
            **kwargs: Additional keyword arguments for caching.

        Returns:
            PreNorm: Normalized latent feedforward layer.
        """
        return PreNorm(latent_dimension, FeedForward(latent_dimension))

    def _build_latent_transformer(
        self,
        num_layers: int,
        latent_dimension: int,
        latent_attention_heads: int,
        latent_attention_head_dim: int,
        latent_kv_heads: int,
        weight_tie_layers: bool,
    ) -> nn.ModuleList:
        """
        Build the latent transformer layers.

        Args:
            num_layers (int): Number of self-attention layers in the latent transformer.
            latent_dimension (int): Dimension of the latent tokens.
            latent_attention_heads (int): Number of heads in latent self-attention layers.
            latent_attention_head_dim (int): Dimension of each latent self-attention head.
            weight_tie_layers (bool): Whether to tie weights across layers.

        Returns:
            nn.ModuleList: List of latent transformer layers.
        """
        cache_args = {"_cache": weight_tie_layers}

        layers = nn.ModuleList([])

        for _ in range(num_layers):
            latent_attn = self._get_latent_attention(
                latent_dimension, latent_attention_heads, latent_attention_head_dim, latent_kv_heads, **cache_args
            )
            latent_ff = self._get_latent_feedforward(latent_dimension, **cache_args)

            layers.append(nn.ModuleList([latent_attn, latent_ff]))

        return layers

    def _build_decoder_cross_attention(
        self,
        queries_dim: int,
        latent_dimension: int,
        cross_attention_heads: int,
        cross_attention_head_dim: int,
        num_kv_heads: int,
    ) -> PreNorm:
        """
        Build the decoder cross-attention layer.

        Args:
            queries_dim (int): Dimension of the decoder queries.
            latent_dimension (int): Dimension of the latent tokens.
            cross_attention_heads (int): Number of heads in cross-attention layers.
            cross_attention_head_dim (int): Dimension of each cross-attention head.

        Returns:
            PreNorm: Normalized decoder cross-attention layer.
        """
        return PreNorm(
            queries_dim,
            # Attention(queries_dim, latent_dimension, heads=cross_attention_heads, head_dim=cross_attention_head_dim),
            BuiltinGQAttention(
                queries_dim,
                latent_dimension,
                n_q_heads=cross_attention_heads,
                n_kv_heads=num_kv_heads,
                head_dim=cross_attention_head_dim,
            ),
            context_dimension=latent_dimension,
        )

    def _build_decoder_feedforward(self, queries_dim: int) -> PreNorm:
        """
        Build the decoder feedforward layer.

        Args:
            queries_dim (int): Dimension of the decoder queries.

        Returns:
            PreNorm: Normalized decoder feedforward layer.
        """
        return PreNorm(queries_dim, FeedForward(queries_dim))

    def _build_position_encoding(self, shape: Tuple[int, ...]) -> Optional[nn.Module]:
        """
        Build the position encoding module based on the specified type.

        Args:
            shape (Tuple[int, ...]): Shape of the input data.

        Returns:
            Optional[nn.Module]: Position encoding module if specified, else None.
        """

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
        elif self.position_encoding_type == "trainable":
            return build_position_encoding(
                position_encoding_type=self.position_encoding_type,
                index_dims=shape[1:-1],
                trainable_position_encoding_kwargs=self.trainable_position_encoding_kwargs,
            )
        else:
            raise ValueError(f"Unsupported position encoding type: {self.position_encoding_type}")

    def _apply_position_encoding(self, input_data: torch.Tensor) -> torch.Tensor:
        batch_size, *_, _, device = *input_data.shape, input_data.device
        if self.position_encoding_type in ["fourier", "trainable"]:
            pos_encoder = self._build_position_encoding(input_data.shape)
            pos_encoding = pos_encoder(batch_size=batch_size).to(device)
            input_data = torch.cat((input_data, pos_encoding), dim=-1)
        return input_data

    def _calculate_total_input_dim(self) -> int:
        if self.position_encoding_type == "fourier":
            return self.dim + self.fourier_channels
        elif self.position_encoding_type == "trainable":
            return self.dim + self.trainable_position_encoding_kwargs.get("num_channels", 0)
        else:
            return self.dim

    def _process_latents(
        self, latent_representation: torch.Tensor, flattened_input: torch.Tensor, attention_mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """
        Process the latent representation through cross-attention and self-attention layers.

        Args:
            latent_representation (torch.Tensor): Initial latent representation.
            flattened_input (torch.Tensor): Flattened input tensor.
            attention_mask (Optional[torch.Tensor]): Attention mask for the input.

        Returns:
            Processed latent representation.
        """
        cross_attention, cross_feedforward = self.cross_attend_blocks

        # Apply cross-attention between latents and input
        latent_representation = (
            cross_attention(latent_representation, context=flattened_input, mask=attention_mask) + latent_representation
        )
        latent_representation = cross_feedforward(latent_representation) + latent_representation

        # Apply self-attention layers to latents
        for self_attention, self_feedforward in self.layers:
            latent_representation = self_attention(latent_representation) + latent_representation
            latent_representation = self_feedforward(latent_representation) + latent_representation

        return latent_representation

    def _decode(self, latent_representation: torch.Tensor, queries: torch.Tensor) -> torch.Tensor:
        """
        Decode the latent representation using the provided queries.

        Args:
            latent_representation (torch.Tensor): Processed latent representation.
            queries (torch.Tensor): Decoder queries.

        Returns:
            Decoded output.
        """
        # Apply decoder cross-attention
        decoded_output = self.decoder_cross_attn(queries, context=latent_representation)

        # Apply decoder feedforward if available
        if self.decoder_feedforward is not None:
            decoded_output = decoded_output + self.decoder_feedforward(decoded_output)

        # Project to logits
        return self.to_logits(decoded_output)

    def forward(
        self, input_data: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, queries: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass of the PerceiverIO model.

        Args:
            input_data (torch.Tensor): Input tensor.
            attention_mask (Optional[torch.Tensor]): Attention mask for the input.
            queries (Optional[torch.Tensor]): Decoder queries.

        Returns:
            Output tensor (logits or latent representation).
        """
        batch_size, *axis, _ = input_data.shape
        assert len(axis) == self.num_input_axes, f"Input data must have {self.num_input_axes} axes, got {len(axis)}"

        # Apply positional encoding to input data
        input_data = self._apply_position_encoding(input_data)
        flattened_input = rearrange(input_data, "b ... d -> b (...) d")

        # Initialize latent representation
        latent_representation = repeat(self.latents, "n d -> b n d", b=batch_size)

        # Apply sequence dropout during training if specified
        if self.training and self.sequence_dropout_prob > 0.0:
            flattened_input, attention_mask = dropout_seq(flattened_input, attention_mask, self.sequence_dropout_prob)

        # Process latents through cross-attention and self-attention layers
        latent_representation = self._process_latents(latent_representation, flattened_input, attention_mask)

        # If no queries provided, return the latent representation
        if queries is None:
            return latent_representation

        # Expand queries if necessary
        if queries.ndim == 2:
            queries = repeat(queries, "n d -> b n d", b=batch_size)

        # Decode latent representation using queries
        return self._decode(latent_representation, queries)


def main():
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
        position_encoding_type="trainable",
        trainable_position_encoding_kwargs={
            "num_channels": 256,
            "init_scale": 0.02,
        },
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
    print(f"Image-like output shape (with trainable positional encoding): {output_img.shape}")

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
    print(f"Language-like output shape (with fourier positional encoding): {output_lang_pos.shape}")

    # just checking out the number of parameters for each of the models
    print("\nModel parameters:")
    print(f"Language model (no pos encoding): {sum(p.numel() for p in model_lang.parameters()):,}")
    print(f"Image model (with pos encoding): {sum(p.numel() for p in model_img.parameters()):,}")
    print(f"Language model (with pos encoding): {sum(p.numel() for p in model_lang_pos.parameters()):,}")


# if __name__ == "__main__":
#     main()
