from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from einops import repeat

from src.perceiver_components.helpers_io import (
    Attention,
    FeedForward,
    PreNorm,
    cache_fn,
    dropout_seq,
)
from src.perceiver_components.pos_encoder import build_position_encoding


class FlexiblePerceiverIO(nn.Module):
    def __init__(
        self,
        input_configs: Dict[str, Dict[str, int]],
        num_layers: int,
        num_latent_tokens: int = 512,
        latent_dimension: int = 512,
        cross_attention_heads: int = 1,
        latent_attention_heads: int = 8,
        cross_attention_head_dim: int = 64,
        latent_attention_head_dim: int = 64,
        weight_tie_layers: bool = False,
        decoder_feedforward: bool = False,
        sequence_dropout_prob: float = 0.0,
        queries_dim: int = 128,
        logits_dimension: Optional[int] = None,
        position_encoding_type: str = "fourier",  # TODO: add trainable/none
        num_fourier_bands: int = 64,
        max_frequency: float = 224.0,
        trainable_position_encoding_kwargs: Optional[dict] = None,
    ):
        super().__init__()
        self.input_configs = input_configs
        self.latent_dimension = latent_dimension
        self.sequence_dropout_prob = sequence_dropout_prob
        self.position_encoding_type = position_encoding_type
        self.num_fourier_bands = num_fourier_bands
        self.max_frequency = max_frequency
        self.trainable_position_encoding_kwargs = trainable_position_encoding_kwargs

        self._init_position_encoding()
        self._init_input_processing()

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

    def _init_position_encoding(self):
        """
        Initialize position encoding modules for each modality.

        This method creates position encoding modules (either Fourier or trainable)
        for each input modality based on the specified configuration.
        """
        self.position_encodings = nn.ModuleDict()
        for modality, config in self.input_configs.items():
            if self.position_encoding_type == "fourier":
                self.position_encodings[modality] = build_position_encoding(
                    position_encoding_type="fourier",
                    index_dims=(config["max_len"],),
                    fourier_position_encoding_kwargs={
                        "num_bands": self.num_fourier_bands,
                        "max_freq": self.max_frequency,
                        "concat_pos": True,
                        "sine_only": False,
                    },
                )
            elif self.position_encoding_type == "trainable":
                self.position_encodings[modality] = build_position_encoding(
                    position_encoding_type="trainable",
                    index_dims=(config["max_len"],),
                    trainable_position_encoding_kwargs=self.trainable_position_encoding_kwargs,
                )

    def _init_input_processing(self):
        """
        Initialize input projection layers for each modality.

        This method creates a linear projection layer for each input modality,
        taking into account the input dimension and the additional dimensions
        from positional encodings.
        """
        self.input_projections = nn.ModuleDict()
        for modality, config in self.input_configs.items():
            input_dim = config["dim"]

            # get the actual output size of the position encoding (more flexible approach)
            if self.position_encoding_type == "fourier":
                pos_encoding = self.position_encodings[modality]
                # Assume a batch size of 1 and sequence length of 1 to get the feature dimension
                pos_encoding_dim = pos_encoding(batch_size=1).shape[-1]
                input_dim += pos_encoding_dim
            elif self.position_encoding_type == "trainable":
                input_dim += self.trainable_position_encoding_kwargs.get("num_channels", 0)

            self.input_projections[modality] = nn.Linear(input_dim, self.latent_dimension)

    def _build_cross_attention_blocks(
        self, latent_dimension: int, cross_attention_heads: int, cross_attention_head_dim: int
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
        return nn.ModuleList(
            [
                PreNorm(
                    latent_dimension,
                    Attention(
                        latent_dimension, self.latent_dimension, heads=cross_attention_heads, head_dim=cross_attention_head_dim
                    ),
                    context_dimension=self.latent_dimension,
                ),
                PreNorm(latent_dimension, FeedForward(latent_dimension)),
            ]
        )

    @cache_fn
    def _get_latent_attention(
        self, latent_dimension: int, latent_attention_heads: int, latent_attention_head_dim: int, **kwargs
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
        return PreNorm(
            latent_dimension, Attention(latent_dimension, heads=latent_attention_heads, head_dim=latent_attention_head_dim)
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
        return nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        self._get_latent_attention(
                            latent_dimension, latent_attention_heads, latent_attention_head_dim, **cache_args
                        ),
                        self._get_latent_feedforward(latent_dimension, **cache_args),
                    ]
                )
                for _ in range(num_layers)
            ]
        )

    def _build_decoder_cross_attention(
        self, queries_dim: int, latent_dimension: int, cross_attention_heads: int, cross_attention_head_dim: int
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
            Attention(queries_dim, latent_dimension, heads=cross_attention_heads, head_dim=cross_attention_head_dim),
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

    def _process_inputs(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Process input tensors by adding positional encodings and applying input projections.

        Args:
            inputs (Dict[str, torch.Tensor]): Dictionary of input tensors for each modality.

        Returns:
            torch.Tensor: Processed and concatenated input tensor.
        """
        processed_inputs = []
        for modality, x in inputs.items():
            batch_size, seq_len, _ = x.shape
            pos_encoding = self.position_encodings[modality](batch_size=batch_size)
            x = torch.cat([x, pos_encoding], dim=-1)
            x = self.input_projections[modality](x)
            processed_inputs.append(x)
        return torch.cat(processed_inputs, dim=1)

    def _process_latents(
        self, latent_representation: torch.Tensor, flattened_input: torch.Tensor, attention_mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """
        Process the latent representation through cross-attention and self-attention layers.

        Args:
            latent_representation (torch.Tensor): Initial latent representation.
            flattened_input (torch.Tensor): Flattened and processed input tensor.
            attention_mask (Optional[torch.Tensor]): Attention mask for the input.

        Returns:
            torch.Tensor: Processed latent representation.
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
            torch.Tensor: Decoded output.
        """
        # Apply decoder cross-attention
        decoded_output = self.decoder_cross_attn(queries, context=latent_representation)

        # Apply decoder feedforward if available
        if self.decoder_feedforward is not None:
            decoded_output = decoded_output + self.decoder_feedforward(decoded_output)

        # Project to logits
        return self.to_logits(decoded_output)

    def forward(
        self,
        inputs: Dict[str, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        queries: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass of the FlexiblePerceiverIO model.

        Args:
            inputs (Dict[str, torch.Tensor]): Dictionary of input tensors for each modality.
            attention_mask (Optional[torch.Tensor]): Attention mask for the input.
            queries (Optional[torch.Tensor]): Decoder queries.

        Returns:
            torch.Tensor: Output tensor (logits or latent representation).
        """
        batch_size = list(inputs.values())[0].shape[0]

        processed_inputs = self._process_inputs(inputs)

        # # Apply sequence dropout during training if specified
        if self.training and self.sequence_dropout_prob > 0.0:
            processed_inputs, attention_mask = dropout_seq(processed_inputs, attention_mask, self.sequence_dropout_prob)

        # Initialize latent representation
        latent_representation = repeat(self.latents, "n d -> b n d", b=batch_size)

        # Process latents through cross-attention and self-attention layers
        latent_representation = self._process_latents(latent_representation, processed_inputs, attention_mask)

        # If no queries provided, return the latent representation
        if queries is None:
            return latent_representation

        # Expand queries if necessary
        if queries.ndim == 2:
            queries = repeat(queries, "n d -> b n d", b=batch_size)

        # Decode latent representation using queries
        return self._decode(latent_representation, queries)


def main():
    # Define input configurations
    input_configs = {
        "image": {"dim": 2048, "max_len": 196},  # e.g., 14x14 grid of 2048-dim features
        "audio": {"dim": 1024, "max_len": 512},  # e.g., 512 time steps of 1024-dim features
        "text": {"dim": 768, "max_len": 128},  # e.g., 128 tokens of 768-dim embeddings
    }

    model = FlexiblePerceiverIO(
        input_configs=input_configs,
        num_layers=6,
        num_latent_tokens=256,
        latent_dimension=512,
        cross_attention_heads=1,
        latent_attention_heads=8,
        cross_attention_head_dim=64,
        latent_attention_head_dim=64,
        queries_dim=128,
        logits_dimension=1000,
        position_encoding_type="fourier",
        num_fourier_bands=64,
        max_frequency=224.0,
    )

    device = torch.device("cpu")
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # train step
    def train_step(inputs, labels):
        model.train()
        optimizer.zero_grad()
        # single classification query
        queries = torch.randn(1, 128).to(
            device
        )  # 1 - single query, 128- dimension of th equery vector (should match queries_dim)
        outputs = model(inputs, queries=queries)

        loss = criterion(outputs.squeeze(1), labels)
        loss.backward()
        optimizer.step()

        return loss.item()

    # example training scenarios
    for epoch in range(1):  # 3 epochs for demonstration
        print(f"Epoch {epoch + 1}")

        # Scensario 1: Training on image embeddings only
        image_embeddings = torch.randn(32, 196, 2048).to(device)
        labels = torch.randint(0, 1000, (32,)).to(device)

        loss = train_step({"image": image_embeddings}, labels)
        print(f"  Image only loss: {loss:.4f}")

        # Scenario 2: Training on audio embeddings only
        audio_embeddings = torch.randn(32, 512, 1024).to(device)
        labels = torch.randint(0, 1000, (32,)).to(device)

        loss = train_step({"audio": audio_embeddings}, labels)
        print(f"  Audio only loss: {loss:.4f}")

        # Scenario 3: Training on text embeddings only
        text_embeddings = torch.randn(32, 128, 768).to(device)
        labels = torch.randint(0, 1000, (32,)).to(device)

        loss = train_step({"text": text_embeddings}, labels)
        print(f"  Text only loss: {loss:.4f}")

        # Scenario 4: Training on all modalities
        image_embeddings = torch.randn(32, 196, 2048).to(device)
        audio_embeddings = torch.randn(32, 512, 1024).to(device)
        text_embeddings = torch.randn(32, 128, 768).to(device)
        labels = torch.randint(0, 1000, (32,)).to(device)

        loss = train_step({"image": image_embeddings, "audio": audio_embeddings, "text": text_embeddings}, labels)
        print(f"  Multi-modal loss: {loss:.4f}")

    # Inference example
    model.eval()
    with torch.no_grad():
        # Single sample with all modalities
        image_emb = torch.randn(1, 196, 2048).to(device)
        audio_emb = torch.randn(1, 512, 1024).to(device)
        text_emb = torch.randn(1, 128, 768).to(device)

        queries = torch.randn(1, 128).to(device)

        output = model({"image": image_emb, "audio": audio_emb, "text": text_emb}, queries=queries)

        predicted_class = output.argmax(dim=-1)
        print(f"Predicted class: {predicted_class.item()}")

    print("Training and inference completed, yEeEeEeEe-HaAaAaW!")


if __name__ == "__main__":
    main()
