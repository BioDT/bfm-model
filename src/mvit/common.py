import torch
import torch.nn as nn


class MLPBlock(nn.Module):
    """Multi-Layer Perceptron block commonly used in Transformer architectures. The cute thing with which it all started.

    Implements a two-layer MLP with configurable hidden dimension and dropout.
    Used as the feed-forward network component in Transformer blocks.

    Args:
        input_dim: Input feature dimension
        hidden_dim: Hidden layer dimension. Default: None (same as input_dim)
        output_dim: Output feature dimension. Default: None (same as input_dim)
        activation_layer: Activation function to use. Default: nn.GELU
        dropout_rate: Dropout probability. Default: 0.0
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = None,
        output_dim: int = None,
        activation_layer: nn.Module = nn.GELU,
        dropout_rate: float = 0.0,
    ):
        super().__init__()
        self.dropout_rate = dropout_rate

        output_dim = output_dim or input_dim
        hidden_dim = hidden_dim or input_dim

        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.activation = activation_layer()
        self.linear2 = nn.Linear(hidden_dim, output_dim)

        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0.0 else nn.Identity()

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Forward pass of MLP block.

        Args:
            input_tensor: Input tensor of shape [batch, *, input_dim]

        Returns:
            Output tensor of shape [batch, *, output_dim]
        """
        hidden_features = self.activation(self.linear1(input_tensor))
        hidden_features = self.dropout(hidden_features)
        output_features = self.linear2(hidden_features)
        output_features = self.dropout(output_features)
        return output_features


class Permute(nn.Module):
    """Simple permutation layer that rearranges tensor dimensions.

    Args:
        permutation_order: Target dimension ordering
    """

    def __init__(self, permutation_order: tuple):
        super().__init__()
        self.permutation_order = permutation_order

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        return input_tensor.permute(*self.permutation_order)


class StochasticDepth(nn.Module):
    """Implements Stochastic Depth (Drop Path) regularization.

    Randomly drops entire paths (channels) in residual networks during training.
    Acts as identity during inference. Helps prevent overfitting.

    Args:
        dropout_probability: Probability of dropping a path. Default: 0.0
    """

    def __init__(self, dropout_probability: float = 0.0):
        super().__init__()
        self.dropout_probability = dropout_probability

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        if self.dropout_probability == 0.0 or not self.training:
            return input_tensor

        survival_probability = 1 - self.dropout_probability

        # making a broadcastable shape for the dropout mask
        mask_shape = (input_tensor.shape[0],) + (1,) * (input_tensor.ndim - 1)  # shape: [batch_size, 1, 1, ...]

        # generate binary mask and scale output to maintain expected value
        # this will basically create a boolean mask, where each element is True with probability survival_probability
        survival_mask = torch.rand(mask_shape, dtype=input_tensor.dtype, device=input_tensor.device) < survival_probability
        # scaling the output to maintain expected value - this is the actual dropout
        scaled_output = input_tensor * survival_mask / survival_probability

        return scaled_output
