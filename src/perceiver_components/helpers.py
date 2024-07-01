import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import einsum, nn


class PreNorm(nn.Module):
    """
    Helper class that applies layer normalization before a given function.

    This module wraps a given function (typically an attention or feed-forward layer)
    with layer normalization. It can optionally apply normalization to a context tensor as well.
    """

    def __init__(self, dimension: int, function: nn.functional, context_dimension: int = None):
        super().__init__()
        self.norm = nn.LayerNorm(dimension)
        self.function = function
        self.norm_context = nn.LayerNorm(context_dimension) if context_dimension is not None else None

    def forward(self, x, **kwargs):
        x = self.norm(x)

        if self.norm_context is not None:
            context = kwargs["context"]
            kwargs["context"] = self.norm_context(context)  # replacing the original context with a normalized version

        return self.function(x, **kwargs)


class _GLU(nn.Module):
    """
    Gated Linear Unit (GLU) variant.

    Differs from the original GLU. It uses a variant described in https://arxiv.org/pdf/2002.05202.pdf,
    which is designed to provide an extra performance boost to the model, as compared to more tradition ReLU or GELU activations.

    :param activation: The activation function to use (default: GELU).
    """

    def __init__(self, activation: nn.functional = F.gelu):
        super().__init__()
        self.activation = activation

    def forward(self, x: torch.Tensor):
        out, gate = x.chunk(2, dim=-1)
        return out * self.activation(gate)


class FeedForward(nn.Module):
    """
    A simple Feed Forward Neural Network with a _GLU variant.
    """

    def __init__(
        self,
        dimension: int,
        multiplier: int = 4,
        hidden_dimension: int = None,
        out_dimension: int = None,
        actiavtion: nn.functional = F.gelu,
        dropout: float = 0.1,
        num_layers: int = 1,
    ):
        super().__init__()
        hidden_dimension = hidden_dimension or dimension * multiplier
        out_dimension = out_dimension or dimension  # probably will always be kept at 'dimension'

        self.layers = nn.ModuleList()
        in_dimension = dimension

        for _ in range(num_layers):
            self.layers.append(
                nn.Sequential(
                    nn.Linear(in_dimension, hidden_dimension * 2),
                    _GLU(actiavtion),
                    nn.Linear(hidden_dimension, out_dimension),  # without the '*2' for because of _GLU splitting
                    nn.Dropout(dropout),
                )
            )
            in_dimension = out_dimension

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class Attention(nn.Module):
    """
    Multi-head Attention module that cna be used for self-attention or cross-attention.

    :param q_dim: Dimensionality of the query input
    :param context_dim: Dimensionality of the key/value input (if None, defaults to `q_dim` for self-attention)
    :param heads: Number of attention heads
    :param head_dim: Dimensionality of each attention head
    :param dropout: Dropout probability applied to the attention weights

    :ivar scale: The scaling factor applied to the dot-product of queries and keys
    :ivar heads: Number of attention heads
    :ivar to_q: Linear layer for projecting the queries
    :ivar to_k: Linear layer for projecting the keys
    :ivar to_v: Linear layer for projecting the values
    :ivar dropout: Dropout layer
    :ivar to_out: Linear layer for projecting the concatenated attention heads back to the query dimension

    .. note::
        To use this as self-attention, pass the same tensor as both `x` and `context` to an instance of this class.
        To use it as cross-attention, pass different tensors for `x` and `context`.
    """

    def __init__(self, q_dim: int, context_dim: int = None, heads: int = 8, head_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        inner_dim = head_dim * heads
        context_dim = context_dim or q_dim

        self.scale = head_dim**-0.5
        self.heads = heads

        # linear projections for Q, K, and V
        self.to_q = nn.Linear(q_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.dropout = nn.Dropout(dropout)
        # project attention output back to the query dimension
        self.to_out = nn.Linear(inner_dim, q_dim)

    @staticmethod
    def _split_heads(tensor: torch.Tensor, num_heads: int):
        """
        Split the last dimension of the input tensor into (num_heads, head_dim)
        and permutes the dimensions to (batch_size, num_heads, seq_len, head_dim).

        :param tensor: The input tensor of shape (batch_size, seq_len, dim)
        :param num_heads: The number of attention heads

        :return: The split tensor of shape (batch_size, num_heads, seq_len, head_dim)
        """
        batch_size, seq_len, dim = tensor.shape
        head_dim = dim // num_heads
        tensor = tensor.reshape(batch_size, seq_len, num_heads, head_dim)
        return tensor.permute(0, 2, 1, 3)

    @staticmethod
    def _merge_heads(tensor: torch.Tensor, num_heads: int):
        """
        Reverse the _split_heads operation.

        :param tensor: The input tensor of shape (batch_size, num_heads, seq_len, head_dim)
        :param num_heads: The number of attention heads

        :return: The merged tensor of shape (batch_size, seq_len, num_heads * head_dim)
        """
        batch_size, _, seq_len, head_dim = tensor.shape
        return tensor.permute(0, 2, 1, 3).reshape(batch_size, seq_len, num_heads * head_dim)

    def forward(self, x: torch.Tensor, context: torch.Tensor = None, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Applies multi-head attention to the input.

        :param x: Query input of shape (batch_size, seq_len_q, q_dim)
        :param context: Key/Value input of shape (batch_size, seq_len_kv, context_dim)
                        If None, use x for self-attention
        :param mask: Attention mask of shape (batch_size, seq_len_q, seq_len_kv)
        :return: Attention output of shape (batch_size, seq_len_q, q_dim)
        """
        h = self.heads

        # Shape: (batch_size, seq_len_q, inner_dim)
        q = self.to_q(x)
        context = context if context is not None else x
        k = self.to_k(context)  # shape: (batch_size, seq_len_kv, inner_dim)
        v = self.to_v(context)  # shape: (batch_size, seq_len_kv, inner_dim)

        # Split heads
        q, k, v = map(lambda t: self._split_heads(t, h), (q, k, v))
        # after splitting:
        # q: (batch_size, num_heads, seq_len_q, head_dim)
        # k, v: (batch_size, num_heads, seq_len_kv, head_dim)

        # scaled dot-product attention
        sim = torch.einsum("bhid, bhjd -> bhij", q, k) * self.scale  # shape: (batch_size, num_heads, seq_len_q, seq_len_kv)

        print(f"Shape of sim: {sim.shape}")
        if mask is not None:
            mask = mask.unsqueeze(1)  # Add head dimension
            print(f"Shape of mask: {mask.shape}")
            max_neg_value = -torch.finfo(sim.dtype).max
            sim = sim.masked_fill(~mask, max_neg_value)

        # compute attention (is all we need)
        attention = sim.softmax(dim=-1)
        attention = self.dropout(attention)

        out = torch.einsum("bhij, bhjd -> bhid", attention, v)

        # Merge heads
        out = self._merge_heads(out, h)
        # out shape: (batch_size, seq_len_q, inner_dim)

        return self.to_out(out)


# TODO:
# 1) Perceiver
