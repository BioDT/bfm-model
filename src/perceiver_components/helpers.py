import torch
import torch.nn.functional as F
from torch import nn


class PreNorm(nn.Module):
    def __init__(self, dimension, function, context_dimension=None):
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
    "https://arxiv.org/pdf/2002.05202"

    def __init__(self, activation=F.gelu):
        super().__init__()
        self.activation = activation

    def forward(self, x: torch.Tensor):
        out, gate = x.chunk(2, dim=-1)
        return out * self.activation(gate)


class FeedForward(nn.Module):
    def __init__(
        self, dimension, multiplier=4, hidden_dimension=None, out_dimension=None, actiavtion=F.gelu, dropout=0.1, num_layers=1
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
    def __init__(self, q_dim, context_dim=None, heads=8, head_dim=64, dropout=0.1):
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
    def _split_heads(tensor, num_heads):
        """
        Split the last dimension of the input tensor into (num_heads, head_dim)
        and permutes the dimensions to (batch_size, num_heads, seq_len, head_dim).
        """
        batch_size, seq_len, dim = tensor.shape
        head_dim = dim // num_heads
        tensor = tensor.reshape(batch_size, seq_len, num_heads, head_dim)
        return tensor.permute(0, 2, 1, 3)

    @staticmethod
    def _merge_heads(tensor, num_heads):
        """
        Reverse the _split_heads operation. Permutes the dimensions and reshapes
        the tensor to (batch_size, seq_len, num_heads * head_dim).
        """
        batch_size, _, seq_len, head_dim = tensor.shape
        return tensor.permute(0, 2, 1, 3).reshape(batch_size, seq_len, num_heads * head_dim)

    def forward(self, x, context=None, mask=None):
        h = self.heads

        # Shape: (batch_size, seq_len_q, inner_dim)
        q = self.to_q(x)
        context = context or x
        k = self.to_k(context)  # shape: (batch_size, seq_len_kv, inner_dim)
        v = self.to_v(context)  # shape: (batch_size, seq_len_kv, inner_dim)

        # Split heads
        q, k, v = map(lambda t: self._split_heads(t, h), (q, k, v))
        # after splitting:
        # q: (batch_size, num_heads, seq_len_q, head_dim)
        # k, v: (batch_size, num_heads, seq_len_kv, head_dim)

        # scaled dot-product attention
        sim = torch.einsum("bhid, bhjd -> bhij", q, k) * self.scale  # shape: (batch_size, num_heads, seq_len_q, seq_len_kv)

        if mask is not None:
            # Add two dimensions to mask for broadcasting over heads and queries
            mask = mask.unsqueeze(1).unsqueeze(1)
            sim = sim.masked_fill(~mask, -torch.finfo(sim.dtype).max)

        # compute attention (is all we need)
        attention = sim.softmax(dim=-1)
        attention = self.dropout(attention)

        out = torch.einsum("bhij, bhjd ->bhid", attention, v)

        # Merge heads
        out = self._merge_heads(out, h)
        # out shape: (batch_size, seq_len_q, inner_dim)

        return self.to_out(out)


# TODO:
# 1) Perceiver
