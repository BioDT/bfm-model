import weakref
from functools import wraps

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import einsum, nn


def cache_fn(f):
    """
    Caching decorator for functions.

    This decorator implements a cache with weak references to allow garbage collection
    of unused cached results. It also supports optional caching and custom cache keys.

    Args:
        f: Function to be cached

    Returns:
        Wrapped version of the input function that implements caching
    """
    cache = weakref.WeakValueDictionary()

    @wraps(f)
    def cached_fn(*args, _cache=True, key=None, **kwargs):
        if not _cache:
            return f(*args, **kwargs)

        if key is None:
            key = (args, frozenset(kwargs.items()))

        try:
            return cache[key]
        except KeyError:
            result = f(*args, **kwargs)
            cache[key] = result
            return result

    return cached_fn


class PreNorm(nn.Module):
    """
    Helper class that applies layer normalization before a given function.

    This module wraps a given function (typically an attention or feed-forward layer)
    with layer normalization. It can optionally apply normalization to a context tensor as well.
    """

    def __init__(self, dimension: int, function: nn.functional, context_dimension: int = None):
        """
        Initialize the PreNorm module.

        Args:
            dimension: The input dimensionality
            function: The function to be wrapped
            context_dimension: The context dimensionality (if None, no context normalization is applied)
        """
        super().__init__()
        self.norm = nn.LayerNorm(dimension)
        self.function = function
        self.norm_context = nn.LayerNorm(context_dimension) if context_dimension is not None else None

    def forward(self, x, **kwargs):
        # print(f"x shape before norm: {x.shape}")
        x = self.norm(x)
        # print(f"x shape after norm: {x.shape}")

        if self.norm_context is not None:
            context = kwargs["context"]
            kwargs["context"] = self.norm_context(context)  # replacing the original context with a normalized version

        return self.function(x, **kwargs)


class _GLU(nn.Module):
    """
    Gated Linear Unit (GLU) variant.

    Differs from the original GLU. It uses a variant described in https://arxiv.org/pdf/2002.05202.pdf,
    which is designed to provide an extra performance boost to the model, as compared to more tradition ReLU or GELU activations.
    """

    def __init__(self, activation: nn.functional = F.gelu):
        """
        Initialize the _GLU module.

        Args:
            activation: The activation function to use (default
        """
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
        activation: nn.functional = F.gelu,
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
                    _GLU(activation),
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
    Multi-head Attention module that can be used for self-attention or cross-attention.

    Attributes:
        scale: The scaling factor applied to the dot-product of queries and keys
        heads: Number of attention heads
        to_q: Linear layer for projecting the queries
        to_k: Linear layer for projecting the keys
        to_v: Linear layer for projecting the values
        dropout: Dropout layer
        to_out: Linear layer for projecting the concatenated attention heads back to the query dimension

    Note:
        To use this as self-attention, pass the same tensor as both `x` and `context` to an instance of this class.
        To use it as cross-attention, pass different tensors for `x` and `context`.
    """

    def __init__(self, q_dim: int, context_dim: int = None, heads: int = 8, head_dim: int = 64, dropout: float = 0.1):
        """
        Initialize the Attention module.

        Args:
            q_dim: Dimensionality of the query input
            context_dim: Dimensionality of the key/value input (if None, defaults to `q_dim` for self-attention)
            heads: Number of attention heads
            head_dim: Dimensionality of each attention head
            dropout: Dropout probability applied to the attention weights
        """
        super().__init__()
        inner_dim = head_dim * heads
        context_dim = context_dim if context_dim is not None else q_dim

        self.scale = head_dim**-0.5
        self.heads = heads

        # linear projections for Q, K, and V
        self.to_q = nn.Linear(q_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        print(
            f"BuiltinAttention q_dim {q_dim} | context dim {context_dim} | num q heads {heads} | head dim {head_dim} | kv_heads {None}"
        )

        self.dropout = nn.Dropout(dropout)
        # project attention output back to the query dimension
        self.to_out = nn.Linear(inner_dim, q_dim)

    @staticmethod
    def _split_heads(tensor: torch.Tensor, num_heads: int):
        """
        Split the last dimension of the input tensor into (num_heads, head_dim)
        and permutes the dimensions to (batch_size, num_heads, seq_len, head_dim).

        Args:
            tensor: The input tensor of shape (batch_size, seq_len, dim)
            num_heads: The number of attention heads

        Returns:
            The split tensor of shape (batch_size, num_heads, seq_len, head_dim)
        """
        batch_size, seq_len, dim = tensor.shape
        head_dim = dim // num_heads
        tensor = tensor.reshape(batch_size, seq_len, num_heads, head_dim)
        return tensor.permute(0, 2, 1, 3)

    @staticmethod
    def _merge_heads(tensor: torch.Tensor, num_heads: int):
        """
        Reverse the _split_heads operation.

        Args:
            tensor: The input tensor of shape (batch_size, num_heads, seq_len, head_dim)
            num_heads: The number of attention heads

        Returns:
            The merged tensor of shape (batch_size, seq_len, num_heads * head_dim)
        """
        batch_size, _, seq_len, head_dim = tensor.shape
        return tensor.permute(0, 2, 1, 3).reshape(batch_size, seq_len, num_heads * head_dim)

    def forward(self, x: torch.Tensor, context: torch.Tensor = None, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Applies multi-head attention to the input.

        Args:
            x: Query input of shape (batch_size, seq_len_q, q_dim)
            context: Key/Value input of shape (batch_size, seq_len_kv, context_dim)
                     If None, use x for self-attention
            mask: Attention mask of shape (batch_size, seq_len_q, seq_len_kv)

        Returns:
            Attention output of shape (batch_size, seq_len_q, q_dim)
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

        if mask is not None:
            # Reshape and repeat mask for each head
            mask = rearrange(mask, "b ... -> b (...)")
            mask = repeat(mask, "b j -> b h () j", h=h)
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


class GQAAttention(nn.Module):
    """
    Multi-head Attention with Grouped Query Attention (GQA).

    - n_q_heads: number of query heads
    - n_kv_heads: number of key/value heads
    - head_dim: dimension per head
    - dropout: dropout for attention weights
    - q_dim, context_dim: input dimensions for Q and K/V
    """

    def __init__(
        self,
        q_dim: int,
        context_dim: int = None,
        n_q_heads: int = 8,
        n_kv_heads: int = None,
        head_dim: int = 64,
        dropout: float = 0.1,
    ):
        """
        Args:
            q_dim: Dimensionality of the query input
            context_dim: Dimensionality of the key/value input (if None, defaults to `q_dim` for self-attention)
            n_q_heads: Number of query heads
            n_kv_heads: Number of key/value heads (defaults to same as n_q_heads if None)
            head_dim: Dimensionality per attention head
            dropout: Dropout probability for attention weights
        """
        super().__init__()
        self.n_q_heads = n_q_heads
        self.n_kv_heads = n_kv_heads if n_kv_heads is not None else n_q_heads
        self.head_dim = head_dim

        context_dim = context_dim if context_dim is not None else q_dim
        self.scale = head_dim**-0.5

        # Projections:
        self.to_q = nn.Linear(q_dim, self.n_q_heads * head_dim, bias=False)
        self.to_k = nn.Linear(context_dim, self.n_kv_heads * head_dim, bias=False)
        self.to_v = nn.Linear(context_dim, self.n_kv_heads * head_dim, bias=False)

        self.dropout = nn.Dropout(dropout)
        # Final projection projects back to q_dim (same as total Q dimension)
        self.to_out = nn.Linear(self.n_q_heads * head_dim, q_dim)

    @staticmethod
    def _split_heads(tensor: torch.Tensor, num_heads: int):
        """
        Splits the last dimension into (num_heads, head_dim) and rearranges
        to (batch_size, num_heads, seq_len, head_dim).
        """
        b, seq_len, dim = tensor.shape
        head_dim = dim // num_heads
        tensor = tensor.reshape(b, seq_len, num_heads, head_dim)
        return tensor.permute(0, 2, 1, 3)  # (b, num_heads, seq_len, head_dim)

    @staticmethod
    def _merge_heads(tensor: torch.Tensor, num_heads: int):
        """
        Inverse of _split_heads. Merges num_heads into the last dimension.
        """
        b, _, seq_len, head_dim = tensor.shape
        return tensor.permute(0, 2, 1, 3).reshape(b, seq_len, num_heads * head_dim)

    def forward(self, x: torch.Tensor, context: torch.Tensor = None, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len_q, q_dim) for queries
            context: (batch_size, seq_len_kv, context_dim) for keys/values
                     if None, self-attention (context = x)
            mask: (batch_size, seq_len_q, seq_len_kv) boolean mask (True=keep, False=mask out)

        Returns:
            (batch_size, seq_len_q, q_dim)
        """
        context = x if context is None else context

        # Project Q, K, V
        q = self.to_q(x)  # (b, seq_len_q, n_q_heads*head_dim)
        k = self.to_k(context)  # (b, seq_len_kv, n_kv_heads*head_dim)
        v = self.to_v(context)  # (b, seq_len_kv, n_kv_heads*head_dim)

        # Split heads
        q = self._split_heads(q, self.n_q_heads)  # (b, n_q_heads, seq_len_q, head_dim)
        k = self._split_heads(k, self.n_kv_heads)  # (b, n_kv_heads, seq_len_kv, head_dim)
        v = self._split_heads(v, self.n_kv_heads)  # (b, n_kv_heads, seq_len_kv, head_dim)

        # If we have fewer KV heads than Q heads, replicate K and V
        # so the final dot-product matches shape: (b, n_q_heads, seq_len_q, seq_len_kv)
        if self.n_kv_heads < self.n_q_heads:
            repeat_factor = self.n_q_heads // self.n_kv_heads
            # replicate the head dimension
            k = k.repeat_interleave(repeat_factor, dim=1)
            v = v.repeat_interleave(repeat_factor, dim=1)

        # scaled dot-product: (b, n_q_heads, seq_len_q, seq_len_kv)
        sim = torch.einsum("bhid,bhjd->bhij", q, k) * self.scale

        # Apply mask if provided: mask shape (b, seq_len_q, seq_len_kv)
        if mask is not None:
            # Expand to match sim: (b, 1, seq_len_q, seq_len_kv) -> broadcast heads
            mask = mask.unsqueeze(1)
            max_neg_value = -torch.finfo(sim.dtype).max
            sim = sim.masked_fill(~mask, max_neg_value)

        attn = sim.softmax(dim=-1)  # (b, n_q_heads, seq_len_q, seq_len_kv)
        attn = self.dropout(attn)

        # Weighted sum: (b, n_q_heads, seq_len_q, head_dim)
        out = torch.einsum("bhij,bhjd->bhid", attn, v)

        # Merge heads: shape => (b, seq_len_q, n_q_heads*head_dim)
        out = self._merge_heads(out, self.n_q_heads)

        # Final projection to match q_dim
        return self.to_out(out)


class BuiltinGQAAttention(nn.Module):
    """
    Multi-head Attention module using PyTorch's built-in GQA support.
    By setting n_q_heads != n_kv_heads, we enable Grouped Query Attention.

    Attributes:
        n_q_heads: Number of query heads
        n_kv_heads: Number of key/value heads
        head_dim: Dimensionality of each attention head
        q_dim: Dimension of query input
        context_dim: Dimension of key/value input
    """

    def __init__(
        self,
        q_dim: int,
        context_dim: int = None,
        n_q_heads: int = 8,
        n_kv_heads: int = None,
        head_dim: int = 64,
        dropout: float = 0.1,
        is_causal: bool = False,
    ):
        """
        Args:
            q_dim: Dimensionality of the query input
            context_dim: Dim of the key/value input (if None, defaults to q_dim for self-attn)
            n_q_heads: Number of query heads
            n_kv_heads: Number of key/value heads (defaults to same as n_q_heads if None)
            head_dim: Dimensionality per head
            dropout: Dropout probability for attention weights
            is_causal: Whether to apply causal masking by default in attention
        """
        super().__init__()

        self.n_q_heads = n_q_heads
        self.n_kv_heads = n_kv_heads if n_kv_heads is not None else n_q_heads
        self.head_dim = head_dim
        self.is_causal = is_causal
        self.q_dim = q_dim
        context_dim = context_dim if context_dim is not None else q_dim
        self.context_dim = context_dim
        print(
            f"BuiltinAttention q_dim {q_dim} | context dim {context_dim} | num q heads {n_q_heads} | head dim {head_dim} | kv_heads {n_kv_heads}"
        )

        self.dropout_p = dropout

        # Linear projections
        self.q_proj = nn.Linear(q_dim, n_q_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(self.context_dim, self.n_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(self.context_dim, self.n_kv_heads * head_dim, bias=False)

        # Final projection
        self.out_proj = nn.Linear(n_q_heads * head_dim, q_dim)

    def forward(
        self, x: torch.Tensor, context: torch.Tensor = None, mask: torch.Tensor = None, is_causal: bool = None
    ) -> torch.Tensor:
        """
        Args:
            x: Queries (batch_size, seq_len_q, q_dim)
            context: Keys/Values (batch_size, seq_len_kv, context_dim).
                     If None, use x (self-attn).
            mask: Optional attention mask. Should be broadcastable to shape:
                       (batch_size, seq_len_q, seq_len_kv).
                       True = attend, False = mask out.
            is_causal: Whether to apply causal masking. If not specified, defaults to self.is_causal.

        Returns:
            Tensor of shape (batch_size, seq_len_q, q_dim) after attention.
        """
        if context is None:
            context = x

        is_causal = self.is_causal if is_causal is None else is_causal

        bsz, seq_len_q, _ = x.shape
        seq_len_kv = context.size(1)

        # 1) Project Q, K, V
        q = self.q_proj(x)  # (bsz, seq_len_q, n_q_heads * head_dim)
        k = self.k_proj(context)  # (bsz, seq_len_kv, n_kv_heads * head_dim)
        v = self.v_proj(context)  # (bsz, seq_len_kv, n_kv_heads * head_dim)

        # 2) Reshape into [batch, n_q_heads, seq_len, head_dim] for Q
        #    and [batch, n_kv_heads, seq_len, head_dim] for K/V.
        #    Then F.scaled_dot_product_attention expects shapes [batch, heads, seq_len, head_dim].
        q = q.reshape(bsz, seq_len_q, self.n_q_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(bsz, seq_len_kv, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(bsz, seq_len_kv, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # 3) Convert boolean mask => shape [bsz, 1, seq_len_q, seq_len_kv] or None
        #    PyTorch's scaled_dot_product_attention expects shape [bsz, heads, seq_len_q, seq_len_kv].
        #    But with GQA, heads can differ for Q vs. K. The function handles it automatically if enable_gqa=True.
        #    We just need to ensure the first dimension matches the number of Q heads = n_q_heads.
        #    If mask is [bsz, seq_len_q, seq_len_kv], we can unsqueeze(1).
        final_attn_mask = None
        if mask is not None:
            # Expecting True=keep, False=mask -> we invert or we rely on the built-in for masked_fill
            # scaled_dot_product_attention expects float mask with -inf for masked tokens, or None.
            # We can convert boolean to float mask or keep it as bool. Let's keep it as bool.
            # Expand to [bsz, 1, seq_len_q, seq_len_kv] so it can broadcast to all heads.
            final_attn_mask = mask.unsqueeze(1)  # (bsz, 1, seq_len_q, seq_len_kv)

        # 4) Let PyTorch do the heavy lifting with built-in GQA logic
        #    shape returned: (bsz, n_q_heads, seq_len_q, head_dim)
        #    enable_gqa=True tells PyTorch we have different # of heads for Q vs K/V
        #    and it will replicate K/V heads if needed.
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=final_attn_mask,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=is_causal,
            enable_gqa=True,  # <--- GQA flag
        )

        # 5) out shape is [bsz, n_q_heads, seq_len_q, head_dim]. We need to bring it back to [bsz, seq_len_q, n_q_heads * head_dim].
        out = out.transpose(1, 2).reshape(bsz, seq_len_q, self.n_q_heads * self.head_dim)

        # 6) Final linear
        out = self.out_proj(out)
        return out
