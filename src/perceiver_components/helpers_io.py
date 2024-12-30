import weakref
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import nn

from src.perceiver_components.helpers import _GLU
from src.perceiver_components.helpers import Attention as BaseAttention
from src.perceiver_components.helpers import GQAAttention as BaseGQAttention
from src.perceiver_components.helpers import BuiltinGQAAttention as BaseBuiltinGQAttention
from src.perceiver_components.helpers import FeedForward as BaseFeedForward
from src.perceiver_components.helpers import PreNorm


def dropout_seq(seq: torch.Tensor, mask: Optional[torch.Tensor], dropout: float):
    """
    Apply structured dropout to a sequence of tokens.

    Implements a form of structured dropout where entire tokens/postiions in the sequence are dropped out, rather than individual elements.

    Args:
        seq (torch.Tensor): The input sequence tensor of shape (batch_size, seq_length, ...)
        mask (torch.Tensor): An optional boolean mask of shape (batch_size, seq_length) indicating which elements in the sequence are valid (True) or padding (False).
        dropout (float): The probability of dropping out a token, should be in range [0, 1).

    Returns:
        A tuple containing:
            - The sequence after applying dropout, with shape (batch_size, new_seq_length, ...)
              where new_seq_length <= seq_length.
            - The updated mask (if a mask was provided), with shape (batch_size, new_seq_length).
    """
    batch_size, seq_length, *_, device = *seq.shape, seq.device
    # Generate random logits for dropout decision
    logits = torch.randn(batch_size, seq_length, device=device)

    # Apply mask to logits if provided
    if mask is not None:
        logits = logits.masked_fill(~mask, -torch.finfo(logits.dtype).max)

    # Compute number of tokens to keep
    keep_probability = 1.0 - dropout
    num_keep = max(1, int(keep_probability * seq_length))

    # Select indices of tokens to keep
    keep_indices = logits.topk(num_keep, dim=1).indices

    # Create batch indices for advanced indexing
    batch_indices = torch.arange(batch_size, device=device)
    batch_indices = rearrange(batch_indices, "b -> b 1")

    # Apply dropout to sequence
    seq = seq[batch_indices, keep_indices]

    # Update mask if provided
    if mask is not None:
        seq_counts = mask.sum(dim=-1)
        seq_keep_counts = torch.ceil(seq_counts * keep_probability).int()
        keep_mask = torch.arange(num_keep, device=device) < rearrange(seq_keep_counts, "b -> b 1")

        mask = mask[batch_indices, keep_indices] & keep_mask

    return seq, mask


def cache_fn(f):
    """
    A decorator that caches the results of a function using weak references.

    This decorator implements a cache with weak references to allow garbage collection of unused cached results. It supports optional caching and custom cache keys.

    Args:
        f (Callable): The function to be cached.

    Returns:
        A wrapped version of the input function that implements caching.

    Usage:
        @cache_fn
        def expensive_function(x, y):
            # Expensive computation here
            return result

        # Call with caching
        result1 = expensive_function(1, 2)

        # Call without caching
        result2 = expensive_function(1, 2, _cache=False)

        # Call with a custom cache key
        result3 = expensive_function(1, 2, key="custom_key")
    """
    cache = weakref.WeakValueDictionary()

    def cached_fn(*args, _cache=True, key=None, **kwargs):
        if not _cache:
            return f(*args, **kwargs)

        if key is None:
            key = (args, frozenset(kwargs.items()))

        if key not in cache:
            result = f(*args, **kwargs)
            cache[key] = result
        return cache[key]

    return cached_fn


class FeedForward(BaseFeedForward):
    def __init__(
        self,
        dimension: int,
        multiplier: int = 4,
        hidden_dimension: int = None,
        out_dimension: int = None,
        activation: nn.functional = F.gelu,
    ):
        super().__init__(
            dimension,
            multiplier=multiplier,
            hidden_dimension=hidden_dimension,
            out_dimension=out_dimension,
            activation=activation,
            dropout=0,  # dropout to 0, fixed
            num_layers=1,  # single layer, fixed as well
        )


class Attention(BaseAttention):
    def __init__(self, q_dim: int, context_dim: int = None, heads: int = 8, head_dim: int = 64):
        super().__init__(q_dim, context_dim, heads, head_dim, dropout=0)  # keep dropout to 0 for Perceiver IO

class GQAttention(BaseGQAttention):
    def __init__(self, q_dim: int, context_dim: int = None, n_q_heads: int = 8, n_kv_heads: int = 4, head_dim: int = 64):
        super().__init__(q_dim, context_dim, n_q_heads, n_kv_heads, head_dim, dropout=0)  # keep dropout to 0 for Perceiver IO

class BuiltinGQAttention(BaseBuiltinGQAttention):
    def __init__(self, q_dim: int, context_dim: int = None, n_q_heads: int = 8, n_kv_heads: int = 4, head_dim: int = 64):
        super().__init__(q_dim, context_dim, n_q_heads, n_kv_heads, head_dim, dropout=0)  # keep dropout to 0 for Perceiver IO

