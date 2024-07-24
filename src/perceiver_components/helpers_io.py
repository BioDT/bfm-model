import weakref
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import nn

from src.perceiver_components.helpers import _GLU
from src.perceiver_components.helpers import Attention as BaseAttention
from src.perceiver_components.helpers import FeedForward as BaseFeedForward
from src.perceiver_components.helpers import PreNorm


def dropout_seq(seq, mask, dropout):
    """
    A helper function to apply structured dropout to a sequence of tokens.
    """
    b, n, *_, device = *seq.shape, seq.device
    logits = torch.randn(b, n, device=device)

    if mask is not None:
        logits = logits.masked_fill(~mask, -torch.finfo(logits.dtype).max)

    keep_prob = 1.0 - dropout
    num_keep = max(1, int(keep_prob * n))
    keep_indices = logits.topk(num_keep, dim=1).indices

    batch_indices = torch.arange(b, device=device)
    batch_indices = rearrange(batch_indices, "b -> b 1")

    seq = seq[batch_indices, keep_indices]

    if mask is not None:
        seq_counts = mask.sum(dim=-1)
        seq_keep_counts = torch.ceil(seq_counts * keep_prob).int()
        keep_mask = torch.arange(num_keep, device=device) < rearrange(seq_keep_counts, "b -> b 1")

        mask = mask[batch_indices, keep_indices] & keep_mask

    return seq, mask


def cache_fn(f):
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
