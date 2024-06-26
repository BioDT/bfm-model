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


# TODO:
# 1) Attention
# 2) Perceiver
