"""
Copyright (C) 2025 TNO, The Netherlands. All rights reserved.
"""
import math
from typing import Literal, Optional
import torch
from torch import nn

__all__ = ["VeRA", "VeRARollout", "VeRAMode"]

VeRAMode = Literal["single", "all"]


class VeRA(nn.Module):
    """
    Vector-based Random-matrix Adaptation (VeRA) for nn.Linear
    https://arxiv.org/abs/2310.11454
    ΔW = diag(λ_d) · (Aᵀ ⊙ λ_b) · Bᵀ
    where  A∈R^{r x in}, B∈R^{out x r} are frozen random matrices,
           λ_b∈R^{r}, λ_d∈R^{out} are trainable vectors,
           ⊙ is element-wise multiplication applied with broadcasting
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 256,
        dropout: float = 0.0,
        d_initial: float = 0.1,
        shared_A: Optional[torch.Tensor] = None,
        shared_B: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.r = r
        self.dropout = nn.Dropout(dropout)

        # frozen random projections (shared or layer‑local)
        if shared_A is None:
            A = torch.empty(r, in_features)
            nn.init.kaiming_uniform_(A, a=math.sqrt(5))
            self.register_buffer("vera_A", A, persistent=False)
        else:  # slice to the current input size
            self.register_buffer("vera_A", shared_A[:, :in_features], persistent=False)

        if shared_B is None:
            B = torch.empty(out_features, r)
            nn.init.kaiming_uniform_(B, a=math.sqrt(5))
            self.register_buffer("vera_B", B, persistent=False)
        else:
            self.register_buffer("vera_B", shared_B[:out_features, :], persistent=False)

        # trainable scaling vectors
        self.lambda_b = nn.Parameter(torch.ones(r)) # length‑r
        self.lambda_d = nn.Parameter(torch.full((out_features,), d_initial)) # length‑out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, *, in) · (in, r)  ->  (B, *, r)
        x_dropped = self.dropout(x)
        h = x_dropped @ self.vera_A.T # random projection
        h = h * self.lambda_b # scale inside the sub‑space
        # (B, *, r) · (r, out) ->  (B, *, out)
        delta = h @ self.vera_B.T
        delta = delta * self.lambda_d # per‑output gating
        return delta


class VeRARollout(nn.Module):
    """
    Per-roll-out-step VeRA
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 256,
        dropout: float = 0.0,
        d_initial: float = 0.1,
        max_steps: int = 40,
        mode: VeRAMode = "single",
    ):
        super().__init__()
        self.mode = mode
        self.max_steps = max_steps

        layers = max_steps if mode == "all" else 1
        self.veras = nn.ModuleList(
            [
                VeRA(
                    in_features,
                    out_features,
                    r=r,
                    dropout=dropout,
                    d_initial=d_initial,
                )
                for _ in range(layers)
            ]
        )

    def forward(self, x: torch.Tensor, step: int) -> torch.Tensor:
        if step >= self.max_steps:
            return 0
        if self.mode == "single":
            return self.veras[0](x)
        elif self.mode == "all":
            return self.veras[step](x)
        else:
            raise ValueError(f"Unknown mode {self.mode}")