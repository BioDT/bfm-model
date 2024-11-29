import math
from typing import Literal

import torch
from torch import nn

LoRAMode = Literal["single", "all"]


class LowRankAdaptation(nn.Module):
    """
    This class implements the Low-Rank Adaptation (LoRA) technnique, which is a method to fine-tune
    pre-trained models efficiently. LoRA adds a pair of rank decomposition matrices to the weights
    of a linear layer, allowing for parameter-efficient adaptation.

    The LoRA technique works by decomposing the weihgt update into two low-rank matrices A and B,
    such that the update is thus a product of two matrices instead of a full matrix. This significantly reduces the number of
    trainable parametrs while still allowing effective fine-tuning.

    Key components:
    - lora_A: The first low-rank matrix (r x in_features)
    - lora_B: The second low-rank matrix (out_features x r)
    - r: The rank of the decomposition
    - alpha: A scaling factor for the LoRA adaptation
    - dropout: Optional dropuot applied before the LoRA computation

    The forward pass computes x -> x @ A^T @ B^T, scaled by alpha/rank.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 4,
        alpha: int = 1,
        dropout: float = 0.0,
    ):
        """
        Args:
            in_features (int): Number of input features for the LoRA layer.
            out_features (int): Number of output features for the LoRA layer.
            r (int): Rank of the low-rank decomposition. Controls the compression ratio of LoRA. Default: 4
            alpha (int): Scaling factor for the LoRA adaptation. Adjusts the magnitude of the LoRA update. Default: 1
            dropout (float): Dropout probability applied before the LoRA computation. Used for regularization. Default: 0.0

        The LoRA technique adds trainable rank decomposition matrices to the weights of a linear layer.
        This allows for efficient fine-tuning by significantly reducing the number of trainable parameters
        while still enabling effective adaptation of the model.
        """
        super().__init__()
        assert rank > 0, "The rank must be > 0."

        self.lora_alpha = alpha
        self.rank = rank

        self.lora_dropout = nn.Dropout(dropout)

        self.lora_matrix_A = nn.Parameter(torch.empty((self.rank, in_features)))  # the first low-rank matrix A
        self.lora_matrix_B = nn.Parameter(torch.empty((out_features, self.rank)))  # the second low-rank matrix B

        self.lora_scaling_factor = self.lora_alpha / self.rank  # this scaling helps to control the magnitude of the LoRA update

        self.init_weights()

    def init_weights(self) -> None:
        """Initialise the weights of matrices A and B."""
        # Initialise A the same way as the default for `nn.Linear` and set B to zero.
        nn.init.kaiming_uniform_(self.lora_matrix_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_matrix_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the LoRA to the input tensor:
        1. apply dropout to the input
        2. multiply the result with the transpose of lora_A
        3. multiply the result with the transpose of lora_B
        4. scale the final result by the scaling factor

        Args:
            x: Input tensor to the linear layer. Shape: [batch_size, ..., in_features]

        Returns:
            Additive correction tensor for the output of the linear layer.
            Shape: [batch_size, ..., out_features]
            This correction is meant to be added to the output of the original linear layer.
        """
        x_dropped = self.lora_dropout(x)
        lora_correction = x_dropped @ self.lora_matrix_A.T @ self.lora_matrix_B.T
        return self.lora_scaling_factor * lora_correction


class LoRARollout(nn.Module):
    """Per-roll-out-step LoRA finetuning."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: int = 8,
        dropout: float = 0.0,
        max_steps: int = 40,
        mode: LoRAMode = "single",
    ):
        """
        Args:
            in_features (int): Number of input features for the LoRA adaptation.
            out_features (int): Number of output features for the LoRA adaptation.
            rank (int, optional): Rank of the low-rank matrices used in LoRA. Controls the capacity of the adaptation. Default: 8
            alpha (int, optional): Scaling factor for the LoRA update. Higher values lead to larger updates. Default: 8
            dropout (float, optional): Dropout probability applied before the LoRA computation. Used for regularization. Default: 0.0
            max_steps (int, optional): Maximum number of roll-out steps supported. Determines the number of LoRA layers in 'all' mode. Default: 40
            mode (LoRAMode, optional): Determines how LoRA is applied across roll-out steps.
                'single': Uses the same LoRA for all roll-out steps.
                'all': Uses a different LoRA for every roll-out step up to max_steps.
                Default: "single"

        The LoRARollout module allows for per-step LoRA adaptations during model rollout,
        enabling a decent degree of control over the adaptation process at different time steps.
        """
        super().__init__()

        self.max_steps = max_steps
        self.mode = mode

        lora_layers = max_steps if mode == "all" else 1
        self.lora_modules = nn.ModuleList(
            [LowRankAdaptation(in_features, out_features, rank=rank, alpha=alpha, dropout=dropout) for _ in range(lora_layers)]
        )

    def forward(self, x: torch.Tensor, step: int) -> torch.Tensor:
        """
        Apply the LoRA adaptation to the input tensor at the specified step.

        Args:
            x (torch.Tensor): Input tensor to the linear layer.
            step (int): Roll-out step, specifying which LoRA module to use, starting at zero.

        Returns:
            torch.Tensor: Correction tensor for the output of the linear layer.
        """
        # Check if step is non-negative and within the valid range, and also checking the mode being valid
        assert step >= 0, f"Step must be non-negative, found {step}."
        assert self.mode in ["single", "all"], f"Invalid mode: {self.mode}"

        if step >= self.max_steps:
            return torch.zeros_like(x)

        lora_index = (
            0 if self.mode == "single" else min(step, len(self.lora_modules) - 1)
        )  # if single, use the first lora module, if all, use the lora module at the current step
        return self.lora_modules[lora_index](x)
