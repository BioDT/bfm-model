"""
Copyright 2025 (C) TNO. Licensed under the MIT license.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional

class RandomMasking(nn.Module):
    """
    Random masking strategy for patch-based inputs.
    
    Randomly masks a percentage of patches and replaces them with a learnable mask token.
    This forces the model to learn spatial-temporal patterns from context.
    
    Args:
        mask_ratio (float): percentage of patches to mask (0.0 to 1.0)
        embed_dim (int): dimension of patch embeddings
    """
    
    def __init__(self, mask_ratio: float = 0.3, embed_dim: int = 1024):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.embed_dim = embed_dim
        
        # learnable mask token that replaces masked patches
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.normal_(self.mask_token, std=0.02)
    
    def forward(
        self, 
        x: torch.Tensor, 
        force_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Apply random masking to input patches.
        
        Args:
            x: input tensor of shape [batch_size, num_patches, embed_dim]
            force_mask: optional pre-computed mask for consistency across forward passes
            
        Returns:
            masked_x: tensor with masked patches replaced by mask token
            mask: boolean mask indicating which patches were masked (True = masked)
            ids_restore: indices to restore original order after unmasking
        """
        B, N, D = x.shape

        if force_mask is not None:
            # use provided mask for consistency (e.g., between encoder and reconstruction)
            mask = force_mask
            ids_restore = torch.arange(N, device=x.device).unsqueeze(0).expand(B, -1)
        else:
            # calculate number of patches to keep
            len_keep = int(N * (1 - self.mask_ratio))
            
            # generate random noise for each patch
            noise = torch.rand(B, N, device=x.device)
            
            # sort noise to get indices - smallest noise values are kept
            ids_shuffle = torch.argsort(noise, dim=1)
            ids_restore = torch.argsort(ids_shuffle, dim=1)
            
            # generate binary mask: 0 is keep, 1 is remove
            mask = torch.ones([B, N], device=x.device, dtype=torch.bool)
            mask[:, :len_keep] = False
            # unshuffle to get mask in original order
            mask = torch.gather(mask, dim=1, index=ids_restore)
        
        # apply mask: replace masked positions with mask token
        mask_tokens = self.mask_token.expand(B, N, -1)
        masked_x = x.clone()
        masked_x[mask] = mask_tokens[mask]
        
        return masked_x, mask, ids_restore
    
    def get_mask_token(self) -> torch.Tensor:
        return self.mask_token

def create_masking_module(
    masking_type: str = "random",
    mask_ratio: float = 0.3,
    embed_dim: int = 1024,
    **kwargs
) -> nn.Module:
    """
    Factory function to create masking modules.
    
    Args:
        masking_type: type of masking ("random", "block", "tube")
        mask_ratio: percentage of patches to mask
        embed_dim: dimension of embeddings
        **kwargs: additional arguments for specific masking types
        
    Returns:
        masking module instance
    """
    if masking_type == "random":
        return RandomMasking(mask_ratio=mask_ratio, embed_dim=embed_dim)
    else:
        raise ValueError(f"Unknown masking type: {masking_type}")