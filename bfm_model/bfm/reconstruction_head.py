import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple


class ReconstructionHead(nn.Module):
    """
    Reconstruction head for masked token reconstruction.
    
    Takes encoded representations and reconstructs the original masked patches.
    Uses a simple transformer-based architecture for efficiency.
    
    Args:
        embed_dim (int): dim of encoded representations
        decoder_embed_dim (int): dim of decoder embeddings
        decoder_depth (int): nr of transformer layers in decoder
        decoder_num_heads (int): nr of attention heads
        patch_size (int): unused here (kept for compatibility)
        in_channels (int): unused here (kept for compatibility)
        mlp_ratio (float): ratio of MLP hidden dim to embedding dim
    """
    
    def __init__(
        self,
        embed_dim: int = 1024,
        decoder_embed_dim: int = 512,
        decoder_depth: int = 2,
        decoder_num_heads: int = 8,
        patch_size: int = 4,
        in_channels: int = 1,  # will be dynamically set based on variable groups
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.decoder_embed_dim = decoder_embed_dim
        self.patch_size = patch_size
        self.in_channels = in_channels
        
        # for encoder dim to decoder dim
        self.embed_proj = nn.Linear(embed_dim, decoder_embed_dim)
        
        # decoder blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=decoder_embed_dim,
                num_heads=decoder_num_heads,
                mlp_ratio=mlp_ratio,
            )
            for _ in range(decoder_depth)
        ])
        
        
        self.norm = nn.LayerNorm(decoder_embed_dim) # final norm
        # project back to token embedding dim D
        self.pred_proj = nn.Linear(decoder_embed_dim, embed_dim)
        self.group_projections = nn.ModuleDict() # optional group-specific projections -> also to D
    
    def add_group_projection(self, group_name: str, num_channels: int):
        """
        Add a projection head for a specific variable group.
        
        Args:
            group_name: name of the var group
            num_channels: nr of channels in this group
        """
        # in embedding-reconstruction mode, every head outputs D
        self.group_projections[group_name] = nn.Linear(self.decoder_embed_dim, self.embed_dim)
    
    def forward(
        self, 
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        group_name: Optional[str] = None
    ) -> torch.Tensor:
        """
        Reconstruct masked token embeddings from encoded representations.
        
        Args:
            x: encoded representations [batch_size, num_patches, embed_dim]
            mask: boolean mask indicating which patches to reconstruct
            group_name: optional variable group name for group-specific projection
            
        Returns: 
            reconstructed embeddings [batch_size, num_tokens or num_masked, embed_dim]
        """
        # project to decoder dimension
        x = self.embed_proj(x)
        
        # apply transformer blocks
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)  # final norm
        
        # project to embedding space D
        if group_name and group_name in self.group_projections:
            x = self.group_projections[group_name](x)
        else:
            x = self.pred_proj(x)
        
        # if mask provided, only return masked patches
        if mask is not None:
            x = x[mask]
        
        return x


class TransformerBlock(nn.Module):
    """
    Basic transformer block with self-attention and MLP.
    
    Args:
        dim (int): input/output dimension
        num_heads (int): number of attention heads
        mlp_ratio (float): ratio of MLP hidden dim to embedding dim
        drop (float): dropout rate
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
    ):
        super().__init__()
        
        # MHSA
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            dim,
            num_heads,
            dropout=drop,
            batch_first=True
        )
        
        # MLP
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through transformer block.
        
        Args:
            x: input tensor [batch_size, seq_len, dim]
            
        Returns:
            output tensor [batch_size, seq_len, dim]
        """
        # self-attention with residual
        attn_out, _ = self.attn(
            self.norm1(x),
            self.norm1(x),
            self.norm1(x)
        )
        x = x + attn_out
        
        # mlp with residual
        x = x + self.mlp(self.norm2(x))
        
        return x


class MultiGroupReconstructionHead(nn.Module):
    """
    Reconstruction head that handles multiple variable groups.
    
    Each variable group may have different numbers of channels, so we need separate projection heads for each group.
    
    Args:
        embed_dim (int): dim of encoded representations
        decoder_embed_dim (int): dim of decoder embeddings
        decoder_depth (int): nr of transformer layers
        decoder_num_heads (int): nr of attention heads
        patch_size (int): size of patch
        variable_groups (Dict[str, int]): mapping of group names to channel counts
    """
    
    def __init__(
        self,
        embed_dim: int = 1024,
        decoder_embed_dim: int = 512,
        decoder_depth: int = 2,
        decoder_num_heads: int = 8,
        patch_size: int = 4,
        variable_groups: Optional[Dict[str, int]] = None,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.decoder_embed_dim = decoder_embed_dim
        self.patch_size = patch_size
        
        # shared encoder to decoder projection
        self.embed_proj = nn.Linear(embed_dim, decoder_embed_dim)
        
        # shared transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=decoder_embed_dim,
                num_heads=decoder_num_heads,
                mlp_ratio=mlp_ratio,
            )
            for _ in range(decoder_depth)
        ])
        self.norm = nn.LayerNorm(decoder_embed_dim)  # shared normalization
        
        # group-specific projection heads
        self.group_projections = nn.ModuleDict()
        if variable_groups:
            for group_name, num_channels in variable_groups.items():
                self.group_projections[group_name] = nn.Linear(
                    decoder_embed_dim,
                    patch_size * patch_size * num_channels
                )
        
        # default projection for unknown groups
        self.default_proj = nn.Linear(
            decoder_embed_dim,
            patch_size * patch_size  # single channel default
        )
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        group_embeddings: Optional[Dict[str, Tuple[torch.Tensor, torch.Tensor]]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Reconstruct masked patches for multiple variable groups.
        
        Args:
            x: encoded representations [batch_size, num_patches, embed_dim]
            mask: boolean mask for all patches
            group_embeddings: dict mapping group names to (embeddings, group_mask) tuples
            
        Returns:
            dict mapping group names to reconstructed patches
        """
        # shared processing
        x = self.embed_proj(x)
        
        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x)
        
        # group-specific reconstructions
        reconstructions = {}
        
        if group_embeddings:
            for group_name, (group_embed, group_mask) in group_embeddings.items():
                if group_name in self.group_projections:
                    group_recon = self.group_projections[group_name](x)
                else:
                    group_recon = self.default_proj(x)
                
                # use mask if provided
                if mask is not None and group_mask is not None:
                    combined_mask = mask & group_mask
                    if combined_mask.any():
                        reconstructions[group_name] = group_recon[combined_mask]
                else:
                    reconstructions[group_name] = group_recon
        else:
            # if no group info, use default projection
            default_recon = self.default_proj(x)
            if mask is not None:
                default_recon = default_recon[mask]
            reconstructions["default"] = default_recon
        
        return reconstructions