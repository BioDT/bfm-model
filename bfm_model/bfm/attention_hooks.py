from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionWeightCapture:
    """
    Captures attention weights from BuiltinGQAttention modules.

    This class captures the `capture_attention_weights` flag on attention modules and collects the weights after forward passes.
    """

    def __init__(self):
        self.attention_modules = []  # a lsit of (name, module) tuples
        self.attention_weights = {}  # a dict of layer_name: attention_tensor

    def enable_capture(self, model: nn.Module, target_module_name: str = "BuiltinGQAttention", cross_attn_only: bool = True):
        """
        Enable attention weight capture on specified modules.

        Args:
            model: the model (or submodule) to enable capture on
            target_module_name: the name of the attention module class
            cross_attn_only: if True, only capture cross-attention (not self-attention)
        """
        self.attention_modules = []

        for name, module in model.named_modules():
            if module.__class__.__name__ == target_module_name:
                # filter for cross-attention if requested
                should_capture = True
                if cross_attn_only:
                    # only capture the first encoder cross-attention (probably the most important for modality contributions)
                    # skip decoder_cross_attn to reduce memory usage
                    should_capture = "cross_attend_blocks.0.function" in name

                if should_capture:
                    module.capture_attention_weights = True
                    self.attention_modules.append((name, module))
                    print(f"Enabled attention capture on: {name}")

        print(f"Successfully enabled attention capture on {len(self.attention_modules)} modules")

    def disable_capture(self):
        """Disable attention weight capture on all modules."""
        for name, module in self.attention_modules:
            module.capture_attention_weights = False
        print("Disabled attention capture")

    def collect_weights(self) -> Dict[str, torch.Tensor]:
        """
        Collect attention weights from all modules (pops from history).

        Returns:
            a dict of layer_name: attention_tensor
            Shape: {layer_name: [batch, num_heads, num_queries, num_keys]}
        """
        weights = {}
        for idx, (name, module) in enumerate(self.attention_modules):
            # pop from history
            if hasattr(module, 'attention_weights_history') and len(module.attention_weights_history) > 0:
                layer_name = f"layer_{idx}"
                weights[layer_name] = module.attention_weights_history.pop(0)  # Pop oldest
        return weights

    def reset(self):
        """Reset captured attention weights."""
        for name, module in self.attention_modules:
            module.last_attention_weights = None
            if hasattr(module, 'attention_weights_history'):
                module.attention_weights_history.clear()
        self.attention_weights = {}

    def get_attention_weights(self) -> Dict[str, torch.Tensor]:
        """
        Get captured attention weights.

        Returns:
            a dict of layer_name: attention_tensor
        """
        return self.collect_weights()


def build_modality_mapping(batch, patch_size: int = 4) -> Dict[str, Tuple[int, int]]:
    """
    Build a mapping of modality groups to their patch index ranges.

    This one here determines which patches in the concateanted input sequence
    belong to which modality group, based on the encoder's concatenation order.

    Args:
        batch: the input batch containing variable groups
        patch_size: the size of the patches (default: 4)

    Returns:
        a dict of modality_name: (start_idx, end_idx) tuples
        Example: {'surface': (0, 1600), 'climate': (1600, 3200), ...}
    """
    H = len(batch.batch_metadata.latitudes[0])
    W = len(batch.batch_metadata.longitudes[0])
    num_patches_per_group = (H // patch_size) * (W // patch_size)

    modality_mapping = {}
    current_idx = 0

    # order follows encoder.forward() concatenation order
    # 1. surface
    if batch.surface_variables and len(batch.surface_variables) > 0:
        modality_mapping["surface"] = (current_idx, current_idx + num_patches_per_group)
        current_idx += num_patches_per_group

    # 2. edaphic
    if batch.edaphic_variables and len(batch.edaphic_variables) > 0:
        modality_mapping["edaphic"] = (current_idx, current_idx + num_patches_per_group)
        current_idx += num_patches_per_group

    # 3. atmospheric (multi-level) - each level is added separately
    if batch.atmospheric_variables and len(batch.atmospheric_variables) > 0:
        # try to infer number of levels from tensor shape
        sample_var = next(iter(batch.atmospheric_variables.values()))
        if sample_var.ndim == 5:  # [B, T, L, H, W]
            num_levels = sample_var.shape[2]
            for level_idx in range(num_levels):
                modality_mapping[f"atmos_level_{level_idx}"] = (
                    current_idx,
                    current_idx + num_patches_per_group
                )
                current_idx += num_patches_per_group
        else:
            # fallback: treat as single group if shape is unexpected
            modality_mapping["atmospheric"] = (current_idx, current_idx + num_patches_per_group)
            current_idx += num_patches_per_group

    # 4. climate
    if batch.climate_variables and len(batch.climate_variables) > 0:
        modality_mapping["climate"] = (current_idx, current_idx + num_patches_per_group)
        current_idx += num_patches_per_group

    # 5. species
    if batch.species_variables and len(batch.species_variables) > 0:
        modality_mapping["species"] = (current_idx, current_idx + num_patches_per_group)
        current_idx += num_patches_per_group

    # 6. vegetation
    if batch.vegetation_variables and len(batch.vegetation_variables) > 0:
        modality_mapping["vegetation"] = (current_idx, current_idx + num_patches_per_group)
        current_idx += num_patches_per_group

    # 7. land
    if batch.land_variables and len(batch.land_variables) > 0:
        modality_mapping["land"] = (current_idx, current_idx + num_patches_per_group)
        current_idx += num_patches_per_group

    # 8. agriculture
    if batch.agriculture_variables and len(batch.agriculture_variables) > 0:
        modality_mapping["agriculture"] = (current_idx, current_idx + num_patches_per_group)
        current_idx += num_patches_per_group

    # 9. forest
    if batch.forest_variables and len(batch.forest_variables) > 0:
        modality_mapping["forest"] = (current_idx, current_idx + num_patches_per_group)
        current_idx += num_patches_per_group

    # 10. redlist
    if batch.redlist_variables and len(batch.redlist_variables) > 0:
        modality_mapping["redlist"] = (current_idx, current_idx + num_patches_per_group)
        current_idx += num_patches_per_group

    # 11. misc
    if batch.misc_variables and len(batch.misc_variables) > 0:
        modality_mapping["misc"] = (current_idx, current_idx + num_patches_per_group)
        current_idx += num_patches_per_group

    return modality_mapping


def aggregate_attention_by_modality(
    attention_weights: torch.Tensor,
    modality_mapping: Dict[str, Tuple[int, int]],
    aggregation: str = "mean"
) -> Dict[str, float]:
    """
    Aggregate attention weights by modality group.

    Args:
        attention_weights: the attention tensor [batch, num_heads, num_queries, num_keys]
        modality_mapping: a dict of modality_name: (start_idx, end_idx) tuples
        aggregation: the aggregation method ('mean', 'sum', 'max')

    Returns:
        a dict mapping modality names to aggregated attention scores
    """
    # avg across batch and heads
    attn = attention_weights.mean(dim=(0, 1))  # [num_queries, num_keys]

    modality_scores = {}

    for modality_name, (start_idx, end_idx) in modality_mapping.items():
        # extract attention weights for this modality's patches
        modality_attn = attn[:, start_idx:end_idx]

        # aggregate
        if aggregation == "mean":
            score = modality_attn.mean().item()
        elif aggregation == "sum":
            score = modality_attn.sum().item()
        elif aggregation == "max":
            score = modality_attn.max().item()
        else:
            raise ValueError(f"Unknown aggregation method: {aggregation}")

        modality_scores[modality_name] = score

    return modality_scores


def aggregate_attention_spatial(
    attention_weights: torch.Tensor,
    modality_mapping: Dict[str, Tuple[int, int]],
    modality_name: str,
    H: int,
    W: int,
    patch_size: int = 4
) -> np.ndarray:
    """
    Aggregate attention weights for a specific modality into a spatial map.

    Args:
        attention_weights: the attention tensor [batch, num_heads, num_queries, num_keys]
        modality_mapping: Dictionary mapping modalities to patch index ranges
        modality_name: the name of the modality to visualize
        H: the height of the original grid
        W: the width of the original grid
        patch_size: the size of the patches

    Returns:
        a spatial attention map [H_patches, W_patches]
    """
    if modality_name not in modality_mapping:
        raise ValueError(f"Modality {modality_name} not found in mapping")

    start_idx, end_idx = modality_mapping[modality_name]

    # Average across batch, heads, and queries
    attn = attention_weights.mean(dim=(0, 1, 2))  # [num_keys]

    # Extract attention for this modality's patches
    modality_attn = attn[start_idx:end_idx].numpy()

    # Reshape to spatial grid
    H_patches = H // patch_size
    W_patches = W // patch_size
    spatial_attn = modality_attn.reshape(H_patches, W_patches)

    return spatial_attn


def aggregate_spatial_maps_all_modalities(
    attention_weights: torch.Tensor,
    modality_mapping: Dict[str, Tuple[int, int]],
    H: int,
    W: int,
    patch_size: int = 8
) -> Dict[str, np.ndarray]:
    """
    Compute spatial attention maps for ALL modalities efficiently (well, maybe not too).

    This function computes spatial attention patterns for each modality in a single pass,
    which is more efficient than calling aggregate_attention_spatial repeatedly.

    Args:
        attention_weights: the attention tensor [batch, num_heads, num_queries, num_keys]
        modality_mapping: a dict mapping modalities to patch index ranges
        H: the height of the original grid
        W: the width of the original grid
        patch_size: the size of the patches (default: 8)

    Returns:
        a dict mapping modality_name -> spatial_map [H_patches, W_patches]
    """
    # avg across batch, heads, and queries
    attn = attention_weights.mean(dim=(0, 1, 2))  # [num_keys]

    # calculate spatial grid dimensions
    H_patches = H // patch_size
    W_patches = W // patch_size

    spatial_maps = {}

    for modality_name, (start_idx, end_idx) in modality_mapping.items():
        # extract attention for this modality's patches
        modality_attn = attn[start_idx:end_idx].cpu().numpy()

        # reshape to spatial grid
        try:
            spatial_map = modality_attn.reshape(H_patches, W_patches)
            spatial_maps[modality_name] = spatial_map
        except ValueError as e:
            # handle edge case where patch count doesn't match expected grid
            print(f"Warning: Could not reshape {modality_name} attention "
                  f"(expected {H_patches}x{W_patches}={H_patches*W_patches}, "
                  f"got {len(modality_attn)} patches). Skipping.")
            continue

    return spatial_maps


def compute_cross_modality_correlation(
    attention_weights: torch.Tensor,
    modality_mapping: Dict[str, Tuple[int, int]]
) -> np.ndarray:
    """
    Compute correlation matrix between modality attention patterns.

    This function analyzes how similarly the model processes different modalities
    by computing correlations between their attention patterns across all queries.

    Args:
        attention_weights: the attention tensor [batch, num_heads, num_queries, num_keys]
        modality_mapping: a dict mapping modalities to patch index ranges

    Returns:
        a correlation matrix [num_modalities, num_modalities]
        corr[i, j] = correlation between modality i and modality j's attention patterns
    """
    # avg across batch and heads (keep queries dimension)
    attn = attention_weights.mean(dim=(0, 1))  # [num_queries, num_keys]

    modalities = list(modality_mapping.keys())
    num_modalities = len(modalities)

    # extract attention patterns for each modality
    # pattern = how much each query attends to this modality
    modality_patterns = []

    for modality_name in modalities:
        start_idx, end_idx = modality_mapping[modality_name]

        # avg attention to this modality's patches across the patch dimension
        # result: [num_queries] - attention from each query to this modality
        pattern = attn[:, start_idx:end_idx].mean(dim=1).cpu().numpy()
        modality_patterns.append(pattern)

    # compute pairwise correlation matrix
    # stack patterns into matrix [num_modalities, num_queries]
    pattern_matrix = np.stack(modality_patterns, axis=0)

    # compute correlation matrix
    # corr[i, j] = correlation between pattern[i] and pattern[j], obviously
    corr_matrix = np.corrcoef(pattern_matrix)

    return corr_matrix


def compute_modality_contribution_matrix(
    attention_weights_dict: Dict[str, torch.Tensor],
    modality_mapping: Dict[str, Tuple[int, int]]
) -> np.ndarray:
    """
    Compute a matrix of modality contributions across all layers.

    Args:
        attention_weights_dict: a dict of attention weights per layer
        modality_mapping: dict mapping modalities to patch index ranges

    Returns:
        a matrix of shape [num_modalities, num_layers] with contribution scores
    """
    modality_names = list(modality_mapping.keys())
    num_layers = len(attention_weights_dict)

    contribution_matrix = np.zeros((len(modality_names), num_layers))

    for layer_idx, (layer_name, attn_weights) in enumerate(attention_weights_dict.items()):
        modality_scores = aggregate_attention_by_modality(attn_weights, modality_mapping)

        for mod_idx, mod_name in enumerate(modality_names):
            if mod_name in modality_scores:
                contribution_matrix[mod_idx, layer_idx] = modality_scores[mod_name]

    return contribution_matrix, modality_names
