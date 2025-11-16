"""
Usage:
    streamlit run attention_viewer.py -- --data_dir ./standardize_with_annealed_mask
"""

import argparse
import io
from pathlib import Path
from typing import Dict, List, Tuple

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
import torch
from cartopy.util import add_cyclic_point
from matplotlib.colors import LinearSegmentedColormap


def _get_dir() -> Path:
    """Parse command-line arguments for data directory."""
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--data_dir", default="standardize_with_annealed_mask", type=Path)
    ns, _ = p.parse_known_args()
    return ns.data_dir.resolve()


DATA_DIR = _get_dir()


# MODALITY CONTRIBUTION FUNCTIONS


def aggregate_attention_by_modality(
    attention_weights: torch.Tensor,
    modality_mapping: Dict[str, Tuple[int, int]],
) -> Dict[str, float]:
    """
    Aggregate attention weights by modality group.

    Args:
        attention_weights: the attention tensor [batch, num_heads, num_queries, num_keys]
        modality_mapping: a dict mapping modalities to patch index ranges

    Returns:
        a dict mapping modality names to mean attention scores
    """
    # avg across batch and heads
    attn = attention_weights.mean(dim=(0, 1))  # [num_queries, num_keys]

    modality_scores = {}

    for modality_name, (start_idx, end_idx) in modality_mapping.items():
        # extract attention weights for this modality's patches
        modality_attn = attn[:, start_idx:end_idx]
        score = modality_attn.mean().item()
        modality_scores[modality_name] = score

    return modality_scores


def plot_modality_contribution_bar(
    modality_scores: Dict[str, float],
    title: str = "modality contributions to predictions"
) -> plt.Figure:
    """Create bar chart of modality contributions."""
    fig, ax = plt.subplots(figsize=(10, 6))

    modalities = list(modality_scores.keys())
    scores = list(modality_scores.values())

    # color mapping for different modality types
    colors = []
    for mod in modalities:
        if "surface" in mod:
            colors.append("#e74c3c")  # red
        elif "climate" in mod:
            colors.append("#3498db")  # blue
        elif "species" in mod:
            colors.append("#2ecc71")  # green
        elif "vegetation" in mod or "ndvi" in mod.lower():
            colors.append("#27ae60")  # dark green
        elif "atmos" in mod:
            colors.append("#9b59b6")  # purple
        elif "soil" in mod or "edaphic" in mod:
            colors.append("#d35400")  # orange
        elif "land" in mod:
            colors.append("#f39c12")  # yellow
        else:
            colors.append("#95a5a6")  # gray

    bars = ax.bar(range(len(modalities)), scores, color=colors, alpha=0.7, edgecolor='black')

    ax.set_xticks(range(len(modalities)))
    ax.set_xticklabels(modalities, rotation=45, ha='right')
    ax.set_ylabel("Mean Attention Weight")
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    # add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height,
                f'{height:.4f}',
                ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    return fig


def plot_modality_contribution_across_layers(
    attention_files: List[Path],
    modality_mapping: Dict[str, Tuple[int, int]],
    selected_modalities: List[str] = None
) -> plt.Figure:
    """Plot how modality contributions evolve across layers."""
    # collect contributions across all layers
    layer_contributions = {}

    for file_path in attention_files[:1]:  # use first file as example
        attn_data = torch.load(file_path, map_location='cpu', weights_only=False)

        # handle both formats
        if 'modality_contributions' in attn_data:
            # aggregated format - already have scores per layer
            layer_contributions = attn_data['modality_contributions']
        else:
            # full matrix format - need to compute
            encoder_attn = attn_data['encoder_cross_attn']
            for layer_name, attn_weights in encoder_attn.items():
                scores = aggregate_attention_by_modality(attn_weights, modality_mapping)
                layer_contributions[layer_name] = scores

    # convert to dataframe
    df = pd.DataFrame(layer_contributions).T

    if selected_modalities:
        df = df[[col for col in df.columns if col in selected_modalities]]

    fig, ax = plt.subplots(figsize=(12, 6))
    df.plot(kind='bar', ax=ax, width=0.8, alpha=0.7)

    ax.set_xlabel("encoder layer")
    ax.set_ylabel("mean attention weight")
    ax.set_title("modality contributions across encoder layers", fontsize=14, fontweight='bold')
    ax.legend(title="modality", bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(axis='y', alpha=0.3)
    plt.xticks(rotation=0)
    plt.tight_layout()

    return fig


def plot_modality_heatmap(
    attention_files: List[Path],
    modality_mapping: Dict[str, Tuple[int, int]]
) -> plt.Figure:
    """Create heatmap of modality contributions across windows."""
    # collect contributions across windows
    window_contributions = []
    window_names = []

    for file_path in attention_files:
        attn_data = torch.load(file_path, map_location='cpu', weights_only=False)

        # handle both aggregated and full matrix formats
        if 'modality_contributions' in attn_data:
            # aggregated format - average across layers
            all_scores = list(attn_data['modality_contributions'].values())
            if all_scores:
                avg_scores = {}
                modalities = all_scores[0].keys()
                for mod in modalities:
                    avg_scores[mod] = np.mean([s[mod] for s in all_scores])
                window_contributions.append(avg_scores)
                window_names.append(file_path.stem.replace('attention_window_', ''))
        else:
            # full matrix format
            encoder_attn = attn_data.get('encoder_cross_attn', {})
            all_scores = []
            for layer_name, attn_weights in encoder_attn.items():
                scores = aggregate_attention_by_modality(attn_weights, modality_mapping)
                all_scores.append(scores)

            if all_scores:
                avg_scores = {}
                modalities = all_scores[0].keys()
                for mod in modalities:
                    avg_scores[mod] = np.mean([s[mod] for s in all_scores])
                window_contributions.append(avg_scores)
                window_names.append(file_path.stem.replace('attention_', ''))

    df = pd.DataFrame(window_contributions, index=window_names)

    fig, ax = plt.subplots(figsize=(16, len(window_names) * 0.6 + 2))
    sns.heatmap(df, annot=True, fmt='.4f', cmap='YlOrRd', ax=ax,
                cbar_kws={'label': 'mean attention'},
                annot_kws={'size': 8})  # Smaller font for 13x23 matrix

    ax.set_xlabel("modality group")
    ax.set_ylabel("prediction window")
    ax.set_title("modality contributions across prediction windows", fontsize=14, fontweight='bold')
    plt.tight_layout()

    return fig


# SPATIAL ATTENTION FUNCTIONS


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
        modality_mapping: a dict mapping modalities to patch index ranges
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

    # avg across batch, heads, and queries
    attn = attention_weights.mean(dim=(0, 1, 2))  # [num_keys]

    # extract attention for this modality's patches
    modality_attn = attn[start_idx:end_idx].numpy()

    # reshape to spatial grid
    H_patches = H // patch_size
    W_patches = W // patch_size
    spatial_attn = modality_attn.reshape(H_patches, W_patches)

    return spatial_attn


def plot_spatial_attention_map(
    spatial_attn: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    modality_name: str,
    timestamp: str,
    patch_size: int = 4
) -> plt.Figure:
    """
    plot spatial attention map on a geographic projection

    Args:
        spatial_attn: spatial attention map [H_patches, W_patches]
        lats: latitude coordinates
        lons: longitude coordinates
        modality_name: name of the modality
        timestamp: timestamp string
        patch_size: patch size for upsampling

    Returns:
        matplotlib figure
    """
    # upsample attention map to match original grid (simple repeat)
    spatial_attn_upsampled = np.repeat(np.repeat(spatial_attn, patch_size, axis=0), patch_size, axis=1)

    # ensure shapes match
    H_target, W_target = len(lats), len(lons)
    if spatial_attn_upsampled.shape != (H_target, W_target):
        # resize if needed
        from scipy.ndimage import zoom
        zoom_factors = (H_target / spatial_attn_upsampled.shape[0],
                       W_target / spatial_attn_upsampled.shape[1])
        spatial_attn_upsampled = zoom(spatial_attn_upsampled, zoom_factors, order=1)

    proj = ccrs.PlateCarree()
    fig, ax = plt.subplots(1, 1, figsize=(12, 6), subplot_kw=dict(projection=proj))

    # add cyclic point for smooth visualization
    attn_cyc, lon_cyc = add_cyclic_point(spatial_attn_upsampled, coord=lons)

    # custom colormap (white -> yellow -> red for attention)
    cmap = LinearSegmentedColormap.from_list('attention', ['white', 'yellow', 'orange', 'red'])

    mesh = ax.pcolormesh(lon_cyc, lats, attn_cyc, cmap=cmap, transform=proj, vmin=0, vmax=spatial_attn_upsampled.max())
    ax.add_feature(cfeature.COASTLINE, lw=0.5)
    ax.add_feature(cfeature.BORDERS, lw=0.3, alpha=0.5)

    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', linestyle='--', alpha=0.5)
    gl.top_labels = gl.right_labels = False
    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")

    cbar = fig.colorbar(mesh, ax=ax, shrink=0.6, label='attention weight')
    ax.set_title(f"spatial attention map: {modality_name}\ntimestamp: {timestamp}",
                 fontsize=12, fontweight='bold')

    plt.tight_layout()
    return fig


# CROSS-MODALITY ANALYSIS FUNCTIONS


def compute_modality_correlation(
    attention_weights: torch.Tensor,
    modality_mapping: Dict[str, Tuple[int, int]]
) -> Tuple[np.ndarray, List[str]]:
    """
    Compute correlation matrix between modalities based on co-attention patterns.

    Args:
        attention_weights: the attention tensor [batch, num_heads, num_queries, num_keys]
        modality_mapping: a dict mapping modalities to patch index ranges

    Returns:
        a correlation matrix and a list of modality names
    """
    # avg across batch and heads
    attn = attention_weights.mean(dim=(0, 1))  # [num_queries, num_keys]

    modalities = list(modality_mapping.keys())
    num_modalities = len(modalities)
    corr_matrix = np.zeros((num_modalities, num_modalities))

    # extract attention patterns for each modality
    modality_patterns = []
    for modality_name, (start_idx, end_idx) in modality_mapping.items():
        # attention pattern: how much each query attends to this modality
        pattern = attn[:, start_idx:end_idx].mean(dim=1).numpy()  # [num_queries]
        modality_patterns.append(pattern)

    # compute pairwise correlations
    for i, pattern_i in enumerate(modality_patterns):
        for j, pattern_j in enumerate(modality_patterns):
            corr_matrix[i, j] = np.corrcoef(pattern_i, pattern_j)[0, 1]

    return corr_matrix, modalities


def plot_modality_correlation_heatmap(
    corr_matrix: np.ndarray,
    modality_names: List[str]
) -> plt.Figure:
    """Plot correlation heatmap between modalities."""
    fig, ax = plt.subplots(figsize=(14, 12))

    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                xticklabels=modality_names, yticklabels=modality_names,
                center=0, vmin=-1, vmax=1, ax=ax, square=True,
                annot_kws={'size': 7})  # Smaller font for 23x23 matrix

    ax.set_title("cross-modality attention correlation", fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


# STREAMLIT APP STUFF 


st.set_page_config(page_title="BFM encoder attention viewer", layout="wide")

st.sidebar.title("BFM encoder attention viewer")
st.sidebar.markdown("visualize modality contributions through attention analysis")

# load attention files
attention_files = sorted(DATA_DIR.glob("attention_window_*.pt"))
if not attention_files:
    st.sidebar.error(f"no attention_window_*.pt files found in {DATA_DIR}")
    st.stop()

st.sidebar.success(f"found {len(attention_files)} attention files")

# file selection
file_sel = st.sidebar.selectbox("select attention file", [f.name for f in attention_files])
attn_file_path = DATA_DIR / file_sel

# load attention data
attn_data = torch.load(attn_file_path, map_location='cpu', weights_only=False)

# handle both old (full matrix) and new (aggregated) formats
if 'modality_contributions' in attn_data:
    # new compact format with pre-aggregated scores
    modality_contributions = attn_data['modality_contributions']
    encoder_attn = None  # no full attention matrix
    use_aggregated = True
else:
    # old format with full attention matrices
    encoder_attn = attn_data.get('encoder_cross_attn', {})
    modality_contributions = None
    use_aggregated = False

modality_mapping = attn_data['modality_mapping']
metadata = attn_data['metadata']

# display metadata
st.sidebar.markdown("### file metadata")
timestamp = str(metadata['timestamp']) if metadata['timestamp'] is not None else "N/A"
st.sidebar.text(f"Timestamp: {timestamp}")

if use_aggregated:
    st.sidebar.text(f"Layers: {len(modality_contributions)}")
    st.sidebar.text(f"Modalities: {len(modality_mapping)}")
    st.sidebar.info("using compact aggregated format")
    layer_names = list(modality_contributions.keys())
else:
    st.sidebar.text(f"Layers captured: {len(encoder_attn) if encoder_attn else 0}")
    st.sidebar.text(f"Modalities: {len(modality_mapping)}")
    if not encoder_attn or len(encoder_attn) == 0:
        st.error("no attention data found!")
        st.stop()
    layer_names = list(encoder_attn.keys())

# layer selection
selected_layer = st.sidebar.selectbox("Select encoder layer", layer_names)

# modality selection
all_modalities = list(modality_mapping.keys())
selected_modalities_multi = st.sidebar.multiselect(
    "Select modalities for comparison",
    all_modalities,
    default=all_modalities[:min(5, len(all_modalities))]
)

# tabs
st.header(f"Attention Analysis: {file_sel}")
tab1, tab2, tab3 = st.tabs([
    "modality contributions",
    "spatial attention maps",
    "cross-modality analysis"
])

# TAB 1: MODALITY CONTRIBUTIONS
with tab1:
    st.subheader("modality contribution summary")
    st.markdown("""This tab shows which modality groups (climate, species, NDVI, etc.) contribute most to predictions. Higher attention weights indicate stronger contribution.""")

    # get modality scores (either from aggregated data or compute from full matrix)
    if use_aggregated:
        modality_scores = modality_contributions[selected_layer]
    else:
        attn_weights = encoder_attn[selected_layer]
        modality_scores = aggregate_attention_by_modality(attn_weights, modality_mapping)

    # bar chart
    st.markdown(f"#### Modality contributions (selected layer: {selected_layer})")
    fig_bar = plot_modality_contribution_bar(modality_scores, title=f"Layer: {selected_layer}")
    st.pyplot(fig_bar)

    # download button
    buf = io.BytesIO()
    fig_bar.savefig(buf, dpi=300, format='png', bbox_inches='tight')
    st.download_button(
        "download bar chart (300 dpi)",
        buf.getvalue(),
        file_name=f"modality_contributions_{selected_layer}.png",
        mime="image/png"
    )

    st.markdown("---")

    # across layers
    if len(selected_modalities_multi) > 0:
        st.markdown("#### contributions across encoder layers")
        fig_layers = plot_modality_contribution_across_layers(
            [attn_file_path],
            modality_mapping,
            selected_modalities_multi
        )
        st.pyplot(fig_layers)

    st.markdown("---")

    # Heatmap across windows
    st.markdown("#### contributions across prediction windows")
    fig_heatmap = plot_modality_heatmap(attention_files, modality_mapping)
    st.pyplot(fig_heatmap)

    # Quantitative summary
    st.markdown("#### quantitative summary")
    df_scores = pd.DataFrame(list(modality_scores.items()), columns=['modality', 'attention weight'])
    df_scores = df_scores.sort_values('attention weight', ascending=False)
    df_scores['percentage'] = (df_scores['attention weight'] / df_scores['attention weight'].sum() * 100).round(2)
    st.dataframe(df_scores.style.format({'attention weight': '{:.6f}', 'percentage': '{:.2f}%'}))


# TAB 2: SPATIAL ATTENTION MAPS
with tab2:
    st.subheader("spatial attention maps")
    st.markdown("""Visualize which geographic regions the model attends to for each modality. Red areas indicate high attention (important regions for prediction).""")

    # check if pre-computed spatial maps are available
    if 'spatial_attention_maps' in attn_data:
        # use pre-computed spatial maps
        spatial_maps = attn_data['spatial_attention_maps']
        st.success("using pre-computed spatial attention maps!")

        if metadata['latitudes'] is None or metadata['longitudes'] is None:
            st.warning("no geographic metadata available for spatial visualization")
        else:
            lats = np.array(metadata['latitudes'])
            lons = np.array(metadata['longitudes'])

            spatial_modality = st.selectbox("select modality for spatial attention", all_modalities)

            if spatial_modality in spatial_maps:
                spatial_attn = spatial_maps[spatial_modality]

                try:
                    fig_spatial = plot_spatial_attention_map(
                        spatial_attn, lats, lons, spatial_modality, timestamp, patch_size=8
                    )
                    st.pyplot(fig_spatial)

                    buf = io.BytesIO()
                    fig_spatial.savefig(buf, dpi=300, format='png', bbox_inches='tight')
                    st.download_button(
                        "download spatial map (300 dpi)",
                        buf.getvalue(),
                        file_name=f"spatial_attention_{spatial_modality}.png",
                        mime="image/png"
                    )

                    st.markdown("#### spatial statistics")
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("max attention", f"{spatial_attn.max():.6f}")
                    col2.metric("mean attention", f"{spatial_attn.mean():.6f}")
                    col3.metric("min attention", f"{spatial_attn.min():.6f}")
                    col4.metric("std dev", f"{spatial_attn.std():.6f}")

                except Exception as e:
                    st.error(f"could not generate spatial attention map: {e}")
                    import traceback
                    st.code(traceback.format_exc())
            else:
                st.warning(f"spatial map for {spatial_modality} not found in file")

    elif use_aggregated:
        # old compact format (no spatial maps) - development remnant, let it be here for now
        st.info("spatial attention maps not available in this file")
        st.markdown("this file uses the old compact format with only aggregated scores")
        st.markdown("re-run data generation to create hybrid format files with spatial maps")

    else:
        # full matrix format (backward compatibility)
        st.info("computing spatial maps from full attention matrix")

        if metadata['latitudes'] is None or metadata['longitudes'] is None:
            st.warning("no geographic metadata available for spatial visualization")
        else:
            lats = np.array(metadata['latitudes'])
            lons = np.array(metadata['longitudes'])
            H, W = len(lats), len(lons)

            spatial_modality = st.selectbox("select modality for spatial attention", all_modalities)
            attn_weights = encoder_attn[selected_layer]

            try:
                spatial_attn = aggregate_attention_spatial(
                    attn_weights, modality_mapping, spatial_modality, H, W, patch_size=8
                )
                fig_spatial = plot_spatial_attention_map(
                    spatial_attn, lats, lons, spatial_modality, timestamp, patch_size=8
                )
                st.pyplot(fig_spatial)

                buf = io.BytesIO()
                fig_spatial.savefig(buf, dpi=300, format='png', bbox_inches='tight')
                st.download_button(
                    "download spatial map (300 dpi)",
                    buf.getvalue(),
                    file_name=f"spatial_attention_{spatial_modality}_{selected_layer}.png",
                    mime="image/png"
                )

                st.markdown("#### spatial statistics")
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("max attention", f"{spatial_attn.max():.6f}")
                col2.metric("mean attention", f"{spatial_attn.mean():.6f}")
                col3.metric("min attention", f"{spatial_attn.min():.6f}")
                col4.metric("std dev", f"{spatial_attn.std():.6f}")

            except Exception as e:
                st.error(f"could not generate spatial attention map: {e}")
                import traceback
                st.code(traceback.format_exc())


# TAB 3: CROSS-MODALITY ANALYSIS
with tab3:
    st.subheader("cross-modality analysis")
    st.markdown("""Analyze relationships between modalities based on co-attention patterns. high correlation suggests modalities are processed similarly by the model.""")

    # check if pre-computed correlation matrix is available
    if 'modality_correlation' in attn_data:
        # use pre-computed correlation matrix
        corr_matrix = attn_data['modality_correlation']
        modality_names = list(modality_mapping.keys())
        st.success("using pre-computed modality correlation matrix!")

        st.markdown("#### attention correlation between modalities")
        fig_corr = plot_modality_correlation_heatmap(corr_matrix, modality_names)
        st.pyplot(fig_corr)

        buf = io.BytesIO()
        fig_corr.savefig(buf, dpi=300, format='png', bbox_inches='tight')
        st.download_button(
            "download correlation heatmap (300 dpi)",
            buf.getvalue(),
            file_name="modality_correlation.png",
            mime="image/png"
        )

        st.markdown("---")
        st.markdown("#### pairwise correlation lookup")
        st.markdown("view correlation coefficient between any two modalities:")

        col1, col2 = st.columns(2)
        with col1:
            mod1 = st.selectbox("Modality 1", all_modalities, key="mod1")
        with col2:
            mod2 = st.selectbox("Modality 2", [m for m in all_modalities if m != mod1], key="mod2")

        if mod1 and mod2:
            idx1 = modality_names.index(mod1)
            idx2 = modality_names.index(mod2)
            corr_value = corr_matrix[idx1, idx2]

            st.metric(
                f"Correlation: {mod1} ↔ {mod2}",
                f"{corr_value:.4f}",
                delta=None,
                help="Values close to 1 indicate strong positive correlation (similar processing), close to -1 indicate negative correlation, close to 0 indicate independence."
            )

            # Interpretation
            if corr_value > 0.7:
                st.success(f"strong positive correlation: {mod1} and {mod2} are processed very similarly by the model.")
            elif corr_value > 0.3:
                st.info(f"moderate positive correlation: {mod1} and {mod2} share some processing patterns.")
            elif corr_value > -0.3:
                st.warning(f"weak correlation: {mod1} and {mod2} are processed relatively independently.")
            else:
                st.error(f"negative correlation: {mod1} and {mod2} have opposing attention patterns.")

    elif use_aggregated:
        # old compact format (no correlation matrix)
        st.info("cross-modality correlation matrix not available in this file")
        st.markdown("this file uses the old compact format with only aggregated scores")
        st.markdown("re-run data generation to create hybrid format files with correlation analysis")

    else:
        # full matrix format (backward compatibility)
        st.info("computing correlation matrix from full attention matrix")

        attn_weights = encoder_attn[selected_layer]
        corr_matrix, modality_names = compute_modality_correlation(attn_weights, modality_mapping)

        st.markdown("#### attention correlation between modalities")
        fig_corr = plot_modality_correlation_heatmap(corr_matrix, modality_names)
        st.pyplot(fig_corr)

        buf = io.BytesIO()
        fig_corr.savefig(buf, dpi=300, format='png', bbox_inches='tight')
        st.download_button(
            "download correlation heatmap (300 dpi)",
            buf.getvalue(),
            file_name=f"modality_correlation_{selected_layer}.png",
            mime="image/png"
        )

        st.markdown("---")
        st.markdown("#### pairwise attention analysis")
        st.markdown("Compare attention patterns between two modalities:")

        col1, col2 = st.columns(2)  
        with col1:
            mod1 = st.selectbox("modality 1", all_modalities, key="mod1")
        with col2:
            mod2 = st.selectbox("modality 2", [m for m in all_modalities if m != mod1], key="mod2")

        if mod1 and mod2:
            attn = attn_weights.mean(dim=(0, 1))
            start1, end1 = modality_mapping[mod1]
            start2, end2 = modality_mapping[mod2]

            pattern1 = attn[:, start1:end1].mean(dim=1).numpy()
            pattern2 = attn[:, start2:end2].mean(dim=1).numpy()

            fig_scatter, ax = plt.subplots(figsize=(8, 6))
            ax.scatter(pattern1, pattern2, alpha=0.5)
            ax.set_xlabel(f"{mod1} attention")
            ax.set_ylabel(f"{mod2} attention")
            ax.set_title(f"attention pattern comparison: {mod1} vs {mod2}")
            ax.grid(True, alpha=0.3)

            corr_coef = np.corrcoef(pattern1, pattern2)[0, 1]
            ax.text(0.05, 0.95, f"correlation: {corr_coef:.3f}",
                    transform=ax.transAxes, fontsize=12,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

            plt.tight_layout()
            st.pyplot(fig_scatter)
