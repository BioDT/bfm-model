"""
Copyright 2025 (C) TNO. Licensed under the MIT license.

Batch-comparison viewer for spatiotemporal rollouts

** UNDER DEVELOPMENT
"""

import argparse
import re
from collections import defaultdict
from math import atan2, cos, radians, sin, sqrt
from pathlib import Path
from types import SimpleNamespace
from typing import DefaultDict, Dict, List
import pandas as pd
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader
import hydra
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import shapely.vectorized as sv
import streamlit as st
import torch
from cartopy.util import add_cyclic_point
from hydra.core.global_hydra import GlobalHydra
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.spatial import ConvexHull
import scipy.ndimage as ndi
from bfm_model.bfm.dataloader_monthly import LargeClimateDataset

GlobalHydra.instance().clear()

hydra.initialize(config_path="", version_base=None)

cfg = hydra.compose(config_name="configs/train_config.yaml", overrides=["+data.scaling.enabled=True"])


scaling = SimpleNamespace(
    enabled=True,
    stats_path="/bfm/bfm-model/batch_statistics/monthly_batches_stats_splitted_channels.json",
    mode="normalize",
)
print(scaling.stats_path)

# TODO ADD: Manual input
data_dir = "/Users/azd/Desktop/git_projects/bfm/bfm-model/rollout_monthly"

def crop_variables(variables, new_H, new_W, **kwargs):
    """Fallback: crop tensors or dicts to (new_H,new_W)."""
    if isinstance(variables, dict):
        return {k: v[..., :new_H, :new_W] for k, v in variables.items()}
    return variables[..., :new_H, :new_W]


test_dataset = LargeClimateDataset(data_dir=data_dir, scaling_settings=scaling)


def _parse_args():
    p = argparse.ArgumentParser(add_help=False)
    # OrIGINAL
    p.add_argument("--gt_dir", type=Path, default="/folder_with_the_produced_rollouts")
    p.add_argument("--crop", nargs=2, type=int, default=[160, 280], metavar=("H", "W"))
    return p.parse_known_args()[0]


ARGS = _parse_args()
GT_DIR, RO_DIR = ARGS.gt_dir.resolve(), ARGS.rollout_dir.resolve()
NEW_H, NEW_W = ARGS.crop

# File matching
# -----------------------------------------------------------------------------
DATE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})")


def _date(fn: str) -> str | None:
    m = DATE_RE.search(fn)
    return m.group(1) if m else None


def _pair_files(gt_dir: Path, ro_dir: Path):
    gt_files = {_date(f.name): f for f in gt_dir.glob("batch_*.pt")}
    ro_files = {_date(f.name): f for f in ro_dir.glob("prediction_*.pt")}
    return sorted([(gt_files[d], ro_files[d]) for d in gt_files if d in ro_files], key=lambda p: p[0].name)


PAIRS = _pair_files(GT_DIR, RO_DIR)
if not PAIRS:
    st.error("No matching files.")
    st.stop()

# Sidebar global settings
# ----------------------------------------------------------------------------
st.sidebar.title("Batch comparison - timeline mode")
metric_choice = st.sidebar.radio("Error metric", ["RMSE", "MAE"], index=0)

# Helpers
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def build_land_mask(lon_grid: np.ndarray, lat_grid: np.ndarray) -> np.ndarray:
    """
    Returns a boolean mask (HxW) that is True over land, using Natural Earth
    110 m land polygons.  Works with 2-D lon/lat grids.
    """
    reader = shpreader.Reader(shpreader.natural_earth(resolution="110m", category="physical", name="land"))
    mask = np.zeros(lon_grid.shape, dtype=bool)

    for geom in reader.geometries():
        mask |= sv.contains(geom, lon_grid, lat_grid)

    return mask


def _crop_batch(batch: Dict[str, Dict[str, torch.Tensor]]):
    for slot, vdict in list(batch.items()):
        if slot.endswith("_variables"):
            batch[slot] = crop_variables(vdict, NEW_H, NEW_W, handle_nans=False)
    return batch


def _align(pred: torch.Tensor, obs: torch.Tensor):
    """Return shape-aligned tensors (adds batch dim; matches channels)."""
    if obs.ndim == pred.ndim - 1:
        obs = obs.unsqueeze(0)
    if pred.ndim == 5 and obs.ndim == 4:
        pred = pred[:, :, 0]
    elif pred.ndim == 4 and obs.ndim == 5:
        obs = obs[:, :, 0]
    return pred, obs


def _intify_species_keys(batch: Dict[str, Dict[str, torch.Tensor]]) -> Dict[str, Dict[str, torch.Tensor]]:
    """Return a copy of batch where species keys are ints, not strings."""
    if "species_variables" not in batch:
        return batch
    new = batch.copy()
    sp_dict = {}
    for k, v in batch["species_variables"].items():
        try:  # '1340361' -> 1340361
            k_int = int(k)
        except (TypeError, ValueError):
            k_int = k  # keep non-numeric IDs unchanged
        sp_dict[k_int] = v
    new["species_variables"] = sp_dict
    return new


def _spatial_mean_error(pred: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
    """Return a 1-D (T,) tensor: error averaged over batch + channel + space."""
    pred, obs = _align(pred, obs)
    err = torch.abs(pred - obs) if metric_choice == "MAE" else (pred - obs) ** 2
    if metric_choice == "RMSE":
        err = torch.sqrt(err)
    err = err.mean(dim=(-2, -1))
    if err.ndim == 3:
        err = err.mean(dim=(0, 2))
    elif err.ndim == 2:
        err = err.mean(dim=0)
    return err


def _file_error(pred: torch.Tensor, obs: torch.Tensor) -> float:
    """Spatial+temporal mean error for an entire batch window."""
    err_ts = _spatial_mean_error(pred, obs)
    return float(err_ts.cpu().numpy()[0])  # (T,)


def monthly_acc(pred: torch.Tensor,
                obs:  torch.Tensor,
                clim: torch.Tensor,
                months: np.ndarray) -> np.ndarray:
    """
    pred, obs : (T, H, W) tensors   - anomalies are pred - clim[month]
    clim: (12, H, W) tensor   - monthly climatology built from OBS
    months: (T,) 0-based month index (0-Jan … 11-Dec) for each frame
    Returns: (T,) ACC per month   (NaN if variance = 0)
    """
    acc = []
    for t in range(pred.shape[0]):
        m = months[t]
        ao = obs[t]  - clim[m]
        ap = pred[t] - clim[m]
        mask = (~torch.isnan(ao)) & (~torch.isnan(ap))
        ao, ap = ao[mask], ap[mask]
        num = (ao * ap).sum()
        den = torch.sqrt((ao**2).sum()) * torch.sqrt((ap**2).sum())
        acc.append(float(num / (den + 1e-12)))
    return np.asarray(acc)

# Load and process all matched file pairs
# ----------------------------------------------------------------------------
var_groups: Dict[str, List[str]] = {}
err_pair: DefaultDict[str, List[float]] = defaultdict(list)
series_err: DefaultDict[str, List[np.ndarray]] = defaultdict(list)
occ_pair_gt: DefaultDict[str, List[float]] = defaultdict(list)
occ_pair_pr: DefaultDict[str, List[float]] = defaultdict(list)
hov_err: DefaultDict[str, List[torch.Tensor]] = defaultdict(list)
spec_ts_gt, spec_ts_pr = defaultdict(list), defaultdict(list)

# "surface_variables",
# "edaphic_variables",
# "atmospheric_variables",
# "climate_variables",
# "species_variables",
# "vegetation_variables",
# "land_variables",
# "agriculture_variables",
# "forest_variables",
# "redlist_variables",
# "misc_variables",

for idx, (gt_path, ro_path) in enumerate(PAIRS):
    GT = torch.load(gt_path, map_location="cpu")
    RO = torch.load(ro_path, map_location="cpu")
    PRED_dict = RO[0]._asdict()
    print(type(RO), len(PRED_dict))

    GT = _crop_batch(GT)
    lons = np.array(GT["batch_metadata"]["longitudes"])[:NEW_W]
    lats = np.array(GT["batch_metadata"]["latitudes"])[:NEW_H]
    print(f"LATS {len(lats)} and \n LONS {len(lons)}")

    PRED = _intify_species_keys(test_dataset.scale_batch(PRED_dict, direction="original"))
    if not var_groups:
        var_groups = {slot: sorted(vdict.keys()) for slot, vdict in GT.items() if slot.endswith("species_variables")} #TODO Change the variabler group here!!!
    test_var = list(var_groups[next(iter(var_groups))])[0]
    for slot, vars_ in var_groups.items():
        for v in vars_:
            gt_ten = GT[slot][v]
            pr_ten = PRED[slot][v]
            pr_ten, gt_ten = _align(pr_ten, gt_ten)
            if idx >= 0:
                gt_ten = gt_ten[:, 1:, ...]
                pr_ten = pr_ten[:, 1:, ...]

            # ----------- scalar error per file (for scorecard) -------------
            err_pair[v].append(_file_error(pr_ten, gt_ten))

            # ----------- time‑series error (for hov & future) --------------
            per_timestep = _spatial_mean_error(pr_ten, gt_ten).cpu().numpy()
            series_err[v].append(per_timestep)

            # Hovmöller (lat‑mean)
            hov_lat = torch.abs(pr_ten - gt_ten) if metric_choice == "MAE" else (pr_ten - gt_ten) ** 2
            if metric_choice == "RMSE":
                hov_lat = torch.sqrt(hov_lat)
            hov_lat = hov_lat.mean(dim=-2)
            if hov_lat.ndim == 4:
                hov_lat = hov_lat.mean(dim=(0, 2))
            elif hov_lat.ndim == 3:
                hov_lat = hov_lat.mean(dim=0)
            hov_err[v].append(hov_lat)

    if "species_variables" in GT:
        for sp, gt_ten in GT["species_variables"].items():

            pr_ten = PRED["species_variables"][sp]
            pr_ten, gt_ten = _align(pr_ten, gt_ten)

            if idx >= 0:
                gt_ten, pr_ten = gt_ten[:, 1:, ...], pr_ten[:, 1:, ...]

            occ_gt = float((gt_ten > 0).float().mean().cpu())
            occ_pr = float((pr_ten > 0).float().mean().cpu())
            occ_pair_gt[sp].append(occ_gt)
            occ_pair_pr[sp].append(occ_pr)
            print(f"GT SHAPE AND PRED SHAPE{gt_ten.shape} | {pr_ten.shape}")
            spec_ts_gt[sp].append(gt_ten.float().mean(dim=(-2, -1)).squeeze(0).cpu().numpy())
            spec_ts_pr[sp].append(pr_ten.float().mean(dim=(-2, -1)).squeeze(0).cpu().numpy())
            print(f"Occurances GT {gt_ten.shape} | Occurances Pred {pr_ten.shape}")
            print(
                f"Occurance value: {gt_ten.float().mean(dim=(-2, -1)).squeeze(0).cpu().numpy()} | Prediction Value: {pr_ten.float().mean(dim=(-2, -1)).squeeze(0).cpu().numpy()}"
            )


num_pairs = len(PAIRS)
lead_times = list(range(1, num_pairs + 1))
err_full = {v: np.concatenate(lst) for v, lst in series_err.items()}
hov_full = {v: torch.cat(lst, dim=0) for v, lst in hov_err.items()}
spec_ts_full_gt = {sp: np.concatenate(lst) for sp, lst in spec_ts_gt.items()}
spec_ts_full_pr = {sp: np.concatenate(lst) for sp, lst in spec_ts_pr.items()}

print(err_full)
print(PAIRS)

# Total number of timesteps across all files (for Hovmöller)
T_total = next(iter(err_full.values())).shape[0] if err_full else 0

err_pair: DefaultDict[str, List[float]] = defaultdict(list)
for idx, (gt_path, _) in enumerate(PAIRS):
    for slot, vars_ in var_groups.items():
        for v in vars_:
            pair_err = float(series_err[v][idx][-1])
            err_pair[v].append(pair_err)

# UI Tabs
# -----------------------------------------------------------------------------
TAB_SCORE, TAB_OCC, TAB_HOV, TAB_SPEC, TAB_FACET, TAB_TRAJ, TAB_COMM = st.tabs(
    [
        "Scorecard",
        "Species Occurrence",
        "Hovmöller Error",
        "Species Timeseries",
        "Species Face Maps",
        "Trajectory Map",
        "Community Metrics",
    ]
)


with TAB_COMM:
    st.markdown("### Assemblage-level accuracy")

    gt_stack, pr_stack = [], []
    for gt_path, pr_path in PAIRS:
        GTb = _crop_batch(torch.load(gt_path, map_location="cpu"))
        PRb = torch.load(pr_path, map_location="cpu")[0]._asdict()
        PRb = _intify_species_keys(test_dataset.scale_batch(PRb, direction="original"))

        for sp in sorted(GTb["species_variables"]):
            gt_stack.append(GTb["species_variables"][sp].squeeze())  # (T,H,W)
            pr_stack.append(PRb["species_variables"][sp].squeeze())
    gt_arr = torch.stack(gt_stack) > 0 # (S,T,H,W)  binary
    pr_arr = torch.stack(pr_stack) > 0

    inc_gt = gt_arr.any(dim=1) # (S,H,W)
    inc_pr = pr_arr.any(dim=1)

    S, H, W = inc_gt.shape
    gt_flat = inc_gt.reshape(S, -1).numpy()
    pr_flat = inc_pr.reshape(S, -1).numpy()

    # --- Community Sørensen per cell ------------------------------------
    a = np.logical_and(gt_flat, pr_flat).sum(0)
    b = np.logical_and(gt_flat, ~pr_flat).sum(0)
    c = np.logical_and(~gt_flat, pr_flat).sum(0)
    sor = 2 * a / (2 * a + b + c + 1e-9)
    mean_sor = sor.mean()

    # --- Species-richness RMSE -----------------------------------------
    rich_gt = gt_flat.sum(0);  rich_pr = pr_flat.sum(0)
    sr_rmse = np.sqrt(((rich_gt - rich_pr) ** 2).mean())

    st.markdown(f"**Mean community Sørensen:** {mean_sor:.3f}")
    st.markdown(f"**Species-richness RMSE:** {sr_rmse:.3f} species")

    # --- Richness scatter ----------------------------------------------
    fig1, ax1 = plt.subplots(figsize=(4,4))
    ax1.hexbin(rich_gt, rich_pr, gridsize=35, cmap="Blues")
    lim = max(rich_gt.max(), rich_pr.max()) + 1
    ax1.plot([0, lim], [0, lim], "r--")
    ax1.set_xlabel("Observed richness"); ax1.set_ylabel("Predicted richness")
    ax1.set_title("Richness agreement")
    ax1.text(0.05, 0.9, f"RMSE = {sr_rmse:.2f}",
             transform=ax1.transAxes, fontsize=8, color="darkred")
    st.pyplot(fig1)

    # --- Sørensen similarity map
    sor_map = sor.reshape(H, W)

    proj = ccrs.PlateCarree()
    fig2, ax2 = plt.subplots(figsize=(8, 3), subplot_kw=dict(projection=proj))

    mesh = ax2.pcolormesh(lons, lats, sor_map,
                        vmin=0, vmax=1,
                        cmap="RdYlBu_r",
                        transform=proj)

    # coastlines & grid ---------------------------------------------------------
    ax2.add_feature(cfeature.COASTLINE, lw=0.4)
    gl = ax2.gridlines(draw_labels=True,
                    linewidth=0.5, color="gray", linestyle="--")
    gl.top_labels = gl.right_labels = False
    ax2.set_xlabel("Longitude (°E)")
    ax2.set_ylabel("Latitude (°N)")

    # colour-bar ---------------------------------------------------------------
    cbar = plt.colorbar(mesh, ax=ax2, orientation="vertical", pad=0.02)
    cbar.set_label("Sørensen similarity")
    ax2.set_title(f"Sørensen map")
    st.pyplot(fig2)

# ==================== SCORECARD ============================================
with TAB_SCORE:
    st.subheader(f"{metric_choice} scorecard - per batch")
    slot_sel = st.radio("Variable group", list(var_groups.keys()), key="score_slot")
    vars_sel = var_groups[slot_sel]

    err_mat = np.vstack([err_pair[v] for v in vars_sel]).astype(np.float32)

    use_log = st.checkbox("Log scale", value=True)

    if use_log:
        vmin = np.nanmin(err_mat[err_mat > 0])
        vmax = np.nanmax(err_mat)
        norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)
    else:
        norm = None

    fig, ax = plt.subplots(figsize=(10, max(3, 0.45 * len(vars_sel))))
    im = ax.imshow(err_mat, aspect="auto", cmap="magma_r", norm=norm)
    ax.set_xticks(range(num_pairs))
    date_labels = [p[0].name.split("_")[1] for p in PAIRS]
    ax.set_xticklabels(date_labels, rotation=45, ha="right")
    ax.set_yticks(range(len(vars_sel)))
    ax.set_yticklabels(vars_sel)
    ax.set_xlabel("Batch start date")
    ax.set_title(f"{metric_choice} - domain mean per batch")
    cax = make_axes_locatable(ax).append_axes("right", size="3%", pad=0.15)
    plt.colorbar(im, cax=cax)
    st.pyplot(fig)

# ==================== SPECIES OCCURRENCE ===================================
with TAB_OCC:
    if not occ_pair_gt:
        st.info("No species variables found.")
    else:
        st.subheader("Occurrence probability - per batch")
        sp_sel = st.multiselect("Species", list(occ_pair_gt.keys()), default=list(occ_pair_gt.keys())[:1])
        for sp in sp_sel:
            fig2 = plt.figure(figsize=(7, 3))
            plt.plot(lead_times, occ_pair_gt[sp], marker="o", label="GT", lw=2)
            plt.plot(lead_times, occ_pair_pr[sp], marker="x", label="Rollout", lw=2)
            plt.title(f"Occurrence - {sp}")
            plt.xlabel("Batch index (months)")
            plt.ylabel("Fraction of grid cells")
            plt.grid(True)
            plt.legend()
            st.pyplot(fig2)

# ==================== HOVMÖLLER ERROR ======================================
with TAB_HOV:
    if not hov_full:
        st.info("No variables available for Hovmöller plot.")
    else:
        hov_var = st.selectbox("Variable", list(hov_full.keys()))
        hov = torch.sqrt(hov_full[hov_var]) if metric_choice == "RMSE" else hov_full[hov_var]
        hov_np = hov.cpu().numpy()  # shape (T_total , W)
        lon_deg = (lons % 360).round(1)  # 0–360°, 0.25° spacing
        step = 20  # every 5degs (= 0.25x20)
        lon_ticks = np.arange(0, len(lon_deg), step)
        lon_labels = lon_deg[lon_ticks]

        fig3, ax3 = plt.subplots(figsize=(10, 4))
        mesh = ax3.pcolormesh(np.arange(hov_np.shape[1]), np.arange(1, T_total + 1), hov_np, cmap="coolwarm", shading="auto")

        ax3.set_xlabel("Longitude (°E)")
        ax3.set_ylabel("Lead-time index (months)")
        ax3.set_title(f"Hovmöller error - {hov_var}")
        ax3.set_xticks(lon_ticks)
        ax3.set_xticklabels(lon_labels)

        cax3 = make_axes_locatable(ax3).append_axes("right", size="3%", pad=0.15)
        plt.colorbar(mesh, cax=cax3, label=metric_choice)
        st.pyplot(fig3)

# ─────────────────────────── SPECIES TIME-SERIES TAB ──────────────────────
with TAB_SPEC:
    if not spec_ts_full_gt:
        st.info("No species time-series data.")
        st.stop()

    sp_sel = st.selectbox("Species ID", list(spec_ts_full_gt.keys()),
                          key="spec_ts_select")

    ts_gt = np.asarray(spec_ts_full_gt[sp_sel], dtype=float)
    ts_pr = np.asarray(spec_ts_full_pr[sp_sel], dtype=float)

    T = min(len(ts_gt), len(ts_pr))
    ts_gt, ts_pr = ts_gt[:T], ts_pr[:T]
    x = np.arange(1, T + 1)

    col1, col2 = st.columns(2)
    use_log  = col1.checkbox("Log scale", value=False)
    use_dual = col2.checkbox("Dual axis", value=False)

    eps = 0.05 * np.nanmin(ts_gt[ts_gt > 0]) if use_log else 0.0

    fig, ax = plt.subplots(figsize=(9, 3))

    if use_log:
        ax.set_yscale("log")
        ax.plot(x, ts_gt + eps, label="GT", lw=2, color="C0")
        ax.plot(x, ts_pr + eps, label="Rollout", lw=2, ls="--", color="C3")
        ax.set_ylabel("Concentration (log)")
        ax.legend()

    elif use_dual:
        ax.plot(x, ts_gt, color="C0", lw=2, label="GT")
        ax.set_ylabel("GT conc.", color="C0")
        ax.tick_params(axis="y", colors="C0")

        ax2 = ax.twinx()
        ax2.plot(x, ts_pr, color="C3", lw=2, ls="--", label="Rollout")
        ax2.set_ylabel("Rollout conc.", color="C3")
        ax2.tick_params(axis="y", colors="C3")

        lines = ax.get_lines() + ax2.get_lines()
        labels = [ln.get_label() for ln in lines]
        ax.legend(lines, labels, loc="upper right")

    else:
        ax.plot(x, ts_gt, lw=2, color="C0", label="GT")
        ax.plot(x, ts_pr, lw=2, ls="--", color="C3", label="Rollout")
        ax.set_ylabel("Concentration")
        ax.legend()

    ax.set_xlabel("Timestep")
    ax.set_title(f"Domain-mean distribution — species {sp_sel}")
    ax.grid(True, which="both", ls=":")
    st.pyplot(fig)


with TAB_FACET:
    if "species_variables" not in GT or "species_variables" not in PRED:
        st.info("No species_variables in these batches.")
        st.stop()

    # ── UI controls ────────────────────────────────────────────────────────
    common_sp = sorted(set(GT["species_variables"]).intersection(PRED["species_variables"]))
    if not common_sp:
        st.info("Species list differs between GT and rollout.")
        st.stop()

    sp_id = st.selectbox("Species ID", common_sp)
    source = st.radio("Data source", ["Ground Truth", "Rollout"], horizontal=True)
    win = st.slider("Batch index", 0, len(PAIRS) - 1, 0)

    G_win = _crop_batch(torch.load(PAIRS[win][0], map_location="cpu"))
    P_win = _intify_species_keys(
        test_dataset.scale_batch(torch.load(PAIRS[win][1], map_location="cpu")[0]._asdict(), direction="original")
    )

    tensor = (G_win if source == "Ground Truth" else P_win)["species_variables"][sp_id]
    tensor = tensor.squeeze()  # (T, H, W)
    Tm = tensor.shape[0]

    ncols = min(Tm, 4)
    nrows = int(np.ceil(Tm / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 2.8 * nrows), subplot_kw=dict(projection=ccrs.PlateCarree()))
    axes = np.atleast_2d(axes)

    vmin, vmax = tensor.min(), tensor.max()

    for t in range(Tm):
        r, c = divmod(t, ncols)
        ax = axes[r, c]
        cyc, lon_c = add_cyclic_point(tensor[t], coord=lons)
        mesh = ax.pcolormesh(lon_c, lats, cyc, cmap="viridis", vmin=vmin, vmax=vmax, transform=ccrs.PlateCarree())
        ax.add_feature(cfeature.COASTLINE, lw=0.4)
        ax.set_title(f"t{t}", fontsize=8)
        gl = ax.gridlines(draw_labels=False, linewidth=0.3, color="gray", linestyle="--")
    for t in range(Tm, nrows * ncols):
        axes.ravel()[t].axis("off")

    cax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(mesh, cax=cax, label="Distribution density")
    fig.suptitle(f"{source}: species {sp_id} — batch {win}", y=0.97)
    st.pyplot(fig)

    overlay_mode = st.checkbox("Overlay all timesteps on one map", value=True)


with TAB_TRAJ:
    if "species_variables" not in GT or "species_variables" not in PRED:
        st.info("No species variables in these batches.")
        st.stop()

    common_sp = sorted(set(GT["species_variables"]).intersection(PRED["species_variables"]))
    sp_id = st.selectbox("Species ID", common_sp, key="traj_species")
    source = st.radio("Data source", ["Ground Truth", "Rollout"], horizontal=True, key="traj_src")

    weight_by_abund = st.sidebar.checkbox("Weight centroid by abundance", value=False)
    use_landmask = st.sidebar.checkbox("Apply land mask", value=True)
    show_mask = st.sidebar.checkbox("Show land mask overlay", value=False)
    thresh = st.sidebar.slider("Presence threshold (quantile)", 0.0, 1.0, 0.7,
                            help="Keep only pixels above this abundance quantile")
    clean  = st.sidebar.checkbox("Morphological clean (3x3)", value=True,
                                help="Remove isolated pixels")
    full_ts = []
    for gt_p, pr_p in PAIRS:
        batch = torch.load(gt_p if source == "Ground Truth" else pr_p, map_location="cpu")
        if source == "Rollout":
            batch = _intify_species_keys(test_dataset.scale_batch(batch[0]._asdict(), direction="original"))

        batch = _crop_batch(batch)
        full_ts.append(batch["species_variables"][sp_id].squeeze())  # (T,H,W)
    full_ts = torch.cat(full_ts, dim=0)# (T_total,H,W)

    proj = ccrs.PlateCarree()

    plt.rcParams.update({"axes.titlesize": 10, "axes.labelsize": 8, "xtick.labelsize": 6, "ytick.labelsize": 6})

    fig, ax = plt.subplots(figsize=(7, 8), subplot_kw=dict(projection=proj))

    cmap = plt.cm.get_cmap("tab20", full_ts.shape[0])
    norm = plt.Normalize(0, full_ts.shape[0] - 1)


    slice_t = full_ts[t].cpu().numpy()

    if source == "Rollout":
        q_val = np.quantile(slice_t, thresh)
        mask  = slice_t > q_val
        if clean:
            mask = ndi.binary_opening(mask, structure=np.ones((3, 3)))
    else:
        mask = (full_ts > 0).float()    

    for t in range(full_ts.shape[0]):
        cyc, lon_c = add_cyclic_point((full_ts[t] > 0).float(), coord=lons)
        ax.contourf(lon_c, lats, cyc, levels=[0.5, 1], colors=[cmap(norm(t))], alpha=0.4, transform=proj)

    lon_grid, lat_grid = np.meshgrid(lons, lats)
    base = full_ts.float() if weight_by_abund else (full_ts > 0).float()

    if use_landmask:
        land_mask = torch.tensor(build_land_mask(lon_grid, lat_grid), dtype=base.dtype)
        pres = base * land_mask  # broadcast over time
    else:
        pres = base  # no masking

    w = pres.sum(dim=(-2, -1))  # (T,)
    lon_cent = torch.where(w > 0, (pres * torch.tensor(lon_grid)).sum((-2, -1)) / w, torch.nan)
    lat_cent = torch.where(w > 0, (pres * torch.tensor(lat_grid)).sum((-2, -1)) / w, torch.nan)

    valid = ~torch.isnan(lon_cent)
    lon_cent = lon_cent[valid].cpu().numpy()
    lat_cent = lat_cent[valid].cpu().numpy()

    if show_mask:
        mask_alpha = 0.25
        ax.contourf(
            lon_grid,
            lat_grid,
            build_land_mask(lon_grid, lat_grid),
            levels=[0.5, 1],
            colors=["lightgrey"],
            alpha=mask_alpha,
            transform=proj,
        )

    ax.plot(lon_cent, lat_cent, "-k", lw=1, transform=proj)
    # ax.scatter(lon_cent, lat_cent, s=30, alpha=.8)
    for t in range(len(lon_cent) - 1):
        ax.annotate(
            "",
            xy=(lon_cent[t + 1], lat_cent[t + 1]),
            xytext=(lon_cent[t], lat_cent[t]),
            textcoords=proj,
            arrowprops=dict(arrowstyle="->", color="k", shrinkA=0.3, shrinkB=0.3, lw=0.6),
            annotation_clip=False,
        )
        ax.plot(lon_cent[t], lat_cent[t], "o", color=cmap(norm(t)), transform=proj)

    def _haversine(lon1, lat1, lon2, lat2):
        R = 6371.0  # km
        dlon = radians(lon2 - lon1)
        dlat = radians(lat2 - lat1)
        a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
        return 2 * R * atan2(sqrt(a), sqrt(1 - a))

    dists = [_haversine(lon_cent[i], lat_cent[i], lon_cent[i + 1], lat_cent[i + 1]) for i in range(len(lon_cent) - 1)]
    mean_speed = np.mean(dists)  # km per month
    st.markdown(f"**Mean centroid drift:** {mean_speed:.1f} km mo⁻¹")

    # map cosmetics
    ax.add_feature(cfeature.COASTLINE, lw=0.4)
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color="gray", linestyle="--")
    gl.top_labels = gl.right_labels = False
    ax.set_title(f"{source}: species ID {sp_id} — centroid species trajectory")
    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")

    handles = [
        plt.Line2D([0], [0], ms=5, marker="s", ls="", mfc=cmap(norm(i)), mec=cmap(norm(i))) for i in range(full_ts.shape[0])
    ]

    ax.legend(
        handles,
        [f"t{i}" for i in range(full_ts.shape[0])],
        ncol=1,
        fontsize=6,
        loc="center left",
        bbox_to_anchor=(1.00, 0.5),
        frameon=False,
    )
    st.pyplot(fig)

    dists = np.array([_haversine(lon_cent[i], lat_cent[i], lon_cent[i + 1], lat_cent[i + 1]) for i in range(len(lon_cent) - 1)])

    lon_grid, lat_grid = np.meshgrid(lons, lats)
    areas = []
    for t in range(full_ts.shape[0]):
        yy, xx = np.where(full_ts[t] > 0)
        if len(xx) > 2:
            pts = np.column_stack((lon_grid[yy, xx], lat_grid[yy, xx]))
            hull = ConvexHull(pts).volume  # deg^2
            lat_bar = np.deg2rad(pts[:, 1].mean())
            areas.append(hull * (111.32 * cos(lat_bar)) * 110.57)  # km^2
        else:
            areas.append(0.0)
    areas = np.asarray(areas)
    cumdist = np.concatenate(([0], np.cumsum(dists)))  # km

    fig = plt.figure(figsize=(6, 8))
    gs = fig.add_gridspec(4, 1, height_ratios=[3, 1, 1, 1], hspace=0.35)

    ax_dist = fig.add_subplot(gs[1, 0])
    ax_area = fig.add_subplot(gs[2, 0])
    ax_cum = fig.add_subplot(gs[3, 0])

    ax_dist.step(np.arange(1, len(dists) + 1), dists, where="mid", color="k")
    ax_dist.axhline(dists.mean(), ls="--", lw=0.8, color="grey")
    ax_dist.set_ylabel("km month$^{-1}$")
    ax_dist.set_title("Centroid displacement")

    ax_area.fill_between(np.arange(1, len(areas) + 1), areas, step="mid", alpha=0.4, color="C2")
    ax_area.axhline(areas.mean(), ls="--", lw=0.8, color="grey")
    ax_area.set_ylabel("Hull km$^{2}$")
    ax_area.set_title("Occupied-area timeline")

    ax_cum.plot(np.arange(len(cumdist)), cumdist, color="purple")
    ax_cum.set_xlabel("Timestep")
    ax_cum.set_ylabel("km")
    ax_cum.set_title("Cumulative distance")

    st.pyplot(fig)

TAB_ACC = st.tabs(["Monthly ACC"])[0]

def _ensure_time_axis(t: torch.Tensor) -> torch.Tensor:
    """
    Ensure tensor has shape (T, H, W).  If shape is (H, W) add T=1 dim.
    If shape is (L, H, W) treat L as time.  If shape is (T, L, H, W) leave unchanged.
    """
    if t.ndim == 2:
        return t.unsqueeze(0)
    if t.ndim == 3:
        return t.unsqueeze(0) if t.shape[0] <= 50 else t
    return t

with TAB_ACC:
    phys_vars = []
    for slot, vlist in var_groups.items():
        if not slot.startswith("species"):
            phys_vars.extend([(slot, v) for v in vlist])

    if not phys_vars:
        st.info("No continuous variables available for ACC."); st.stop()

    sel = st.selectbox("Variable", [f"{s} / {v}" for s, v in phys_vars])
    slot, v_name = sel.split(" / ")

    try:
        gt_seq, pr_seq, month_idx = [], [], []
        pressure_levels = None # will fill if 4-D field

        for gt_path, pr_path in PAIRS:
            GTb = _crop_batch(torch.load(gt_path, map_location="cpu"))
            PRb = _crop_batch(test_dataset.scale_batch(
                              torch.load(pr_path, map_location="cpu")[0]._asdict(),
                              direction="original"))

            if pressure_levels is None and "pressure_levels" in GTb["batch_metadata"]:
                pressure_levels = GTb["batch_metadata"]["pressure_levels"]

            ten_gt = _ensure_time_axis(GTb[slot][v_name])# shapes: (T,H,W) or (T,L,H,W)
            ten_pr = _ensure_time_axis(PRb[slot][v_name])

            months = [pd.Timestamp(ts).month - 1
                      for ts in GTb["batch_metadata"]["timestamp"]]

            if ten_gt.ndim == 4:
                levs = pressure_levels
                lev = st.selectbox("Pressure level (hPa)", levs, key=f"pl_{v_name}")
                ci  = levs.index(lev)
                ten_gt = ten_gt[:, ci]
                ten_pr = ten_pr[:, ci]

            gt_seq.append(ten_gt);  pr_seq.append(ten_pr)
            month_idx.extend(months)

        obs_full  = torch.cat(gt_seq,  dim=0)
        pred_full = torch.cat(pr_seq,  dim=0)
        months_arr = np.asarray(month_idx, dtype=int)

        clim = torch.zeros(12, *obs_full.shape[-2:])
        for m in range(12):
            clim[m] = obs_full[months_arr == m].mean(dim=0)

        acc = monthly_acc(pred_full, obs_full, clim, months_arr)
        mean_acc = np.nanmean(acc)

        st.markdown(f"**Mean ACC:** {mean_acc:.2f}")

        fig, ax = plt.subplots(figsize=(8, 3))
        ax.bar(np.arange(1, len(acc)+1), acc, color="C0")
        ax.axhline(0.60, ls="--", color="red", lw=1,
                   label="Useful forecast (0.6)")
        ax.set_ylim(-1, 1); ax.set_xlabel("Frame index"); ax.set_ylabel("ACC")
        ax.legend(); ax.grid(ls=":")
        st.pyplot(fig)

    except Exception as e:
        st.error(f"ACC computation failed: {e}")