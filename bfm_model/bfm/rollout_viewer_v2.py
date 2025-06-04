"""
Batch-comparison viewer for spatiotemporal rollouts

** UNDER DEVELOPMENT
"""
import argparse
import re
from pathlib import Path
from typing import Dict, List, DefaultDict
from collections import defaultdict

import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import streamlit as st
from bfm_model.bfm.dataloader_monthly import LargeClimateDataset
from omegaconf import OmegaConf
import hydra
from hydra.core.global_hydra import GlobalHydra
GlobalHydra.instance().clear()
from types import SimpleNamespace

hydra.initialize(config_path="", version_base=None)
cfg = hydra.compose(config_name="configs/train_config.yaml")
# print(OmegaConf.to_yaml(cfg))

cfg = hydra.compose(config_name="configs/train_config.yaml", overrides=["+data.scaling.enabled=True"])


scaling = SimpleNamespace(
    enabled=True,
    stats_path= "/Users/azd/Desktop/git_projects/bfm/bfm-model/batch_statistics/monthly_batches_stats_splitted_channels.json",
    mode= "normalize",)
print(scaling.stats_path)
data_dir = "/Users/azd/Desktop/git_projects/bfm/bfm-model/data_monthly"

def crop_variables(variables, new_H, new_W, **kwargs):  # type: ignore
    """Fallback: crop tensors or dicts to (new_H,new_W)."""
    if isinstance(variables, dict):
        return {k: v[..., :new_H, :new_W] for k, v in variables.items()}
    return variables[..., :new_H, :new_W]

test_dataset = LargeClimateDataset(
        data_dir=data_dir, scaling_settings=scaling)

def _parse_args():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--gt_dir", type=Path, default="/Users/azd/Desktop/git_projects/bfm/bfm-model/data_monthly")
    p.add_argument("--rollout_dir", type=Path, default="/Users/azd/Desktop/git_projects/bfm/bfm-model/rollout_batches")
    p.add_argument("--crop", nargs=2, type=int, default=[160, 280], metavar=("H", "W"))
    return p.parse_known_args()[0]

ARGS = _parse_args()
GT_DIR, RO_DIR = ARGS.gt_dir.resolve(), ARGS.rollout_dir.resolve()
NEW_H, NEW_W = ARGS.crop

# -----------------------------------------------------------------------------
# File matching
# -----------------------------------------------------------------------------
DATE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})")

def _date(fn: str) -> str | None:
    m = DATE_RE.search(fn)
    return m.group(1) if m else None

def _pair_files(gt_dir: Path, ro_dir: Path):
    gt_files = { _date(f.name): f for f in gt_dir.glob("batch_*.pt") }
    ro_files = { _date(f.name): f for f in ro_dir.glob("prediction_*.pt") }
    return sorted([(gt_files[d], ro_files[d]) for d in gt_files if d in ro_files], key=lambda p: p[0].name)

PAIRS = _pair_files(GT_DIR, RO_DIR)
if not PAIRS:
    st.error("No matching files."); st.stop()

# ----------------------------------------------------------------------------
# Sidebar global settings
# ----------------------------------------------------------------------------
st.sidebar.title("Batch comparison - timeline mode")
metric_choice = st.sidebar.radio("Error metric", ["RMSE", "MAE"], index=0)

# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

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


def _spatial_mean_error(pred: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
    """Return a 1-D `(T,)` tensor: error averaged over batch + channel + space.

    """
    pred, obs = _align(pred, obs)
    err = torch.abs(pred - obs) if metric_choice == "MAE" else (pred - obs) ** 2
    if metric_choice == "RMSE":
        err = torch.sqrt(err)
    # spatial mean first
    err = err.mean(dim=(-2, -1))
    if err.ndim == 3:
        err = err.mean(dim=(0, 2))
    elif err.ndim == 2:
        err = err.mean(dim=0)
    return err


def _file_error(pred: torch.Tensor, obs: torch.Tensor) -> float:
    """Spatial+temporal mean error for an entire batch window."""
    err_ts = _spatial_mean_error(pred, obs)
    return float(err_ts.mean().cpu())  # (T,)

# ----------------------------------------------------------------------------
# Load and process all matched file pairs
# ----------------------------------------------------------------------------
var_groups: Dict[str, List[str]] = {}
err_pair: DefaultDict[str, List[float]] = defaultdict(list)
series_err: DefaultDict[str, List[np.ndarray]] = defaultdict(list)
occ_pair_gt: DefaultDict[str, List[float]] = defaultdict(list)
occ_pair_pr: DefaultDict[str, List[float]] = defaultdict(list)
hov_err: DefaultDict[str, List[torch.Tensor]] = defaultdict(list)

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
    PRED = RO[1]._asdict()
    print(type(RO), len(PRED))

    GT = _crop_batch(GT)

    if not var_groups:
        var_groups = {slot: sorted(vdict.keys()) for slot, vdict in GT.items() if slot.endswith("surface_variables")} # TODO Change this to view the groups
    test_var = list(var_groups[next(iter(var_groups))])[0]
    # print(f"Will be using only the ", test_var)
    for slot, vars_ in var_groups.items():
        for v in vars_:
            gt_ten = GT[slot][v]
            pr_ten = PRED[slot][v]
            gt_ten, pr_ten = _align(pr_ten, gt_ten)
            if idx > 0:
                gt_ten = gt_ten[:,1:,...]
                pr_ten = pr_ten[:,1:,...]

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
                hov_lat = hov_lat.mean(dim=(0,2))
            elif hov_lat.ndim == 3:
                hov_lat = hov_lat.mean(dim=0)
            hov_err[v].append(hov_lat)

    # -------- species occurrence --------
    if "species_variables" in GT:
        for sp, gt_ten in GT["species_variables"].items():
            pr_ten = PRED["species_variables"][sp]
            if idx > 0:
                gt_ten, pr_ten = gt_ten[1:], pr_ten[1:]
            # occ_gt = float((gt_ten > 0).float().mean().cpu())
            # occ_pr = float((pr_ten > 0).float().mean().cpu())
            occ_gt = float((gt_ten > 0).float().mean().cpu())
            occ_pr = float((pr_ten > 0).float().mean().cpu())
            occ_pair_gt[sp].append(occ_gt)
            occ_pair_pr[sp].append(occ_pr)


# print(series_err)
num_pairs = len(PAIRS)
lead_times = list(range(1, num_pairs + 1))
err_full = {v: np.concatenate(lst) for v, lst in series_err.items()}
hov_full = {v: torch.cat(lst, dim=0) for v, lst in hov_err.items()}

print(err_full)
print(PAIRS)

# Total number of timesteps across all files (for Hovmöller)
T_total = next(iter(err_full.values())).shape[0] if err_full else 0

# ---------------- File‑wise aggregate (one value per pair) -------------------
err_pair: DefaultDict[str, List[float]] = defaultdict(list)
for idx, (gt_path, _) in enumerate(PAIRS):
    for slot, vars_ in var_groups.items():
        for v in vars_:
            pair_err = float(series_err[v][idx][-1])
            err_pair[v].append(pair_err)

# -----------------------------------------------------------------------------
# UI Tabs
# -----------------------------------------------------------------------------
TAB_SCORE, TAB_OCC, TAB_HOV = st.tabs(["Scorecard", "Species Occurrence", "Hovmöller Error"])

# ==================== SCORECARD ============================================
with TAB_SCORE:
    st.subheader(f"{metric_choice} scorecard - per batch")
    slot_sel = st.radio("Variable group", list(var_groups.keys()), key="score_slot")
    vars_sel = var_groups[slot_sel]

    err_mat = np.vstack([err_pair[v] for v in vars_sel]).astype(np.float32)

    fig, ax = plt.subplots(figsize=(10, max(3, 0.45 * len(vars_sel))))
    im = ax.imshow(err_mat, aspect="auto", cmap="magma_r")
    ax.set_xticks(range(num_pairs))
    date_labels = [p[0].name.split("_")[1] for p in PAIRS]
    ax.set_xticklabels(date_labels, rotation=45, ha="right")
    ax.set_yticks(range(len(vars_sel))); ax.set_yticklabels(vars_sel)
    ax.set_xlabel("Batch start date (≈ month)")
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
            plt.xlabel("Batch index (months)"); plt.ylabel("Fraction of grid cells")
            plt.grid(True); plt.legend(); st.pyplot(fig2)

# ==================== HOVMÖLLER ERROR ======================================
with TAB_HOV:
    if not hov_full:
        st.info("No variables available for Hovmöller plot.")
    else:
        hov_var = st.selectbox("Variable", list(hov_full.keys()))
        hov = torch.sqrt(hov_full[hov_var]) if metric_choice == "RMSE" else hov_full[hov_var]
        hov_np = hov.cpu().numpy()
        fig3, ax3 = plt.subplots(figsize=(10, 4))
        mesh = ax3.pcolormesh(np.arange(hov_np.shape[1]), np.arange(1, T_total + 1), hov_np,
                              cmap="coolwarm", shading="auto")
        ax3.set_xlabel("Longitude index"); ax3.set_ylabel("Lead-time index (months)")
        ax3.set_title(f"Hovmöller error - {hov_var}")
        cax3 = make_axes_locatable(ax3).append_axes("right", size="3%", pad=0.15)
        plt.colorbar(mesh, cax=cax3, label=metric_choice)
        st.pyplot(fig3)

