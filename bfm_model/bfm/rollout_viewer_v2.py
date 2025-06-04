"""
Batch-comparison viewer for spatiotemporal rollouts

** UNDER DEVELOPMENT
"""
from __future__ import annotations

import argparse
import io
import re
from pathlib import Path
from typing import Dict, List, Tuple, DefaultDict
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import streamlit as st
from sklearn.metrics import mean_absolute_error, mean_squared_error
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


def _spatial_mean_error(pred, obs):
    obs = obs.unsqueeze(0)
    print(f"pred shape {pred.shape} | obs shape: {obs.shape}")
    if pred.ndim == 4:
        pred, obs = pred[:, 0], obs[:, 0]
    e = torch.abs(pred - obs) if metric_choice == "MAE" else (pred - obs) ** 2
    if metric_choice == "RMSE":
        e = torch.sqrt(e)
    return e.mean(dim=(-2, -1))  # (T,)

# ----------------------------------------------------------------------------
# Load and concatenate all pairs
# ----------------------------------------------------------------------------
var_groups: Dict[str, List[str]] = {}
series_err: DefaultDict[str, List[np.ndarray]] = defaultdict(list)
series_occ_gt: DefaultDict[str, List[np.ndarray]] = defaultdict(list)
series_occ_pr: DefaultDict[str, List[np.ndarray]] = defaultdict(list)
hov_err: DefaultDict[str, List[torch.Tensor]] = defaultdict(list)

for idx, (gt_path, ro_path) in enumerate(PAIRS):
    GT = torch.load(gt_path, map_location="cpu")
    RO = torch.load(ro_path, map_location="cpu")
    PRED = RO["predictions"]

    GT = _crop_batch(GT)

    # variable groups (once)
    if not var_groups:
        var_groups = {slot: sorted(vdict.keys()) for slot, vdict in GT.items() if slot.endswith("_variables")}

    for slot, vars_ in var_groups.items():
        for v in vars_:
            gt_ten = GT[slot][v]
            pr_ten = PRED[slot][v]
            # skip first timestep for all but first file to avoid overlap
            if idx > 0:
                gt_ten = gt_ten[1:]
                pr_ten = pr_ten[1:]
            # scorecard series
            series_err[v].append(_spatial_mean_error(pr_ten, gt_ten).cpu().numpy())
            # hovmöller (store tensor)
            hov_e = torch.abs(pr_ten - gt_ten) if metric_choice == "MAE" else (pr_ten - gt_ten) ** 2
            hov_err[v].append(hov_e.mean(dim=-2))  # (T, lon)

    # species occurrence
    if "species_variables" in GT:
        for sp, gt_ten in GT["species_variables"].items():
            pr_ten = PRED["species_variables"][sp]
            if idx > 0:
                gt_ten = gt_ten[1:]
                pr_ten = pr_ten[1:]
            series_occ_gt[sp].append((gt_ten > 0).float().mean(dim=(-2, -1)).cpu().numpy())
            series_occ_pr[sp].append((pr_ten > 0).float().mean(dim=(-2, -1)).cpu().numpy())

# Concatenate per‑timestep errors across files (already done above)
err_full = {v: np.concatenate(lst) for v, lst in series_err.items()}
occ_gt_full = {sp: np.concatenate(lst) for sp, lst in series_occ_gt.items()}
occ_pr_full = {sp: np.concatenate(lst) for sp, lst in series_occ_pr.items()}
hov_full = {v: torch.cat(lst, dim=0) for v, lst in hov_err.items()}

T_total = next(iter(err_full.values())).shape[0] if err_full else 0


# ---------------- File‑wise aggregate (one value per pair) -------------------
err_pair: DefaultDict[str, List[float]] = defaultdict(list)
for idx, (gt_path, _) in enumerate(PAIRS):
    for slot, vars_ in var_groups.items():
        for v in vars_:
            # take mean of spatial‑mean error sequence for this file
            pair_err = float(series_err[v][idx][-1])  # last timestep error  # scalar per file
            err_pair[v].append(pair_err)

num_pairs = len(PAIRS)
lead_times = list(range(1, num_pairs + 1))  # x‑axis for scorecard by month


# ----------------------------------------------------------------------------
# Build UI tabs
# ----------------------------------------------------------------------------
TAB_SCORE, TAB_OCC, TAB_HOV = st.tabs(["Scorecard", "Species Occurrence", "Hovmöller Error"])

with TAB_SCORE:
    st.subheader(f"{metric_choice} scorecard - per batch/month")
    slot_sel = st.radio("Variable group", list(var_groups.keys()), key="score_slot")
    vars_sel = var_groups[slot_sel]

    # Matrix: rows=variables, cols=file‑pairs
    err_mat_pair = np.vstack([err_pair[v] for v in vars_sel]).astype(np.float32)

    fig, ax = plt.subplots(figsize=(10, max(3, 0.45 * len(vars_sel))))
    im = ax.imshow(err_mat_pair, aspect="auto", cmap="magma_r")

    ax.set_xticks(range(num_pairs))
    date_labels = [p[0].name.split("_")[1] for p in PAIRS]
    ax.set_xticklabels(date_labels, rotation=45, ha="right")

    ax.set_yticks(range(len(vars_sel)))
    ax.set_yticklabels(vars_sel)
    ax.set_xlabel("Batch start date (≈ month)")
    ax.set_title(f"{metric_choice} - domain mean per batch")

    div = make_axes_locatable(ax)
    cax = div.append_axes("right", size="3%", pad=0.15)
    plt.colorbar(im, cax=cax)
    st.pyplot(fig)

with TAB_OCC:
    if not occ_gt_full:
        st.info("No species variables found.")
    else:
        st.subheader("Occurrence probability (all batches)")
        sp_sel = st.multiselect("Species", list(occ_gt_full.keys()), default=list(occ_gt_full.keys())[:1])
        for sp in sp_sel:
            fig2 = plt.figure(figsize=(7, 3))
            plt.plot(lead_times, occ_gt_full[sp], label="GT", lw=2)
            plt.plot(lead_times, occ_pr_full[sp], label="Rollout", lw=2)
            plt.title(f"Occurrence – {sp}")
            plt.xlabel("Lead-time index (months)"); plt.ylabel("Fraction of grid cells")
            plt.grid(True); plt.legend(); st.pyplot(fig2)

with TAB_HOV:
    hov_var = st.selectbox("Variable", list(hov_full.keys()))
    hov = hov_full[hov_var]
    if metric_choice == "RMSE":
        hov = torch.sqrt(hov)
    hov_np = hov.cpu().numpy()

    fig3, ax3 = plt.subplots(figsize=(10, 4))
    mesh = ax3.pcolormesh(np.arange(hov_np.shape[1]), np.arange(1, T_total + 1), hov_np,
                          cmap="coolwarm", shading="auto")
    ax3.set_xlabel("Longitude index"); ax3.set_ylabel("Lead-time index (months)")
    ax3.set_title(f"Hovmöller error - {hov_var}")
    div = make_axes_locatable(ax3); cax3 = div.append_axes("right", size="3%", pad=0.15)
    plt.colorbar(mesh, cax=cax3, label=metric_choice)
    st.pyplot(fig3)
