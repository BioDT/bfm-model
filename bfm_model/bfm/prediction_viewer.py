"""
Copyright (C) 2025 TNO, The Netherlands. All rights reserved.

Visualise prediction vs ground-truth batches.

run:
    streamlit run prediction_viewer.py -- --data_dir ./predictions
"""
from __future__ import annotations
import argparse
from pathlib import Path
from typing import Dict

import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs, cartopy.feature as cfeature
from cartopy.util import add_cyclic_point
import pandas as pd

LAT_START, LAT_END = 32.0, 72.0
LON_START, LON_END = -25.0, 45.0
GRID_LAT = np.round(np.arange(LAT_START, LAT_END + 1e-6, 0.25), 3)
GRID_LON = np.round(np.arange(LON_START, LON_END + 1e-6, 0.25), 3)

def _get_dir() -> Path:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--data_dir", default="pre_train_exports", type=Path)
    ns, _ = p.parse_known_args()
    return ns.data_dir.resolve()

DATA_DIR = _get_dir()

def _plot_maps(arr_pred, arr_gt, lats, lons, title):
    arr_pred = np.asarray(arr_pred).squeeze()
    arr_gt = np.asarray(arr_gt).squeeze()
    proj = ccrs.PlateCarree()
    fig, ax = plt.subplots(1, 2, figsize=(10, 4), subplot_kw=dict(projection=proj))
    for a, dat, lbl in zip(ax, (arr_pred, arr_gt), ("Prediction", "Ground Truth")):
        cyc, lon_cyc = add_cyclic_point(dat, coord=lons)
        mesh = a.pcolormesh(lon_cyc, lats, cyc, cmap="viridis", transform=proj)
        a.add_feature(cfeature.COASTLINE, lw=.4)
        a.set_title(f"{title}\n{lbl}")
    fig.colorbar(mesh, ax=ax.ravel().tolist(), shrink=.75)
    st.pyplot(fig)

st.sidebar.title("Prediction vs Ground Truth viewer")
files = sorted(DATA_DIR.glob("window_*.pt"))
if not files:
    st.sidebar.error("No window_*.pt files found")
    st.stop()

file_sel = st.sidebar.selectbox("Window file", [f.name for f in files])
window = torch.load(DATA_DIR / file_sel, map_location="cpu", weights_only=False)
pred, gt = window["pred"], window["gt"]


meta = gt["batch_metadata"]
lats = np.array(meta["latitudes"])[0]
lons = np.array(meta["longitudes"])[0]
timestamp = meta["timestamp"][0]# first slice only
# print(lons, lats)
slot_list = [k for k in pred.keys() if k.endswith("_variables")]
slot = st.sidebar.selectbox("Variable slot", slot_list)

var_list = list(pred[slot].keys())
var_sel  = st.sidebar.multiselect("Variable(s)", var_list, default=[var_list[0]])
if not var_sel: st.info("Pick variable"); st.stop()

# pressure-level selector
pl_sel = None
sample_tensor = next(iter(pred[slot].values()))
if sample_tensor.ndim == 4:   # (1,C,H,W)
    pl_vals = meta["pressure_levels"]
    pl_sel  = st.sidebar.multiselect("Pressure hPa", pl_vals, default=[pl_vals[0]])
    if not pl_sel: st.stop()

st.header(f"{file_sel} — {timestamp}")
for v in var_sel:
    ten_pred = pred[slot][v] # (1,H,W) or (1,C,H,W)
    ten_gt = gt[slot][v][0] # take first time slice

    if ten_pred.ndim == 3:
        _plot_maps(ten_pred[0], ten_gt[0], lats, lons, v)
    else:
        for pl in pl_sel:
            cidx = meta["pressure_levels"].index(pl)
            pred_map = ten_pred[0, cidx]  # (H,W)
            gt_map = ten_gt[0, cidx] # first time slice  (H,W)
            _plot_maps(pred_map, gt_map, lats, lons, f"{v}  {pl} hPa")

st.subheader("Species variable - cumulative mean (moving average)")
spec_slot = "species_variables"
if spec_slot not in pred:
    st.info("No species_variables slot present."); st.stop()

all_species: set[str] = set()
for f in files:
    w = torch.load(f, map_location="cpu")
    all_species.update(w["gt"][spec_slot].keys())

species_id = st.sidebar.selectbox(
    "Species ID for time-series", sorted(all_species))

# species_list = ["1340361", "1340503", "1536449", "1898286", "1920506", "2430567",
#                        "2431885", "2433433", "2434779", "2435240", "2435261", "2437394",
#                        "2441454", "2473958", "2491534", "2891770", "3034825", "4408498",
#                         "5218786", "5219073", "5219173", "5219219", "5844449", "8002952",
#                         "8077224", "8894817", "8909809", "9809229"]

def _scalar_ts(ts) -> pd.Timestamp:
    """Return scalar pd.Timestamp from str / numpy / Timestamp / Index."""
    if isinstance(ts, (list, tuple, np.ndarray)):
        ts = ts[0]
    if isinstance(ts, pd.DatetimeIndex):
        ts = ts[0]
    return pd.Timestamp(ts)

records = []
for f in files:
    w  = torch.load(f, map_location="cpu")
    ts = _scalar_ts(w["gt"]["batch_metadata"]["timestamp"][0])

    # some windows may lack the chosen species -> skip
    if species_id not in w["gt"][spec_slot]:
        continue

    mean_gt= w["gt"  ][spec_slot][species_id][0].float().mean().item()
    mean_pred= w["pred"][spec_slot][species_id][0].float().mean().item()
    records.append({"timestamp": ts, "GT": mean_gt, "PRED": mean_pred})

if not records:
    st.warning(f"No data for species {species_id} in these windows."); st.stop()

df = (pd.DataFrame(records)
        .sort_values("timestamp")
        .set_index("timestamp")
        .astype(float))

df["GT_ma"] = df["GT"].expanding().mean()
df["PRED_ma"] = df["PRED"].expanding().mean()

fig2 = plt.figure(figsize=(8,3))
plt.plot(df.index, df["GT_ma"], label="GT", lw=2)
plt.plot(df.index, df["PRED_ma"], label="PRED", lw=2)
plt.title(f"Cumulative mean — species {species_id}")
plt.legend(); plt.grid(True); plt.ylabel("Distribution mean")
st.pyplot(fig2)