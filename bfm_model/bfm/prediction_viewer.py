"""
Copyright 2025 (C) TNO. Licensed under the MIT license.

Visualise prediction vs ground-truth batches.

run:
    streamlit run prediction_viewer.py -- --data_dir ./predictions
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
import streamlit as st
import torch
from cartopy.util import add_cyclic_point
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from sklearn.metrics import mean_absolute_error, mean_squared_error

try:
    from easy_mpl import taylor_plot as em_taylor_plot

    TAYLOR_BACKEND = "easy_mpl"
except ImportError:  # fallback to legacy skillmetrics
    try:
        import skillmetrics as sm

        TAYLOR_BACKEND = "skillmetrics"
    except ImportError:
        TAYLOR_BACKEND = None

LAT_START, LAT_END = 32.0, 72.0
LON_START, LON_END = -25.0, 45.0
GRID_LAT = np.round(np.arange(LAT_START, LAT_END + 1e-6, 0.25), 3)
GRID_LON = np.round(np.arange(LON_START, LON_END + 1e-6, 0.25), 3)


def _get_dir() -> Path:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--data_dir", default="pre_train_exports-0.00493-weightedspecies", type=Path)
    ns, _ = p.parse_known_args()
    return ns.data_dir.resolve()


DATA_DIR = _get_dir()


def _plot_maps(arr_pred: np.ndarray, arr_gt: np.ndarray, lats: np.ndarray, lons: np.ndarray, title: str) -> plt.Figure:
    """Quick-look two-panel plot (no grid, low DPI)."""
    arr_pred = np.asarray(arr_pred).squeeze()
    arr_gt = np.asarray(arr_gt).squeeze()
    proj = ccrs.PlateCarree()
    fig, ax = plt.subplots(1, 2, figsize=(10, 4), subplot_kw=dict(projection=proj))
    for a, dat, lbl in zip(ax, (arr_pred, arr_gt), ("Prediction", "Ground Truth")):
        cyc, lon_cyc = add_cyclic_point(dat, coord=lons)
        mesh = a.pcolormesh(lon_cyc, lats, cyc, cmap="viridis", transform=proj)
        a.add_feature(cfeature.COASTLINE, lw=0.4)
        a.set_title(f"{title}\n{lbl}")
    fig.colorbar(mesh, ax=ax.ravel().tolist(), shrink=0.75)
    return fig


def _plot_maps_informative(
    arr_pred: np.ndarray, arr_gt: np.ndarray, lats: np.ndarray, lons: np.ndarray, var_group: str, var_name: str, timestamp: str
) -> plt.Figure:
    """High-quality two-panel plot with grid, labels and right-hand colourbar.
    """
    proj = ccrs.PlateCarree()
    fig, ax = plt.subplots(1, 2, figsize=(12, 4), subplot_kw=dict(projection=proj))
    # Draw panels and keep last mesh for colourbar
    mesh = None
    for i, (a, dat, lbl) in enumerate(zip(ax, (arr_pred.squeeze(), arr_gt.squeeze()), ("Prediction", "Ground Truth"))):

        cyc, lon_cyc = add_cyclic_point(dat, coord=lons)
        mesh = a.pcolormesh(lon_cyc, lats, cyc, cmap="viridis", transform=proj)
        a.add_feature(cfeature.COASTLINE, lw=0.4)

        gl = a.gridlines(draw_labels=True, linewidth=0.5, color="gray", linestyle="--")
        gl.top_labels = gl.right_labels = False

        if i == 0:  # left panel
            a.set_ylabel("Latitude (°N)")
        else:  # right panel
            gl.left_labels = False  # hide lat tick-labels
            a.set_ylabel("")

        a.set_xlabel("Longitude (°E)")
        a.set_title(lbl)

    cax = inset_axes(
        ax[-1],  # parent = right map
        width="2.5%",
        height="100%",  # bar is 2.5 % wide, full height
        loc="lower left",
        bbox_to_anchor=(1.02, 0.0, 1, 1),  # just outside the map
        bbox_transform=ax[-1].transAxes,
        borderpad=0,
    )
    fig.colorbar(mesh, cax=cax, label="Distribution density")
    # Super‑title
    fig.suptitle(f"Variable Group: {var_group} | Variable: {var_name} — Timestamp: {timestamp}", fontsize=12)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    return fig


def _flatten(pred: torch.Tensor, gt: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
    pr = pred.detach().cpu().numpy().ravel()
    ob = gt.detach().cpu().numpy().ravel()
    mask = np.isfinite(pr) & np.isfinite(ob)
    return pr[mask], ob[mask]


def _compute_metrics(pred: torch.Tensor, gt: torch.Tensor) -> Dict[str, float]:
    p, o = _flatten(pred, gt)
    rmse = float(np.sqrt(mean_squared_error(o, p)))
    mae = float(mean_absolute_error(o, p))
    bias = float(np.mean(p - o))
    corr = float(np.corrcoef(o, p)[0, 1])
    climo = np.full_like(o, np.mean(o))
    skill = 1.0 - np.sum((p - o) ** 2) / np.sum((climo - o) ** 2)
    return {"RMSE": rmse, "MAE": mae, "Bias": bias, "Corr": corr, "Skill": skill}


####### STRAEAMLIT APP

st.set_page_config(page_title="Prediction evaluation viewer", layout="wide")

st.sidebar.title("Prediction viewer - extended v3")
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
timestamp = str(meta["timestamp"][0])
slot_list = [k for k in pred.keys() if k.endswith("_variables")]
slot = st.sidebar.selectbox("Variable group", slot_list)
var_list = list(pred[slot].keys())
var_sel = st.sidebar.multiselect("Variable(s)", var_list, default=[var_list[0]])
if not var_sel:
    st.info("Pick variable(s)")
    st.stop()

pl_sel = None
sample_tensor = next(iter(pred[slot].values()))
if sample_tensor.ndim == 4:
    pl_vals = meta["pressure_levels"]
    pl_sel = st.sidebar.multiselect("Pressure hPa", pl_vals, default=[pl_vals[0]])
    if not pl_sel:
        st.stop()

st.header(f"{file_sel} — {timestamp}")

TAB_NAMES = (
    "Spatial Maps",
    "Spatial Maps Informative",
    "Metrics Summary",
    "Taylor Diagram",
    "Error Distribution",
    "Species Time-series",
    "Spatial Maps Zoom"
)
(tab_spatial, tab_spatial_inf, tab_metrics, tab_taylor, tab_error, tab_species, tab_spatial_zoom) = st.tabs(TAB_NAMES)

# Spatial Maps ----------------------------------------------------------------
with tab_spatial:
    for v in var_sel:
        ten_pred, ten_gt = pred[slot][v], gt[slot][v][0]
        if ten_pred.ndim == 3:
            st.pyplot(_plot_maps(ten_pred[0], ten_gt[0], lats, lons, v))
        else:
            assert pl_sel is not None
            for pl in pl_sel:
                ci = meta["pressure_levels"].index(pl)
                st.pyplot(_plot_maps(ten_pred[0, ci], ten_gt[0, ci], lats, lons, f"{v} {pl} hPa"))

# Spatial Maps Informative -----------------------------------------------------
with tab_spatial_inf:
    st.info("Download >= 300 dpi figures via the buttons below.")
    for v in var_sel:
        ten_pred, ten_gt = pred[slot][v], gt[slot][v][0]
        if ten_pred.ndim == 3:
            fig = _plot_maps_informative(ten_pred[0], ten_gt[0], lats, lons, slot, v, timestamp)
            st.pyplot(fig)
            buf = io.BytesIO()
            fig.savefig(buf, dpi=300, format="png", bbox_inches="tight")
            st.download_button(
                "Download PNG (300 dpi)", buf.getvalue(), file_name=f"{slot}_{v}_{timestamp}.png", mime="image/png"
            )
        else:
            assert pl_sel is not None
            for pl in pl_sel:
                ci = meta["pressure_levels"].index(pl)
                fig = _plot_maps_informative(ten_pred[0, ci], ten_gt[0, ci], lats, lons, slot, f"{v}_{pl} hPa", timestamp)
                st.pyplot(fig)
                buf = io.BytesIO()
                fig.savefig(buf, dpi=300, format="png", bbox_inches="tight")
                st.download_button(
                    "Download PNG (300 dpi)", buf.getvalue(), file_name=f"{slot}_{v}_{pl}hPa_{timestamp}.png", mime="image/png"
                )


with tab_spatial_zoom:
    st.markdown("### Select geographic window")

    lat_min = st.number_input("Latitude start (°N)", value=float(lats.min()),
                              min_value=float(lats.min()),
                              max_value=float(lats.max())-0.25, step=0.25)
    lat_max = st.number_input("Latitude end  (°N)", value=float(lats.max()),
                              min_value=lat_min+0.25,
                              max_value=float(lats.max()), step=0.25)
    lon_min = st.number_input("Longitude start (°E)", value=float(lons.min()),
                              min_value=float(lons.min()),
                              max_value=float(lons.max())-0.25, step=0.25)
    lon_max = st.number_input("Longitude end   (°E)", value=float(lons.max()),
                              min_value=lon_min+0.25,
                              max_value=float(lons.max()), step=0.25)

    lat_mask = (lats >= lat_min) & (lats <= lat_max)
    lon_mask = (lons >= lon_min) & (lons <= lon_max)
    if lat_mask.sum() < 2 or lon_mask.sum() < 2:
        st.warning("Zoom window too small. Increase extent."); st.stop()

    lats_sub = lats[lat_mask]
    lons_sub = lons[lon_mask]

    for v in var_sel:
        ten_pred, ten_gt = pred[slot][v], gt[slot][v][0]

        def _crop(tensor):  # helper slices H×W last dims
            return tensor[..., lat_mask, :][..., lon_mask]

        if ten_pred.ndim == 3:                                   # surface var
            fig = _plot_maps_informative(
                _crop(ten_pred[0]), _crop(ten_gt[0]),
                lats_sub, lons_sub, slot, v, timestamp)
            st.pyplot(fig)
        else:                                                    # pressure var
            assert pl_sel is not None
            for pl in pl_sel:
                ci = meta["pressure_levels"].index(pl)
                fig = _plot_maps_informative(
                    _crop(ten_pred[0, ci]), _crop(ten_gt[0, ci]),
                    lats_sub, lons_sub, slot, f"{v}_{pl} hPa", timestamp)
                st.pyplot(fig)


with tab_metrics:
    st.subheader("Deterministic metrics across windows")
    recs: List[Dict[str, float]] = []
    for f in files:
        w = torch.load(f, map_location="cpu")
        p, g = w["pred"][slot][var_sel[0]], w["gt"][slot][var_sel[0]][0]
        if p.ndim == 4:
            ci = meta["pressure_levels"].index(pl_sel[0]) if pl_sel else 0
            p, g = p[0, ci], g[0, ci]
        else:
            p, g = p[0], g[0]
        m = _compute_metrics(p, g)
        m["window"] = f.name
        recs.append(m)
    dfm = pd.DataFrame(recs).set_index("window")
    st.dataframe(dfm.style.format("{:.5f}"))
    fig_bar, ax = plt.subplots(figsize=(6, 3))
    dfm.mean().plot(kind="bar", ax=ax)
    ax.set_ylabel("Mean score")
    ax.set_title("Deterministic skill summary")
    st.pyplot(fig_bar)

# Taylor Diagram --------------------------------------------------------------
with tab_taylor:
    st.subheader("Taylor diagram — pattern statistics")
    if TAYLOR_BACKEND is None:
        st.warning("Install either easy_mpl (preferred) or skillmetrics to enable this plot.")
    elif TAYLOR_BACKEND == "easy_mpl":
        # Build observation array (first window) and simulations dict
        w0 = torch.load(files[0], map_location="cpu")
        obs = w0["gt"][slot][var_sel[0]][0]
        if obs.ndim == 4:
            ci = meta["pressure_levels"].index(pl_sel[0]) if pl_sel else 0
            obs = obs[0, ci].detach().cpu().numpy().ravel()
        else:
            obs = obs[0].detach().cpu().numpy().ravel()
        sims: Dict[str, np.ndarray] = {}
        for f in files:
            w = torch.load(f, map_location="cpu")
            p = w["pred"][slot][var_sel[0]]
            if p.ndim == 4:
                ci = meta["pressure_levels"].index(pl_sel[0]) if pl_sel else 0
                p_arr = p[0, ci].detach().cpu().numpy().ravel()
            else:
                p_arr = p[0].detach().cpu().numpy().ravel()
            sims[Path(f).stem] = p_arr
        fig_tay = em_taylor_plot(obs, sims, show=False)
        st.pyplot(fig_tay)
    else:  # skillmetrics fallback
        ref_std, std_ratio, corrcoef, names = [], [], [], []
        for f in files:
            w = torch.load(f, map_location="cpu")
            p = w["pred"][slot][var_sel[0]]
            g = w["gt"][slot][var_sel[0]][0]
            if p.ndim == 4:
                ci = meta["pressure_levels"].index(pl_sel[0]) if pl_sel else 0
                p = p[0, ci].detach().cpu().numpy()
                g = g[0, ci].detach().cpu().numpy()
            else:
                p = p[0].detach().cpu().numpy()
                g = g[0].detach().cpu().numpy()
            ref_std.append(np.std(g))
            std_ratio.append(np.std(p) / np.std(g))
            corrcoef.append(np.corrcoef(p.ravel(), g.ravel())[0, 1])
            names.append(Path(f).stem)
        fig_tay = plt.figure(figsize=(5, 5))
        sm.taylor_diagram(std_ratio, corrcoef, markerLabel=names, markerColor="k", refstd=ref_std[0])
        st.pyplot(fig_tay)

# Error PDF -------------------------------------------------------------------
with tab_error:
    st.subheader("Spatial error PDF (Prediction - Observation)")
    ten_pred, ten_gt = pred[slot][var_sel[0]], gt[slot][var_sel[0]][0]
    if ten_pred.ndim == 4:
        ci = meta["pressure_levels"].index(pl_sel[0]) if pl_sel else 0
        err = (ten_pred[0, ci] - ten_gt[0, ci]).detach().cpu().numpy().ravel()
    else:
        err = (ten_pred[0] - ten_gt[0]).detach().cpu().numpy().ravel()
    err = err[np.isfinite(err)]
    fig_err, ax_err = plt.subplots(figsize=(6, 3))
    ax_err.hist(err, bins=60, density=True, alpha=0.6)
    ax_err.set_xlabel("Error")
    ax_err.set_ylabel("Probability density")
    ax_err.set_title("Spatial error PDF")
    ax_err.grid(True)
    st.pyplot(fig_err)

# Species cumulative mean -----------------------------------------------------
with tab_species:
    spec_slot = "species_variables"
    if spec_slot not in pred:
        st.info("No species_variables slot present.")
    else:
        all_species: set[str] = set()
        for f in files:
            w = torch.load(f, map_location="cpu")
            all_species.update(w["gt"][spec_slot].keys())
        species_id = st.sidebar.selectbox("Species ID", sorted(all_species))

        def _scalar_ts(ts):
            if isinstance(ts, (list, tuple, np.ndarray)):
                ts = ts[0]
            if hasattr(ts, "__iter__") and not isinstance(ts, (str, bytes)):
                ts = ts[0]
            return pd.Timestamp(ts)

        recs = []
        for f in files:
            w = torch.load(f, map_location="cpu")
            ts = _scalar_ts(w["gt"]["batch_metadata"]["timestamp"][0])
            if species_id not in w["gt"][spec_slot]:
                continue
            recs.append(
                {
                    "timestamp": ts,
                    "GT": w["gt"][spec_slot][species_id][0].float().mean().item(),
                    "PRED": w["pred"][spec_slot][species_id][0].float().mean().item(),
                }
            )
        if recs:
            df = pd.DataFrame(recs).set_index("timestamp").astype(float).sort_index()
            df[["GT_ma", "PRED_ma"]] = df[["GT", "PRED"]].expanding().mean()
            fig_ts = plt.figure(figsize=(8, 3))
            plt.plot(df.index, df["GT_ma"], lw=2, label="GT")
            plt.plot(df.index, df["PRED_ma"], lw=2, label="PRED")
            plt.title(f"Cumulative mean — species {species_id}")
            plt.ylabel("Distribution mean")
            plt.grid(True)
            plt.legend()
            st.pyplot(fig_ts)
        else:
            st.warning(f"No data for species {species_id} in these windows.")
