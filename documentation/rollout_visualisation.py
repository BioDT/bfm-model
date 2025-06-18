import io
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import torch
import xarray as xr
from plotly.subplots import make_subplots

PRE_DIR = Path("pre-train_test_exports")
ROLL_DIR = Path("rollouts_export")


def normalize_record_entry(entry):
    """
    Takes either a dict or a namedtuple Batch, returns a dict of variable groups.
    """
    if isinstance(entry, dict):
        return entry
    elif hasattr(entry, "_asdict"):
        d = entry._asdict()
        d.pop("batch_metadata", None)
        return d
    else:
        raise ValueError(f"Cannot normalize record entry of type {type(entry)}")


def plot_raw_triple(var_name: str, pred_tensor: torch.Tensor, gt_tensor: torch.Tensor, timestamps: list) -> go.Figure:
    """
    Plot Ground Truth, Prediction, and Difference side by side.
    """
    # Extract 2D arrays | .float() as its bfloat16 and numpy doesnt like it
    p_arr = pred_tensor.detach().cpu().float().numpy()
    p = p_arr[0, 0] if p_arr.ndim == 4 else p_arr[0]
    g_arr = gt_tensor.detach().cpu().float().numpy()
    g1 = g_arr[0, 0] if g_arr.ndim == 4 else g_arr[0]
    diff = g1 - p
    # Flip vertical for correct orientation
    p = np.flipud(p)
    g1 = np.flipud(g1)
    diff = np.flipud(diff)
    # Timestamp label
    t_label = timestamps[-1] if isinstance(timestamps, (list, tuple)) and timestamps else ""
    # Build subplots
    fig = make_subplots(rows=1, cols=3, subplot_titles=(f"Ground Truth @ {t_label}", f"Prediction @ {t_label}", "Difference"))
    # Add traces
    for data, colorscale, col in [(g1, "Viridis", 1), (p, "Viridis", 2), (diff, "thermal", 3)]:
        img = px.imshow(data, color_continuous_scale=colorscale)
        for trace in img.data:
            fig.add_trace(trace, row=1, col=col)
    fig.update_layout(title_text=f"{var_name} Comparison", showlegend=False, margin=dict(l=10, r=10, t=40, b=10), height=400)
    # Equal aspect
    for c in [1, 2, 3]:
        fig.update_xaxes(constrain="domain", row=1, col=c)
        fig.update_yaxes(scaleanchor="x", row=1, col=c)
    return fig


st.set_page_config(page_title="Rollout Explorer", layout="wide")
st.title("🔍 Rollout Forecast Explorer")


### TABs Setup
tab1, tab2 = st.tabs(["Rollout Explorer", "Pre-trained Visualizer"])


with tab1:
    if not ROLL_DIR.exists():
        st.error(f"Export directory not found: {ROLL_DIR}")
        st.stop()

    files = sorted(ROLL_DIR.glob("window_*.pt"))
    window_indices = [int(p.stem.split("_")[1]) for p in files]
    st.sidebar.header("🗂 Select Window Index")
    window_idx = st.sidebar.selectbox("Window", window_indices)
    records = torch.load(ROLL_DIR / f"window_{window_idx:05d}.pt", map_location="cpu")
    num_steps = len(records)

    first_pred = records[0]["pred"]
    FIXED_GROUPS = [
        "surface_variables",
        "single_variables",
        "atmospheric_variables",
        "species_extinction_variables",
        "land_variables",
        "agriculture_variables",
        "forest_variables",
        "species_variables",
    ]
    ALL_GROUPS = [g for g in FIXED_GROUPS if hasattr(first_pred, g) and getattr(first_pred, g)]

    st.sidebar.header("⚙️ Visualization Settings")
    step = st.sidebar.slider("Roll‑out Step", 1, num_steps, 1)
    group = st.sidebar.selectbox("Variable Group", ALL_GROUPS)
    vars_available = list(getattr(first_pred, group).keys())
    var = st.sidebar.selectbox("Variable", vars_available)

    # check if variable has extra channel dim
    sample_arr = getattr(first_pred, group)[var]
    # dims: [B, T, C?, H, W]
    isatmos = sample_arr.dim() == 5
    if isatmos:
        num_levels = sample_arr.shape[2]
        level = st.sidebar.slider("Channel/Level index", 0, num_levels - 1, 0)
        levels_meta = records[0]["pred"].batch_metadata.pressure_levels
        lvl_label = levels_meta[level] if hasattr(levels_meta, "__len__") else level
    else:
        level = None

    def batch_to_xr(batch, group, var, level=None):
        """Convert a Batch variable to an xarray DataArray with proper coords."""
        arr = getattr(batch, group)[var]
        if arr.dim() == 5 and level is not None:
            data = arr[0, -1, level].numpy()
        else:
            data = arr[0, -1].numpy()

        lats = batch.batch_metadata.latitudes.numpy()
        lons = batch.batch_metadata.longitudes.numpy()
        # if coords are 2D grids, collapse intelligently
        if lats.ndim > 1:
            # if more rows than columns, take first column; else first row
            if lats.shape[0] > 1:
                lats = lats[:, 0]
            else:
                lats = lats[0, :]
        if lons.ndim > 1:
            if lons.shape[1] > 1:
                lons = lons[0, :]
            else:
                lons = lons[:, 0]

        H, W = data.shape
        if lats.ndim != 1 or lats.size != H:
            lats = np.arange(H)
        if lons.ndim != 1 or lons.size != W:
            lons = np.arange(W)

        da = xr.DataArray(
            data,
            coords={
                "lat": ("lat", lats),
                "lon": ("lon", lons),
            },
            dims=("lat", "lon"),
            name=var,
        )
        return da

    st.subheader("🗺️ Prediction vs Ground Truth")
    rec_pred = records[step - 1]["pred"]
    rec_gt = records[step - 1]["gt"]
    t_pred = rec_pred.batch_metadata.timestamp
    t_gt = rec_gt.batch_metadata.timestamp
    ts_pred = t_pred[-1] if isinstance(t_pred, (list, tuple)) else t_pred
    ts_gt = t_gt[-1] if isinstance(t_gt, (list, tuple)) else t_gt

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**Prediction at {ts_pred}**")
        da_pred = batch_to_xr(rec_pred, group, var, level)
        fig1 = px.imshow(da_pred, origin="lower", aspect="equal", color_continuous_scale="Viridis")
        fig1.update_layout(margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig1, use_container_width=True)
    with col2:
        st.markdown(f"**Ground Truth at {ts_gt}**")
        da_gt = batch_to_xr(rec_gt, group, var, level)
        fig2 = px.imshow(da_gt, origin="lower", aspect="equal", color_continuous_scale="Viridis")
        fig2.update_layout(margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig2, use_container_width=True)

    # ----------------------------------------------------------------------------------
    # Separate Timeline Animations & GIF Download
    # ----------------------------------------------------------------------------------

    animation_type = st.sidebar.radio("Choose timeline to animate", ["Prediction", "Ground Truth"])
    st.subheader(f"⏯️ {animation_type} Timeline Animation")
    key = "pred" if animation_type == "Prediction" else "gt"

    # Duration: total ~12s
    total_secs = 12.0
    num = num_steps
    per_frame = total_secs / max(num, 1)

    # Display a Plotly preview
    frames_plotly = []
    for rec in records:
        batch = rec[key]
        da = batch_to_xr(batch, group, var, level)
        frame_fig = px.imshow(da, origin="lower", aspect="equal", color_continuous_scale="Viridis")
        frame_fig.update_layout(
            title=str(
                rec[key].batch_metadata.timestamp[-1]
                if isinstance(rec[key].batch_metadata.timestamp, (list, tuple))
                else rec[key].batch_metadata.timestamp
            ),
            margin=dict(l=0, r=0, t=30, b=0),
        )
        frames_plotly.append(frame_fig)

    # Construct frames for animation
    anim_frames = []
    for i, frame_fig in enumerate(frames_plotly):
        anim_frames.append(go.Frame(data=frame_fig.data, name=str(i), layout=frame_fig.layout))
    # Initial figure
    anim_fig = go.Figure(
        data=anim_frames[0].data,
        frames=anim_frames,
        layout=go.Layout(
            updatemenus=[
                {
                    "type": "buttons",
                    "buttons": [
                        {
                            "label": "Play",
                            "method": "animate",
                            "args": [None, {"frame": {"duration": per_frame * 1000, "redraw": True}, "fromcurrent": True}],
                        }
                    ],
                }
            ]
        ),
    )
    anim_fig.update_layout(title_text=f"{animation_type} Timeline", margin=dict(l=0, r=0, t=30, b=0))
    st.plotly_chart(anim_fig, use_container_width=True)

    # GIF generation and download
    if st.button("Generate & Download GIF"):
        buf_frames = []
        for rec in records:
            batch = rec[key]
            da = batch_to_xr(batch, group, var, level)
            fig = px.imshow(da, origin="lower", aspect="equal", color_continuous_scale="Viridis")
            fig.update_layout(margin=dict(l=0, r=0, t=20, b=0), coloraxis_showscale=False)
            # render to PNG in-memory
            buf = io.BytesIO()
            fig.write_image(buf, format="png", scale=2)
            buf.seek(0)
            img = imageio.imread(buf)
            buf_frames.append(img)
        # create GIF
        gif_buf = io.BytesIO()
        imageio.mimsave(gif_buf, buf_frames, format="GIF", duration=per_frame)
        gif_buf.seek(0)
        st.image(gif_buf.getvalue(), format="GIF")
        st.download_button(
            f"⬇️ Download {animation_type} GIF",
            data=gif_buf.getvalue(),
            file_name=f"{animation_type.lower()}_timeline.gif",
            mime="image/gif",
        )

    # ----------------------------------------------------------------------------------
    # Error growth over time (timestamps)
    # ----------------------------------------------------------------------------------
    st.subheader("📈 Error Growth")
    time_x, errs = [], []
    for rec in records:
        ts = rec["pred"].batch_metadata.timestamp
        ts_lbl = ts[-1] if isinstance(ts, (list, tuple)) else ts
        da_p = batch_to_xr(rec["pred"], group, var, level).values
        da_g = batch_to_xr(rec["gt"], group, var, level).values
        time_x.append(ts_lbl)
        errs.append(float(np.sqrt(((da_p - da_g) ** 2).mean())))

    err_fig = go.Figure()
    err_fig.add_trace(go.Scatter(x=time_x, y=errs, mode="lines+markers"))
    err_fig.update_layout(xaxis_title="Timestamp", yaxis_title="RMSE", margin=dict(l=0, r=0, t=30, b=0))
    st.plotly_chart(err_fig, use_container_width=True)

# --- TAB2: Pretrained Visualizer ---
with tab2:
    st.header("📈 Pre-trained Predictions")
    if not PRE_DIR.exists():
        st.error(f"Directory not found: {PRE_DIR}")
    else:
        files = sorted(PRE_DIR.glob("window_*.pt"))
        idxs = [int(p.stem.split("_")[1]) for p in files]
        win = st.selectbox("Window index", idxs, key="pre_win")
        recs = torch.load(PRE_DIR / f"window_{win:05d}.pt", map_location="cpu")

    num_pre = len(recs)
    # Step selector
    if num_pre > 1:
        pre_step = st.slider("Pretrained step", 1, num_pre, 1, key="pre_step")
    else:
        st.write("Only one pretrained record available (step 1)")
        pre_step = 1

    pred_entry, gt_entry = rec["pred"], rec["gt"]
    # Normalize pred and gt entries
    # TODO Ground Truth = BAtch and Prediction = Dict
    pd_pre = normalize_record_entry(pred_entry)
    gd_pre = normalize_record_entry(gt_entry)

    groups_pre = list(pd_pre.keys())
    grp_pre = st.selectbox("Variable Group", groups_pre, key="pre_grp")
    vars_pre = list(pd_pre[grp_pre].keys())
    var_pre = st.selectbox("Variable", vars_pre, key="pre_var")

    tns_pre = gt_entry.batch_metadata.timestamp

    fig_pre = plot_raw_triple(var_pre, pd_pre[grp_pre][var_pre], gd_pre[grp_pre][var_pre], tns_pre)
    st.plotly_chart(fig_pre, use_container_width=True)

    png_pre = fig_pre.to_image(format="png")
    st.download_button(
        f"Download {var_pre}", data=png_pre, file_name=f"pre_{grp_pre}_{var_pre}_step{pre_step}.png", mime="image/png"
    )
