# =============================================================
# Candlestick Annotator – native Streamlit chart-selection API
# (Modified 2025-05-19)
# -------------------------------------------------------------
# Changelog
#   1. Annotation dot placed at exact (x,y) click location; text removed; size 3.
#   2. View centre persists across timeframe switches; robust to missing key.
#   3. Constant window; user‑configurable step size for Back/Forward.
#   4. Infinite pan (full dataset plotted).
#   5. Increased figure height.
#   6. BUGFIX 2025‑05‑19 16:25 → avoid ValueError when default timeframe key not yet present.
# =============================================================

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from datetime import datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

# -------------------- page config ------------------------------------
st.set_page_config(page_title="Candlestick Annotator", layout="wide")

# -------------------- constants --------------------------------------
TIMEFRAMES: list[str] = ["M1", "M5", "M15", "M30", "H1", "H4", "D1", "W1", "MN"]

LABEL_OPTIONS: dict[str, tuple[str, str]] = {
    "True": ("T", "lime"),
    "False": ("F", "crimson"),
    "TrueFalse": ("TF", "gold"),
}

SESSION_INFO: dict[str, dict[str, object]] = {
    "Tokyo":    {"tz": "Asia/Tokyo",    "start": time(9), "end": time(18), "color": "rgba(102,153,255,0.25)"},
    "London":   {"tz": "Europe/London", "start": time(8), "end": time(17), "color": "rgba(102,255,178,0.25)"},
    "New York": {"tz": "America/New_York", "start": time(8), "end": time(17), "color": "rgba(255,204,102,0.25)"},
}
BAR_H = 0.02  # relative height for session bars
DEFAULT_FIG_HEIGHT = 800

# -------------------- session-state ----------------------------------
state_defaults = {
    "dataframes": {},
    "annotations": {tf: {} for tf in TIMEFRAMES},
    "timeframe": "H1",
    "window_sizes": {tf: 120 for tf in TIMEFRAMES},
    "step_sizes": {tf: 60 for tf in TIMEFRAMES},
    "center_time": None,  # ISO-string
    "autosave_dir": "annotations",
    "dt_constant": "G&s",
    "show_sessions": "True",
    "show_daylines": "True"
}  
for k, v in state_defaults.items():
    st.session_state.setdefault(k, v)

# -------------------- helpers ----------------------------------------

def load_csv(file, key: str) -> None:
    """Read a MT4/5-style CSV and cache it under *key*."""
    df = (
        pd.read_csv(file)
        .dropna()
        .assign(Date=lambda d: pd.to_datetime(d["Date"]))
        .sort_values("Date")
        .reset_index(drop=True)
    )
    st.session_state.dataframes[key] = df
    st.session_state.annotations.setdefault(key, {})
    # initialise centre if not set yet
    if st.session_state.center_time is None:
        st.session_state.center_time = str(df["Date"].iloc[len(df) // 2])
    # if no valid timeframe yet, pick this one
    if st.session_state.timeframe not in st.session_state.dataframes:
        st.session_state.timeframe = key


def load_xlsx(xf) -> None:
    """Import annotation spreadsheet created by this app."""
    required = {"DATE", "TIME", "PRICE", "T/T", "T/F"}
    df = pd.read_excel(xf)
    if not required.issubset(df.columns):
        st.warning("Excel missing columns: " + ", ".join(sorted(required)))
        return

    rev = {v[0]: k for k, v in LABEL_OPTIONS.items()}
    for _, r in df.iterrows():
        iso = pd.to_datetime(f"{r['DATE']} {r['TIME']}")
        tf = str(r["T/F"]).strip()
        lab = rev.get(str(r["T/T"]).strip())
        if lab:
            st.session_state.annotations.setdefault(tf, {})[str(iso)] = {
                "label": lab,
                "y": float(r["PRICE"]),
            }


def save_xlsx(path: Path) -> None:
    """Write all annotations to *path* (XLSX)."""
    rows: list[dict[str, object]] = []
    for tf, amap in st.session_state.annotations.items():
        for i, (iso, m) in enumerate(sorted(amap.items()), 1):
            d = pd.to_datetime(iso)
            rows.append(
                {
                    "NUMBER": i,
                    "D/T": st.session_state.dt_constant,
                    "T/F": tf,
                    "T/T": LABEL_OPTIONS[m["label"]][0],
                    "DATE": d.date(),
                    "TIME": d.time(),
                    "PRICE": m["y"],
                }
            )
    pd.DataFrame(rows).to_excel(path, index=False)


def sess_bounds(cfg: dict[str, object], day_utc: datetime) -> tuple[datetime, datetime]:
    """UTC start/end datetimes for a trading session on *day_utc*."""
    tz = ZoneInfo(cfg["tz"])
    local_date = day_utc.astimezone(tz).date()
    s = datetime.combine(local_date, cfg["start"], tz)
    e = datetime.combine(local_date, cfg["end"], tz)
    if e <= s:
        e += timedelta(days=1)
    return (
        s.astimezone(ZoneInfo("UTC")).replace(tzinfo=None),
        e.astimezone(ZoneInfo("UTC")).replace(tzinfo=None),
    )

# -------------------- sidebar UI -------------------------------------
with st.sidebar:
    st.header("Data")

    csv = st.file_uploader("Candlestick CSV", type="csv")
    if csv:
        tf_csv = st.selectbox("CSV timeframe", TIMEFRAMES, index=TIMEFRAMES.index("H1"))
        key = st.text_input("Dataset key", value=tf_csv)
        if st.button("Add CSV"):
            load_csv(csv, key)
            st.success(f"Loaded {key}")

    xlsx = st.file_uploader("Import annotations (xlsx)", type=["xls", "xlsx"])
    if xlsx and st.button("Import annotations"):
        load_xlsx(xlsx)
        st.success("Imported")

    st.text_input("Excel D/T constant", key="dt_constant")
    st.markdown("---")
    st.session_state.how_sessions = st.checkbox("Show session bars", value=True)
    st.session_state.show_daylines = st.checkbox("Show day separators", value=True)
    st.markdown("---")
    left, right = st.columns(2)
    tf_now = st.session_state.timeframe
    step = st.session_state.step_sizes[tf_now] 
    with left:
        if st.button("⏪ Back"):
            tf_now = st.session_state.timeframe
            df = st.session_state.dataframes[tf_now]
            idx = (df["Date"] - pd.to_datetime(st.session_state.center_time)).abs().idxmin()
            new_idx = max(idx - step, 0)
            st.session_state.center_time = str(df["Date"].iloc[new_idx])
    with right:
        if st.button("Forward ⏩"):
            tf_now = st.session_state.timeframe
            df = st.session_state.dataframes[tf_now]
            idx = (df["Date"] - pd.to_datetime(st.session_state.center_time)).abs().idxmin()
            new_idx = min(idx + step, len(df) - 1)
            st.session_state.center_time = str(df["Date"].iloc[new_idx])
    st.markdown("---")
    keys = list(st.session_state.dataframes.keys())
    if not keys:
        st.info("📄 Load a CSV to begin.")
        st.stop()

    # Active timeframe (dataset)
    default_idx = keys.index(st.session_state.timeframe) if st.session_state.timeframe in keys else 0
    tf_now = st.radio("Active timeframe", keys, index=default_idx)
    st.session_state.timeframe = tf_now  # always update

    # ensure centre makes sense for the new dataset
    if st.session_state.center_time is None:
        tf_now = st.session_state.timeframe
        mid = len(st.session_state.dataframes[tf_now]) // 2
        st.session_state.center_time = str(st.session_state.dataframes[tf_now]["Date"].iloc[mid])

    # window size (number of candles)
    win = st.slider("Window size (candles)", 10, 500, value=st.session_state.window_sizes.get(tf_now, 120))
    st.session_state.window_sizes[tf_now] = win

    # step size (how many candles move per click)
    step = st.number_input("Step size (candles)", min_value=1, max_value=win, value=st.session_state.step_sizes.get(tf_now, win // 2))
    st.session_state.step_sizes[tf_now] = step



    st.markdown("---")
    label_pick = st.radio("Label", list(LABEL_OPTIONS.keys()), horizontal=True)
    st.markdown("---")

    st.text_input("Autosave folder", key="autosave_dir")
    fname = st.text_input("Manual save filename", value="annotations")
    if st.button("💾 Save now"):
        Path(st.session_state.autosave_dir).mkdir(parents=True, exist_ok=True)
        save_xlsx(Path(st.session_state.autosave_dir) / f"{fname}.xlsx")
        st.success("Saved")

# -------------------- main plot --------------------------------------
center_dt = pd.to_datetime(st.session_state.center_time)
df = st.session_state.dataframes[tf_now]
if df.empty:
    st.warning("Dataset is empty.")
    st.stop()

# find centre index & window slice
idx = (df["Date"] - center_dt).abs().idxmin()
win = st.session_state.window_sizes[tf_now]
half = win // 2
start_idx = max(idx - half, 0)
end_idx = min(start_idx + win, len(df))
# adjust start if we hit the end
if end_idx - start_idx < win:
    start_idx = max(end_idx - win, 0)

visible_start = df["Date"].iloc[start_idx]
visible_end = df["Date"].iloc[end_idx - 1]
dff = df.iloc[start_idx:end_idx]  # window slice for helper lines/shapes

fig = go.Figure(layout={"template": "plotly_dark"})
fig.update_layout(height=750, hovermode="x unified", dragmode="pan", title=f"{tf_now} Chart",margin=dict(l=20,r=20,t=20,b=20))






buffer = 2 * win  # e.g., preload ±window_size
start_idx_buffered = max(0, idx - buffer)
end_idx_buffered = min(len(df), idx + buffer)

df_buf = df.iloc[start_idx_buffered:end_idx_buffered]

fig.add_trace(
    go.Candlestick(
        x=df_buf["Date"],
        open=df_buf["Open"],
        high=df_buf["High"],
        low=df_buf["Low"],
        close=df_buf["Close"],
        opacity=0.4,
    )
)
# Repaint clicked candles with full opacity
for iso, m in st.session_state.annotations.get(tf_now, {}).items():
    dt = pd.to_datetime(iso)
    row = df_buf[df_buf["Date"] == dt]
    if not row.empty:
        row = row.iloc[0]
        fig.add_trace(go.Candlestick(
            x=[row["Date"]],
            open=[row["Open"]],
            high=[row["High"]],
            low=[row["Low"]],
            close=[row["Close"]],
            opacity=1.0,
            increasing=dict(line=dict(color="lime")),
            decreasing=dict(line=dict(color="crimson")),
            showlegend=False
        ))

visible_ymin = dff["Low"].min()
visible_ymax = dff["High"].max()
fig.update_yaxes(range=[visible_ymin, visible_ymax])
print(visible_ymin,visible_ymax )


# restrict initial view range to current window
fig.update_xaxes(range=[visible_start, visible_end], rangeslider_visible=False)

# y‑range for helper overlays
y0, y1 = dff["Low"].min(), dff["High"].max()

# day separators (within current window only, for performance)
for day in pd.to_datetime(dff["Date"].dt.date.unique())[1:]:
    if st.session_state.show_daylines:
        fig.add_shape(type="line", x0=day, x1=day, y0=y0, y1=y1, line=dict(color="green", dash="dash"), layer="below")

# session bars (within window)
for i, cfg in enumerate(SESSION_INFO.values()):
    for day in pd.to_datetime(dff["Date"].dt.date.unique()):
        start_utc = datetime.combine(day, time(0), tzinfo=ZoneInfo("UTC"))
        sx, ex = sess_bounds(cfg, start_utc)
        if ex < visible_start or sx > visible_end:
            continue
        y0_p, y1_p = 1 - (i + 1) * BAR_H, 1 - i * BAR_H
        if st.session_state.show_sessions:
            fig.add_shape(type="rect", x0=sx, x1=ex, xref="x", y0=y0_p, y1=y1_p, yref="paper", fillcolor=cfg["color"], line=dict(width=1), layer="above")

# existing annotations (dots only, small size)
for iso, m in st.session_state.annotations.get(tf_now, {}).items():
    dt = pd.to_datetime(iso)
    fig.add_trace(
        go.Scatter(x=[dt], y=[m["y"]], mode="markers", marker=dict(size=5, color=LABEL_OPTIONS[m["label"]][1]), showlegend=False)
    )

plot_cfg = {"scrollZoom": True, "displaylogo": False}

# -------------------- interaction (native chart‑selection) -----------
event = st.plotly_chart(
    fig,
    key="plot",               # stable identity across reruns
    on_select="rerun",        # immediate rerun after selection
    selection_mode="points",
    use_container_width=True,
    config=plot_cfg,
    on_event="relayout"
)


if event and "relayout" in event:
    r = event["relayout"]

    if "xaxis.range[0]" in r and "xaxis.range[1]" in r:
        range_start = pd.to_datetime(r["xaxis.range[0]"])
        range_end = pd.to_datetime(r["xaxis.range[1]"])
        new_center = range_start + (range_end - range_start) / 2

        st.session_state.center_time = str(new_center)
        st.rerun()
    elif "xaxis.range" in r:
        range_start = pd.to_datetime(r["xaxis.range"][0])
        range_end = pd.to_datetime(r["xaxis.range"][1])
        new_center = range_start + (range_end - range_start) / 2

a = st.session_state.annotations[tf_now]
# process a click selection
if event and event.selection and event.selection.points:
    p0 = event.selection.points[0]
    iso_clicked = str(pd.to_datetime(p0["x"]))
    price_clicked = float(p0["y"])
    print(price_clicked)
    # Update center_time based on visible window
    st.session_state.center_time = iso_clicked

    # Toggle annotation
    if iso_clicked in a:
        a.pop(iso_clicked)
    else:
        a[iso_clicked] = {"label": label_pick, "y": price_clicked}

    Path(st.session_state.autosave_dir).mkdir(parents=True, exist_ok=True)
    save_xlsx(Path(st.session_state.autosave_dir) / "autosave.xlsx")

    st.rerun()

st.caption("Candlestick Annotator – sessions shown as thin bars at top • click candles to label")
