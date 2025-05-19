# =============================================================
#  Candlestick Annotator – v2 (2025-05-19)
#  • Exact-price annotation dots (smaller, no text)
#  • Center position preserved when switching timeframe
#  • Fixed window length & user-configurable step navigation
#  • Infinite pan/zoom: whole dataset plotted; initial view window
#  • Streamlit ≥1.34, Plotly ≥5.22
# =============================================================
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo

# -------------------- compatibility helpers -------------------------

def safe_rerun():
    """Cross-version wrapper around Streamlit's rerun API."""
    if hasattr(st, "rerun"):
        st.rerun()
    elif hasattr(st, "experimental_rerun"):
        st.experimental_rerun()
    else:
        raise RuntimeError("No rerun method in this Streamlit build.")

# -------------------- page config ------------------------------------
st.set_page_config(page_title="Candlestick Annotator", layout="wide")

# -------------------- constants --------------------------------------
TIMEFRAMES = ["M1", "M5", "M15", "M30", "H1", "H4", "D1", "W1", "MN"]

LABEL_OPTIONS = {          # label : (Excel code, marker colour)
    "True":      ("T",  "lime"),
    "False":     ("F",  "crimson"),
    "TrueFalse": ("TF", "gold"),
}

SESSION_INFO = {
    "Tokyo":    {"tz": "Asia/Tokyo",       "start": time(9), "end": time(18),
                 "color": "rgba(102,153,255,0.25)"},
    "London":   {"tz": "Europe/London",    "start": time(8), "end": time(17),
                 "color": "rgba(102,255,178,0.25)"},
    "New York": {"tz": "America/New_York", "start": time(8), "end": time(17),
                 "color": "rgba(255,204,102,0.25)"},
}
BAR_H = 0.02  # 2 % of plot height  (≈10–12 px)

# -------------------- session-state ----------------------------------
state_defaults = {
    "dataframes":   {},
    "annotations":  {tf: {} for tf in TIMEFRAMES},
    "timeframe":    "H1",
    "window_sizes": {tf: 120 for tf in TIMEFRAMES},
    "center_time":  None,        # ISO str of the candle datetime at plot centre
    "step_sizes":   {tf: 60 for tf in TIMEFRAMES},  # nav step (candles)
    "autosave_dir": "annotations",
    "dt_constant":  "G&s",
}
for k, v in state_defaults.items():
    st.session_state.setdefault(k, v)

# -------------------- helpers ----------------------------------------

def load_csv(file, key: str):
    """Read a MT4/5-style CSV to DataFrame and cache it."""
    df = (
        pd.read_csv(file)
        .dropna()
        .assign(Date=lambda d: pd.to_datetime(d["Date"]))
        .sort_values("Date")
        .reset_index(drop=True)
    )
    st.session_state.dataframes[key] = df
    st.session_state.annotations.setdefault(key, {})
    if st.session_state.center_time is None:
        st.session_state.center_time = str(df["Date"].iloc[len(df) // 2])


def load_xlsx(xf):
    """Import annotation spreadsheet created by this app."""
    need = {"DATE", "TIME", "PRICE", "T/T", "T/F"}
    df = pd.read_excel(xf)
    if not need.issubset(df.columns):
        st.warning("Excel missing columns: " + ", ".join(sorted(need)))
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


def save_xlsx(path: Path):
    """Write annotations back to XLSX (autosave & manual)."""
    rows = []
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


def sess_bounds(cfg, day_utc):
    """Return UTC start/end datetimes for a trading session on a given UTC day."""
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

    keys = list(st.session_state.dataframes.keys())
    if not keys:
        st.info("📄 Load a CSV to begin.")
        st.stop()

    # ------------------ **FIX** ensure current TF key valid ------------------
    if st.session_state.timeframe not in keys:
        st.session_state.timeframe = keys[0]
    # ------------------------------------------------------------------------

    # Active timeframe -------------------------------------------------
    tf_now = st.radio(
        "Active timeframe",
        keys,
        index=keys.index(st.session_state.timeframe),
    )
    prev_tf = st.session_state.timeframe
    st.session_state.timeframe = tf_now

    # Preserve centre position on TF switch ---------------------------
    df = st.session_state.dataframes[tf_now]
    if st.session_state.center_time is None:  # first load ever
        st.session_state.center_time = str(df["Date"].iloc[len(df) // 2])
    else:
        ct = pd.to_datetime(st.session_state.center_time)
        if ct < df["Date"].iloc[0]:
            st.session_state.center_time = str(df["Date"].iloc[0 + len(df) // 2])
        elif ct > df["Date"].iloc[-1]:
            st.session_state.center_time = str(df["Date"].iloc[-1 - len(df) // 2])

    # Window size & nav step -----------------------------------------
    win = st.slider("Window size (candles)", 10, 500, value=st.session_state.window_sizes.get(tf_now, 120))
    st.session_state.window_sizes[tf_now] = win

    step = st.slider("Step size (candles moved per ⏪/⏩)", 1, win, value=st.session_state.step_sizes.get(tf_now, max(1, win // 4)))
    st.session_state.step_sizes[tf_now] = step

    # Back/Forward buttons -------------------------------------------
    left, right = st.columns(2)
    with left:
        if st.button("⏪ Back"):
            df = st.session_state.dataframes[tf_now]
            idx = (df["Date"] - pd.to_datetime(st.session_state.center_time)).abs().idxmin()
            new_idx = max(idx - step, 0)
            st.session_state.center_time = str(df["Date"].iloc[new_idx])
    with right:
        if st.button("Forward ⏩"):
            df = st.session_state.dataframes[tf_now]
            idx = (df["Date"] - pd.to_datetime(st.session_state.center_time)).abs().idxmin()
            new_idx = min(idx + step, len(df) - 1)
            st.session_state.center_time = str(df["Date"].iloc[new_idx])

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
_df = st.session_state.dataframes[tf_now]

# Derive window slice indices (constant length) -----------------------
total = len(_df)
if total <= win:
    start_idx, end_idx = 0, total
else:
    # place centre candle at middle of window when possible
    idx_center = (_df["Date"] - center_dt).abs().idxmin()
    start_idx = idx_center - win // 2
    start_idx = max(0, min(start_idx, total - win))
    end_idx = start_idx + win

dff = _df.iloc[start_idx:end_idx]

# Build figure with *whole* dataset so pan never goes blank ------------
fig = go.Figure(layout={"template": "plotly_dark"})
fig.update_layout(height=800, hovermode="x unified", dragmode="pan", title=f"{tf_now} Chart",margin=dict(l=10,r=10,t=10,b=10))




fig.add_trace(
    go.Candlestick(
        x=_df["Date"],
        open=_df["Open"],
        high=_df["High"],
        low=_df["Low"],
        close=_df["Close"],
        opacity=1,
    )
)

# Initial viewport range
fig.update_xaxes(range=[dff["Date"].iloc[0], dff["Date"].iloc[-1]], rangeslider_visible=False)


# Overlays only for current window (perf) -----------------------------
y0, y1 = dff["Low"].min(), dff["High"].max()

# --- NEW: Tight y-axis to visible window -----------------------------
PAD = 0.10                 # 10 % head-/foot-room
spread = y1 - y0 or max(abs(y1), 1) * 0.01  # avoid zero spread
yrange = [
    y0 - spread * PAD,
    y1 + spread * PAD,
]
fig.update_yaxes(range=yrange, autorange=False)
for day in pd.to_datetime(dff["Date"].dt.date.unique())[1:]:
    fig.add_shape(type="line", x0=day, x1=day, y0=y0, y1=y1, line=dict(color="white", dash="dash"), layer="below")

for i, cfg in enumerate(SESSION_INFO.values()):
    for day in pd.to_datetime(dff["Date"].dt.date.unique()):
        start_utc = datetime.combine(day, time(0), tzinfo=ZoneInfo("UTC"))
        sx, ex = sess_bounds(cfg, start_utc)
        if ex < dff["Date"].iloc[0] or sx > dff["Date"].iloc[-1]:
            continue
        y0_p, y1_p = 1 - (i + 1) * BAR_H, 1 - i * BAR_H
        fig.add_shape(type="rect", x0=sx, x1=ex, xref="x", y0=y0_p, y1=y1_p, yref="paper", fillcolor=cfg["color"], line=dict(width=0), layer="above")

# Draw annotations (dots, small, no label text) -----------------------
for iso, m in st.session_state.annotations.get(tf_now, {}).items():
    fig.add_trace(
        go.Scatter(
            x=[pd.to_datetime(iso)],
            y=[m["y"]],
            mode="markers",
            marker=dict(size=6, color=LABEL_OPTIONS[m["label"]][1]),
            showlegend=False,
        )
    )

fig.update_layout(hovermode="x unified", dragmode="pan", title=f"{tf_now} Chart")
plot_cfg = {"scrollZoom": True, "displaylogo": False}

# -------------------- interaction (native chart-selection) -----------
# Uses Streamlit 1.34 on_select → rerun

event = st.plotly_chart(
    fig,
    key="plot",  # stable identity across reruns
    on_select="rerun",
    selection_mode="points",
    use_container_width=True,
    config=plot_cfg,
)

# After rerun selection is available ----------------------------------
if event and event.selection and event.selection.points:
    p = event.selection.points[0]
    iso = str(pd.to_datetime(p["x"]))
    price = float(p["y"])  # exact clicked y

    amap = st.session_state.annotations[tf_now]
    if iso in amap:
        amap.pop(iso)  # toggle remove
    else:
        amap[iso] = {"label": label_pick, "y": price}

    Path(st.session_state.autosave_dir).mkdir(parents=True, exist_ok=True)
    save_xlsx(Path(st.session_state.autosave_dir) / "autosave.xlsx")

    safe_rerun()

st.caption(
    "Candlestick Annotator – sessions shown as thin bars at top • click candles to label. "
    "Dots are placed exactly at the clicked price; use ⏪/⏩ or pan/zoom to navigate."
)