# =============================================================
#  Candlestick Annotator – Modified Version
#  Changes implemented per user specifications
# =============================================================
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo

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
    "center_time":  None,
    "autosave_dir": "annotations",
    "dt_constant":  "G&s",
    "step_size":    30,  # New step size parameter
}
for k, v in state_defaults.items():
    st.session_state.setdefault(k, v)

# -------------------- helpers ----------------------------------------
def load_csv(file, key: str):
    """Read a MT4/5-style CSV and cache it."""
    df = (pd.read_csv(file)
            .dropna()
            .assign(Date=lambda d: pd.to_datetime(d["Date"]))
    df = df.sort_values("Date").reset_index(drop=True)
    st.session_state.dataframes[key] = df
    st.session_state.annotations.setdefault(key, {})
    if not st.session_state.center_time:
        st.session_state.center_time = str(df["Date"].iloc[len(df)//2])

def load_xlsx(xf):
    """Import annotation spreadsheet created by this app."""
    need = {"DATE", "TIME", "PRICE", "T/T", "T/F"}
    df = pd.read_excel(xf)
    if not need.issubset(df.columns):
        st.warning("Excel missing columns: " + ", ".join(sorted(need))); return
    rev = {v[0]: k for k, v in LABEL_OPTIONS.items()}
    for _, r in df.iterrows():
        iso = pd.to_datetime(f"{r['DATE']} {r['TIME']}")
        tf  = str(r["T/F"]).strip()
        lab = rev.get(str(r["T/T"]).strip())
        if lab:
            st.session_state.annotations.setdefault(tf, {})[str(iso)] = {
                "label": lab, "y": float(r["PRICE"])
            }

def save_xlsx(path: Path):
    """Write annotations back to XLSX (autosave and manual)."""
    rows = []
    for tf, amap in st.session_state.annotations.items():
        for i, (iso, m) in enumerate(sorted(amap.items()), 1):
            d = pd.to_datetime(iso)
            rows.append({"NUMBER": i, "D/T": st.session_state.dt_constant,
                         "T/F": tf,  "T/T": LABEL_OPTIONS[m["label"]][0],
                         "DATE": d.date(), "TIME": d.time(), "PRICE": m["y"]})
    pd.DataFrame(rows).to_excel(path, index=False)

def sess_bounds(cfg, day_utc):
    """Return UTC start/end datetimes for a trading session on a given UTC day."""
    tz = ZoneInfo(cfg["tz"])
    local_date = day_utc.astimezone(tz).date()
    s = datetime.combine(local_date, cfg["start"], tz)
    e = datetime.combine(local_date, cfg["end"],   tz)
    if e <= s:
        e += timedelta(days=1)
    return (s.astimezone(ZoneInfo("UTC")).replace(tzinfo=None),
            e.astimezone(ZoneInfo("UTC")).replace(tzinfo=None))

# -------------------- sidebar UI -------------------------------------
with st.sidebar:
    st.header("Data")

    csv = st.file_uploader("Candlestick CSV", type="csv")
    if csv:
        tf_csv = st.selectbox("CSV timeframe", TIMEFRAMES,
                              index=TIMEFRAMES.index("H1"))
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

    # Active timeframe
    tf_now = st.radio("Active timeframe", keys,
                      index=keys.index(st.session_state.timeframe)
                      if st.session_state.timeframe in keys else 0)
    st.session_state.timeframe = tf_now

    # Window controls
    win = st.slider("Window size", 10, 500,
                    value=st.session_state.window_sizes.get(tf_now, 120))
    st.session_state.window_sizes[tf_now] = win
    
    # Step size control
    st.session_state.step_size = st.number_input(
        "Navigation step size",
        min_value=1,
        max_value=win,
        value=st.session_state.get("step_size", 30),
        help="Number of candles to move per navigation click"
    )

    left, right = st.columns(2)
    with left:
        if st.button("⏪ Back"):
            df = st.session_state.dataframes[tf_now]
            idx = df[df["Date"] == pd.to_datetime(st.session_state.center_time)].index[0]
            new_idx = max(idx - st.session_state.step_size, 0)
            st.session_state.center_time = str(df["Date"].iloc[new_idx])
    with right:
        if st.button("Forward ⏩"):
            df = st.session_state.dataframes[tf_now]
            idx = df[df["Date"] == pd.to_datetime(st.session_state.center_time)].index[0]
            new_idx = min(idx + st.session_state.step_size, len(df) - 1)
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
df = st.session_state.dataframes[tf_now]

fig = go.Figure(layout={"template": "plotly_dark"})
if not df.empty:
    # Find nearest index to center time
    idx = (df["Date"] - center_dt).abs().idxmin()
    
    # Calculate window boundaries
    start_idx = max(idx - st.session_state.window_sizes[tf_now] // 2, 0)
    end_idx = start_idx + st.session_state.window_sizes[tf_now]
    end_idx = min(end_idx, len(df))
    
    # Adjust start index if window exceeds dataframe length
    if end_idx - start_idx < st.session_state.window_sizes[tf_now]:
        start_idx = max(end_idx - st.session_state.window_sizes[tf_now], 0)
    
    dff = df.iloc[start_idx:end_idx]

    # candlesticks
    fig.add_trace(go.Candlestick(
        x=dff["Date"], open=dff["Open"], high=dff["High"],
        low=dff["Low"], close=dff["Close"], opacity=1))

    y0, y1 = dff["Low"].min(), dff["High"].max()

    # day separators
    for day in pd.to_datetime(dff["Date"].dt.date.unique())[1:]:
        fig.add_shape(type="line", x0=day, x1=day, y0=y0, y1=y1,
                      line=dict(color="white", dash="dash"), layer="below")

    # session bars
    for i, cfg in enumerate(SESSION_INFO.values()):
        for day in pd.to_datetime(dff["Date"].dt.date.unique()):
            start_utc = datetime.combine(day, time(0),
                                         tzinfo=ZoneInfo("UTC"))
            sx, ex = sess_bounds(cfg, start_utc)
            if ex < dff["Date"].iloc[0] or sx > dff["Date"].iloc[-1]:
                continue
            y0_p, y1_p = 1 - (i + 1) * BAR_H, 1 - i * BAR_H
            fig.add_shape(type="rect", x0=sx, x1=ex, xref="x",
                          y0=y0_p, y1=y1_p, yref="paper",
                          fillcolor=cfg["color"], line=dict(width=0),
                          layer="above")

    # existing annotations (smaller dots without text)
    for iso, m in st.session_state.annotations.get(tf_now, {}).items():
        dt = pd.to_datetime(iso)
        if dt in dff["Date"].values:
            fig.add_trace(go.Scatter(
                x=[dt], y=[m["y"]], mode="markers",
                marker=dict(size=3, color=LABEL_OPTIONS[m["label"]][1]),
                showlegend=False))

fig.update_xaxes(type="date", rangeslider_visible=False)
fig.update_layout(
    hovermode="x unified",
    dragmode="pan",
    title=f"{tf_now} Chart",
    xaxis=dict(range=[dff["Date"].iloc[0], dff["Date"].iloc[-1]])
)

plot_cfg = {"scrollZoom": True, "displaylogo": False}

# -------------------- interaction ------------------------------------
event = st.plotly_chart(
    fig,
    key="plot",
    on_select="rerun",
    selection_mode="points",
    use_container_width=True,
    config=plot_cfg,
)

if event and event.selection and event.selection.points:
    p = event.selection.points[0]
    iso = str(pd.to_datetime(p["x"]))
    price = float(p["y"])  # Now uses exact clicked y-value
    
    amap = st.session_state.annotations[tf_now]
    if iso in amap:
        amap.pop(iso)
    else:
        amap.update({iso: {"label": label_pick, "y": price}})
    
    Path(st.session_state.autosave_dir).mkdir(parents=True, exist_ok=True)
    save_xlsx(Path(st.session_state.autosave_dir) / "autosave.xlsx")
    st.rerun()

st.caption("Candlestick Annotator – Click candles to annotate | Drag to pan | Scroll to zoom")