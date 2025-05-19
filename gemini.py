# =============================================================
#   Candlestick Annotator – Modified Version
#   Requires: streamlit, pandas, plotly, pathlib, datetime, zoneinfo
#   AND crucially: streamlit_plotly_events
# =============================================================
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo

# --- page config --------------------------------------------------------
st.set_page_config(page_title="Candlestick Annotator", layout="wide")

# --- plotly-events component (re-integrated for drag/relayout support) -----------
try:
    from streamlit_plotly_events import plotly_events
except ModuleNotFoundError:
    st.error("Install streamlit-plotly-events in this venv:\n"
             "    pip install streamlit-plotly-events")
    st.stop()

# -------------------- constants --------------------------------------
TIMEFRAMES = ["M1", "M5", "M15", "M30", "H1", "H4", "D1", "W1", "MN"]

LABEL_OPTIONS = {           # label : (Excel code, marker colour)
    "True":      ("T",  "lime"),
    "False":     ("F",  "crimson"),
    "TrueFalse": ("TF", "gold"),
}

SESSION_INFO = {
    "Tokyo":     {"tz": "Asia/Tokyo",       "start": time(9), "end": time(18),
                   "color": "rgba(102,153,255,0.25)"},
    "London":    {"tz": "Europe/London",    "start": time(8), "end": time(17),
                   "color": "rgba(102,255,178,0.25)"},
    "New York":  {"tz": "America/New_York", "start": time(8), "end": time(17),
                   "color": "rgba(255,204,102,0.25)"},
}
BAR_H = 0.02  # 2 % of plot height  (≈10–12 px)

# -------------------- session-state ----------------------------------
state_defaults = {
    "dataframes":   {},
    "annotations":  {tf: {} for tf in TIMEFRAMES},
    "timeframe":    "H1",
    "window_sizes": {tf: 120 for tf in TIMEFRAMES}, # Now represents the desired window size
    "center_time":  None, # Initialized to None, will be set on first CSV load
    "autosave_dir": "annotations",
    "dt_constant":  "G&s",
    "candles_to_move": 20, # New default for navigation step
}
for k, v in state_defaults.items():
    st.session_state.setdefault(k, v)

# -------------------- helpers ----------------------------------------
def load_csv(file, key: str):
    """Read a MT4/5-style CSV and cache it."""
    df = (pd.read_csv(file)
            .dropna()
            .assign(Date=lambda d: pd.to_datetime(d["Date"]))
            .sort_values("Date")
            .reset_index(drop=True))
    st.session_state.dataframes[key] = df
    st.session_state.annotations.setdefault(key, {})
    
    # Logic to initialize/update center_time when a new CSV is loaded
    # Try to keep the current center if it exists, otherwise center the new data.
    if st.session_state.center_time is None:
        st.session_state.center_time = str(df["Date"].iloc[len(df)//2])
    else:
        current_center_dt = pd.to_datetime(st.session_state.center_time)
        if not df.empty:
            closest_idx = (df["Date"] - current_center_dt).abs().idxmin()
            st.session_state.center_time = str(df["Date"].iloc[closest_idx])
        else:
            st.session_state.center_time = None # No data in new CSV


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

    csv = st.file_uploader("Candlestick CSV", type="csv", key="csv_uploader")
    if csv:
        tf_csv = st.selectbox("CSV timeframe", TIMEFRAMES,
                              index=TIMEFRAMES.index("H1"), key="csv_tf_select")
        key = st.text_input("Dataset key", value=tf_csv, key="csv_dataset_key")
        if st.button("Add CSV", key="add_csv_button"):
            load_csv(csv, key)
            st.success(f"Loaded {key}")
            st.rerun() # Rerun to display newly loaded data immediately

    xlsx = st.file_uploader("Import annotations (xlsx)", type=["xls", "xlsx"], key="xlsx_uploader")
    if xlsx and st.button("Import annotations", key="import_annotations_button"):
        load_xlsx(xlsx)
        st.success("Imported")
        st.rerun() # Rerun to display imported annotations

    st.text_input("Excel D/T constant", key="dt_constant")
    st.markdown("---")

    keys = list(st.session_state.dataframes.keys())
    if not keys:
        st.info("📄 Load a CSV to begin.")
        st.stop()

    # Active timeframe
    tf_now = st.radio("Active timeframe", keys,
                      index=keys.index(st.session_state.timeframe)
                      if st.session_state.timeframe in keys else 0, key="active_timeframe_radio")
    
    # Logic to keep center_time when changing timeframe
    if tf_now != st.session_state.timeframe:
        old_center_dt = pd.to_datetime(st.session_state.center_time)
        
        # Update current timeframe in session state
        st.session_state.timeframe = tf_now 

        # Find closest point in the new dataframe to the old center time
        new_df = st.session_state.dataframes[tf_now]
        if not new_df.empty:
            closest_idx = (new_df["Date"] - old_center_dt).abs().idxmin()
            st.session_state.center_time = str(new_df["Date"].iloc[closest_idx])
        else:
            st.session_state.center_time = None # No data in new timeframe

        # Rerun to update the chart with the new timeframe centered
        st.rerun()


    # window + nav
    win = st.slider("Window size", 10, 500,
                    value=st.session_state.window_sizes.get(tf_now, 120), key=f"window_size_slider_{tf_now}")
    st.session_state.window_sizes[tf_now] = win # Update session state

    # New: Candles to move forward/backward
    st.number_input("Candles to move", min_value=1, max_value=win, 
                    value=st.session_state.candles_to_move, 
                    key="candles_to_move_input", 
                    on_change=lambda: st.session_state.update(candles_to_move=st.session_state.candles_to_move))


    left, right = st.columns(2)
    with left:
        if st.button("⏪ Back", key="back_button"):
            df_current = st.session_state.dataframes[tf_now]
            if not df_current.empty:
                idx = (df_current["Date"] - pd.to_datetime(
                                   st.session_state.center_time)).abs().idxmin()
                
                # Move by candles_to_move
                new_idx = max(idx - st.session_state.candles_to_move, 0)
                st.session_state.center_time = str(
                    df_current["Date"].iloc[new_idx])
                st.rerun()
    with right:
        if st.button("Forward ⏩", key="forward_button"):
            df_current = st.session_state.dataframes[tf_now]
            if not df_current.empty:
                idx = (df_current["Date"] - pd.to_datetime(
                                   st.session_state.center_time)).abs().idxmin()
                
                # Move by candles_to_move
                new_idx = min(idx + st.session_state.candles_to_move, len(df_current) - 1)
                st.session_state.center_time = str(
                    df_current["Date"].iloc[new_idx])
                st.rerun()

    st.markdown("---")
    label_pick = st.radio("Label", list(LABEL_OPTIONS.keys()), horizontal=True, key="label_picker")
    st.markdown("---")

    st.text_input("Autosave folder", key="autosave_dir_input")
    fname = st.text_input("Manual save filename", value="annotations", key="manual_save_filename")
    if st.button("💾 Save now", key="save_now_button"):
        Path(st.session_state.autosave_dir).mkdir(parents=True, exist_ok=True)
        save_xlsx(Path(st.session_state.autosave_dir) / f"{fname}.xlsx")
        st.success("Saved")

# -------------------- main plot --------------------------------------
center_dt = pd.to_datetime(st.session_state.center_time)
df_active = st.session_state.dataframes[tf_now]

fig = go.Figure(layout={"template": "plotly_dark", "height": 600}) # Fixed height for plot

if not df_active.empty:
    # Ensure center_dt is within dataframe bounds if possible for initial index lookup
    if center_dt < df_active["Date"].iloc[0]:
        center_dt = df_active["Date"].iloc[0]
    elif center_dt > df_active["Date"].iloc[-1]:
        center_dt = df_active["Date"].iloc[-1]

    idx = (df_active["Date"] - center_dt).abs().idxmin()

    # Calculate start and end indices to maintain desired window size (win)
    # This logic aims to get 'win' candles centered around 'idx',
    # adjusting if it hits dataframe boundaries.
    start_idx = max(0, idx - win // 2)
    end_idx = min(len(df_active) - 1, idx + win // 2)

    # If the initial slice is smaller than 'win' due to boundaries,
    # expand it on the other side if possible.
    if (end_idx - start_idx + 1) < win:
        if start_idx == 0: # At start boundary, expand right
            end_idx = min(len(df_active) - 1, start_idx + win - 1)
        elif end_idx == len(df_active) - 1: # At end boundary, expand left
            start_idx = max(0, end_idx - win + 1)
    
    # Final check to ensure we don't go out of bounds for slicing, especially if 'win' is large
    # and dataset is small.
    start_idx = max(0, start_idx)
    end_idx = min(len(df_active) - 1, end_idx)

    # Ensure dff has at least one candle if df_active is not empty
    if df_active.empty:
        dff = pd.DataFrame()
    else:
        dff = df_active.iloc[start_idx : end_idx + 1]

    # candlesticks
    if not dff.empty: # Only add candlestick trace if dff is not empty
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

        # existing annotations (dot only, smaller size)
        for iso, m in st.session_state.annotations.get(tf_now, {}).items():
            dt = pd.to_datetime(iso)
            # Only display if within the current visible data range
            if not dff.empty and dff["Date"].iloc[0] <= dt <= dff["Date"].iloc[-1]:
                fig.add_trace(go.Scatter(
                    x=[dt], y=[m["y"]], mode="markers", # Removed text
                    marker=dict(size=3, # Smaller size
                                 color=LABEL_OPTIONS[m["label"]][1]),
                    showlegend=False,
                    name=f"Annotation: {m['label']}" # Name for hover info if needed
                    ))

fig.update_xaxes(type="date", rangeslider_visible=False)
fig.update_layout(hovermode="x unified", dragmode="pan", # Keep dragmode pan for user interaction
                  title=f"{tf_now} Chart")

plot_cfg = {"scrollZoom": True, "displaylogo": False}

# -------------------- interaction (with streamlit_plotly_events) -----
# Using plotly_events to capture both clicks and relayout (pan/zoom) events
event_data = plotly_events(
    fig,
    key="plot",
    override_height=fig.layout.height, # Ensure the chart height matches the figure
    click_event=True, # Enable click events for annotations
    relayout_event=True, # Enable relayout events for pan/zoom
    select_event=False, # Disable native selection, we handle clicks
    hover_event=False,
    use_container_width=True,
    config=plot_cfg,
)

# Process events from plotly_events
if event_data:
    # A single plotly_events call returns a list of events. We process the first one.
    event = event_data[0] 

    if "points" in event: # This is a click event (e.g., for annotation)
        clicked_point = event["points"][0]
        iso   = str(pd.to_datetime(clicked_point["x"]))
        price = float(clicked_point["y"]) # Captures the y-value of the clicked data point

        amap = st.session_state.annotations[tf_now]
        if iso in amap:
            amap.pop(iso) # Toggle -> remove existing annotation
        else:
            amap.update({iso: {"label": label_pick, "y": price}})

        # Autosave annotations on click
        Path(st.session_state.autosave_dir).mkdir(parents=True, exist_ok=True)
        save_xlsx(Path(st.session_state.autosave_dir) / "autosave.xlsx")
        st.rerun() # Rerun to update annotations on plot

    elif "xaxis.range[0]" in event and "xaxis.range[1]" in event:
        # This is a relayout event (pan/zoom by dragging the chart)
        x_min_str = event["xaxis.range[0]"]
        x_max_str = event["xaxis.range[1]"]

        # Convert string ranges to datetime objects
        x_min_dt = pd.to_datetime(x_min_str)
        x_max_dt = pd.to_datetime(x_max_str)

        # Calculate new center time
        new_center_time = x_min_dt + (x_max_dt - x_min_dt) / 2
        st.session_state.center_time = str(new_center_time)

        # Calculate new window size based on the new range
        # Find the actual start and end indices of the visible range in the full dataframe
        df_dates = df_active["Date"]
        
        # Use searchsorted for efficient lookup of indices within the full DataFrame
        start_idx_visible = df_dates.searchsorted(x_min_dt, side='left')
        end_idx_visible = df_dates.searchsorted(x_max_dt, side='right') - 1
        
        # Ensure indices are within bounds
        start_idx_visible = max(0, start_idx_visible)
        end_idx_visible = min(len(df_dates) - 1, end_idx_visible)

        if end_idx_visible >= start_idx_visible:
            # new_win_size represents the count of candles in the visible range
            new_win_size = end_idx_visible - start_idx_visible + 1
            # Update the desired window size in session state (for slider and future navigation)
            # Clamp the value to the slider's min/max to prevent issues
            st.session_state.window_sizes[tf_now] = max(10, min(500, new_win_size)) 
        
        # No need to rerun here as relayout events often come in quick succession.
        # The next interaction (click, button) or Streamlit's natural rerun cycle
        # will pick up the updated center_time and window_size.
        # However, to immediately update the view after a drag, st.rerun() is necessary.
        st.rerun()


st.caption("Candlestick Annotator – sessions shown as thin bars at top "
           "• click candles to label")