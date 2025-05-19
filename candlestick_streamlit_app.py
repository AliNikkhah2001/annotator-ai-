#!/usr/bin/env python3
# Candlestick-annotator – visible candles + pan/zoom on any streamlit-plotly-events version
import streamlit as st, pandas as pd, plotly.graph_objects as go, inspect
from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo
from pathlib import Path

st.set_page_config(page_title="Candlestick Annotator", layout="wide")

try:
    from streamlit_plotly_events import plotly_events
except ModuleNotFoundError:
    st.error("pip install streamlit-plotly-events"); st.stop()

# ── API-agnostic wrapper ──────────────────────────────────────
def plotly_events_auto(fig, key: str):
    """Call plotly_events() correctly on any release (0.0.6 … master)."""
    base = dict(override_height=700, override_width="100%", key=key)
    sig  = inspect.signature(plotly_events).parameters
    if "relayout_event" in sig:         # GitHub master
        return plotly_events(fig, click_event=True, relayout_event=True, **base)
    if "events" in sig:                 # dev/pre-0.0.7 wheels
        return plotly_events(fig, events=["click", "relayout"], **base)
    return plotly_events(fig, click_event=True, **base)   # PyPI 0.0.6 – no relayout

# ── constants ─────────────────────────────────────────────────
TF   = ["M1","M5","M15","M30","H1","H4","D1","W1","MN"]
LBL  = {"True":("T","rgba(0,0,0,0.45)"),
        "False":("F","rgba(0,0,0,0.45)"),
        "TrueFalse":("TF","rgba(0,0,0,0.45)")}
SES  = {"Tokyo":  {"tz":"Asia/Tokyo","start":time(9),"end":time(18),
                   "color":"rgba(102,153,255,0.25)"},
        "London": {"tz":"Europe/London","start":time(8),"end":time(17),
                   "color":"rgba(102,255,178,0.25)"},
        "New York":{"tz":"America/New_York","start":time(8),"end":time(17),
                   "color":"rgba(255,204,102,0.25)"}}
BAR_H, BUF = 0.02, 3

# ── session defaults ─────────────────────────────────────────
for k,v in dict(
        dataframes={}, annotations={tf:{} for tf in TF}, timeframe="H1",
        window_sizes={tf:200 for tf in TF}, center_time=None,
        autosave_dir="annotations", dt_constant="G&s").items():
    st.session_state.setdefault(k,v)

# ── IO helpers ───────────────────────────────────────────────
def load_csv(file, key):
    """
    Robust CSV reader:
    • accepts 'Date' or 'Date'+'Time' columns
    • converts OHLC to float
    • **synthesises missing Open** from previous Close, then (High+Low)/2
    """
    raw = pd.read_csv(file)

    # build timestamp
    if {"Date","Time"}.issubset(raw.columns):
        ts = raw["Date"].astype(str)+" "+raw["Time"].astype(str)
    elif "Date" in raw.columns:
        ts = raw["Date"]
    else:
        st.error("CSV needs 'Date' or 'Date'+'Time' columns"); return

    df = (raw.assign(Date=pd.to_datetime(ts, errors="coerce"))
              .rename(columns=lambda s: s.strip().title())
              .dropna(subset=["Date"]))

    # enforce numeric OHLC where present
    for col in ["Open","High","Low","Close"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # ── NEW: synthesise Open if absent/NaN ────────────────────
    if "Open" in df.columns and df["Open"].isna().all():
        df["Open"] = df["Close"].shift(1)
    if "Open" in df.columns:
        df["Open"].fillna((df["High"] + df["Low"]) / 2, inplace=True)

    df = df.dropna(subset=["Open","High","Low","Close"]) \
           .sort_values("Date").reset_index(drop=True)
    if df.empty:
        st.error("No valid rows after cleaning – check the CSV."); return

    st.session_state.dataframes[key] = df
    st.session_state.annotations.setdefault(key, {})
    st.session_state.center_time = str(df["Date"].iloc[len(df)//2])

def load_xlsx(p):
    need={"DATE","TIME","PRICE","T/T","T/F"}; df=pd.read_excel(p)
    if not need.issubset(df.columns): st.warning("XLSX missing cols"); return
    rev={v[0]:k for k,v in LBL.items()}
    for _,r in df.iterrows():
        iso=pd.to_datetime(f"{r['DATE']} {r['TIME']}"); tf=str(r["T/F"]).strip()
        lab=rev.get(str(r["T/T"]).strip())
        if lab:
            st.session_state.annotations.setdefault(tf,{})[str(iso)]={"label":lab,"y":float(r["PRICE"])}

def save_xlsx(path:Path):
    rows=[]
    for tf, amap in st.session_state.annotations.items():
        for i,(iso,m) in enumerate(sorted(amap.items()),1):
            d=pd.to_datetime(iso)
            rows.append({"NUMBER":i,"D/T":st.session_state.dt_constant,
                         "T/F":tf,"T/T":LBL[m["label"]][0],
                         "DATE":d.date(),"TIME":d.time(),"PRICE":m["y"]})
    pd.DataFrame(rows).to_excel(path,index=False)

# ── chart helpers ────────────────────────────────────────────
def sess_bounds(cfg, day_utc):
    tz=ZoneInfo(cfg["tz"]); d=day_utc.astimezone(tz).date()
    s=datetime.combine(d,cfg["start"],tz); e=datetime.combine(d,cfg["end"],tz)
    if e<=s: e+=timedelta(days=1)
    return (s.astimezone(ZoneInfo("UTC")).replace(tzinfo=None),
            e.astimezone(ZoneInfo("UTC")).replace(tzinfo=None))

def make_fig(tf, centre, win):
    df=st.session_state.dataframes[tf]; fig=go.Figure(layout={"template":"plotly_dark"})
    if df.empty: return fig
    idx=(df["Date"]-centre).abs().idxmin(); half=win//2
    dff=df.iloc[max(idx-BUF*half,0):idx+BUF*half+1].dropna(subset=["Open","High","Low","Close"])
    if dff.empty: return fig
    fig.add_trace(go.Candlestick(x=dff["Date"],open=dff["Open"],high=dff["High"],
                                 low=dff["Low"],close=dff["Close"],
                                 increasing=dict(line_color="rgb(0,200,0)",fillcolor="rgba(0,200,0,0.8)"),
                                 decreasing=dict(line_color="rgb(220,40,40)",fillcolor="rgba(220,40,40,0.8)"),
                                 line=dict(width=1)))
    y0,y1=dff["Low"].min(),dff["High"].max()
    fig.update_xaxes(range=[dff["Date"].iloc[max(len(dff)//2-half,0)],
                            dff["Date"].iloc[min(len(dff)//2+half,len(dff)-1)]])
    for d in pd.to_datetime(dff["Date"].dt.date.unique()):
        fig.add_shape(type="line",x0=d,x1=d,y0=y0,y1=y1,
                      line=dict(color="white",dash="dash"),layer="below")
    for i,cfg in enumerate(SES.values()):
        for d in pd.to_datetime(dff["Date"].dt.date.unique()):
            sx,ex=sess_bounds(cfg,datetime.combine(d,time(0),tzinfo=ZoneInfo("UTC")))
            y0p,y1p=1-(i+1)*BAR_H,1-i*BAR_H
            fig.add_shape(type="rect",x0=sx,x1=ex,xref="x",y0=y0p,y1=y1p,yref="paper",
                          fillcolor=cfg["color"],line=dict(width=0),layer="above")
    for iso,m in st.session_state.annotations.get(tf,{}).items():
        t=pd.to_datetime(iso)
        if t in dff["Date"].values:
            fig.add_shape(type="line",x0=t,x1=t,y0=y0,y1=y1,
                          line=dict(color=LBL[m["label"]][1],width=2),layer="above")
    fig.update_layout(hovermode="x unified",dragmode="pan",
                      margin=dict(t=30,r=10,b=30,l=10),title=f"{tf} Chart")
    return fig

def relayout_shift(ev,tf):
    if "xaxis.range[0]" not in ev: return
    s,e=pd.to_datetime(ev["xaxis.range[0]"]),pd.to_datetime(ev["xaxis.range[1]"])
    if pd.isna(s)|pd.isna(e): return
    st.session_state.center_time=str(s+(e-s)/2)
    df=st.session_state.dataframes[tf]
    if len(df)>1:
        step=df["Date"].diff().median()
        if pd.notna(step) and step: st.session_state.window_sizes[tf]=max(int((e-s)/step),50)
# ── sidebar ──────────────────────────────────────────────────
with st.sidebar:
    st.header("Load data")
    csv=st.file_uploader("Candlestick CSV",type="csv")
    if csv:
        tf_csv=st.selectbox("Timeframe of CSV",TF,index=TF.index("H1"))
        key=st.text_input("Dataset key",value=tf_csv)
        if st.button("Add CSV"): load_csv(csv,key); st.success("Loaded "+key)

    xlsx=st.file_uploader("Import annotations (xlsx)",type=["xls","xlsx"])
    if xlsx and st.button("Import"): load_xlsx(xlsx); st.success("Imported")

    st.text_input("Excel D/T constant",key="dt_constant"); st.markdown("---")

    keys=list(st.session_state.dataframes.keys())
    if not keys: st.info("Load a CSV."); st.stop()

    tf_now=st.radio("Active timeframe",keys,
                    index=keys.index(st.session_state.timeframe)
                    if st.session_state.timeframe in keys else 0)
    if tf_now!=st.session_state.timeframe:
        mid=len(st.session_state.dataframes[tf_now])//2
        st.session_state.center_time=str(st.session_state.dataframes[tf_now]["Date"].iloc[mid])
    st.session_state.timeframe=tf_now

    win=st.slider("Window size (candles)",50,1000,
                  value=st.session_state.window_sizes.get(tf_now,200))
    st.session_state.window_sizes[tf_now]=win

    back,fwd=st.columns(2)
    with back:
        if st.button("⏪ Back"):
            df=st.session_state.dataframes[tf_now]
            idx=(df["Date"]-pd.to_datetime(st.session_state.center_time)).abs().idxmin()
            st.session_state.center_time=str(df["Date"].iloc[max(idx-win,0)])
    with fwd:
        if st.button("Forward ⏩"):
            df=st.session_state.dataframes[tf_now]
            idx=(df["Date"]-pd.to_datetime(st.session_state.center_time)).abs().idxmin()
            st.session_state.center_time=str(df["Date"].iloc[min(idx+win,len(df)-1)])

    st.markdown("---")
    label_pick=st.radio("Label",list(LBL.keys()),horizontal=True)
    st.markdown("---")
    st.text_input("Autosave folder",key="autosave_dir")
    fname=st.text_input("Manual save filename",value="annotations")
    if st.button("💾 Save"):
        Path(st.session_state.autosave_dir).mkdir(parents=True,exist_ok=True)
        save_xlsx(Path(st.session_state.autosave_dir)/f"{fname}.xlsx")
        st.success("Saved")

# ── main plot & event loop ───────────────────────────────────
centre=pd.to_datetime(st.session_state.center_time or datetime.utcnow())
fig=make_fig(tf_now,centre,win)
events=plotly_events_auto(fig,key="plot")

for ev in events:
    if "x" in ev and "y" in ev and "xaxis.range[0]" not in ev:   # click → annotation
        iso,y=ev["x"],ev["y"]; amap=st.session_state.annotations[tf_now]
        amap.pop(iso,None) if iso in amap else amap.update({iso:{"label":label_pick,"y":y}})
        Path(st.session_state.autosave_dir).mkdir(parents=True,exist_ok=True)
        save_xlsx(Path(st.session_state.autosave_dir)/"autosave.xlsx")
        st.experimental_rerun()
    elif "xaxis.range[0]" in ev:                                 # pan/zoom
        relayout_shift(ev,tf_now); st.experimental_rerun()

st.caption("Drag to pan • Zoom with wheel • Click candle to toggle annotation • "
           "Session bars top: Tokyo / London / NY")
