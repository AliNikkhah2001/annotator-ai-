import streamlit as st, pandas as pd, plotly.graph_objects as go
from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo
from pathlib import Path

st.set_page_config("Candlestick Annotator", layout="wide")

try:
    from streamlit_plotly_events import plotly_events
except ModuleNotFoundError:
    st.error("pip install streamlit-plotly-events"); st.stop()

TIMEFRAMES = ["M1","M5","M15","M30","H1","H4","D1","W1","MN"]
LABELS = {"True":("T","rgba(0,0,0,0.45)"),
          "False":("F","rgba(0,0,0,0.45)"),
          "TrueFalse":("TF","rgba(0,0,0,0.45)")}
SESSION = {
    "Tokyo":{"tz":"Asia/Tokyo","start":time(9),"end":time(18),
             "color":"rgba(102,153,255,0.25)"},
    "London":{"tz":"Europe/London","start":time(8),"end":time(17),
             "color":"rgba(102,255,178,0.25)"},
    "New York":{"tz":"America/New_York","start":time(8),"end":time(17),
             "color":"rgba(255,204,102,0.25)"}}
BAR_H, BUFFER_FACTOR = 0.02, 3

for k,v in {
    "dataframes":{},
    "annotations":{tf:{} for tf in TIMEFRAMES},
    "timeframe":"H1",
    "window_sizes":{tf:200 for tf in TIMEFRAMES},
    "center_time":None,
}.items(): st.session_state.setdefault(k,v)

def load_csv(f,key):
    df=(pd.read_csv(f).dropna()
          .assign(Date=lambda d: pd.to_datetime(d["Date"]))
          .sort_values("Date").reset_index(drop=True))
    st.session_state.dataframes[key]=df
    st.session_state.center_time=str(df["Date"].iloc[len(df)//2])

def sess_bounds(cfg,day_utc):
    tz=ZoneInfo(cfg["tz"])
    ld=day_utc.astimezone(tz).date()
    s=datetime.combine(ld,cfg["start"],tz)
    e=datetime.combine(ld,cfg["end"],tz)
    if e<=s:e+=timedelta(days=1)
    return s.astimezone(ZoneInfo("UTC")).replace(tzinfo=None),\
           e.astimezone(ZoneInfo("UTC")).replace(tzinfo=None)

def make_fig(tf,center,win,debug=False):
    df=st.session_state.dataframes[tf]
    fig=go.Figure(layout={"template":"plotly_dark"})
    if df.empty: return fig
    idx=(df["Date"]-center).abs().idxmin()
    dff=df.iloc[max(idx-win*BUFFER_FACTOR//2,0): idx+win*BUFFER_FACTOR//2+1]

    if debug:
        st.subheader("DEBUG – slice info")
        st.write(f"rows: {len(dff)}   first: {dff['Date'].iloc[0]}   "
                 f"last: {dff['Date'].iloc[-1]}")
        st.write(f"low: {dff['Low'].min()}  high: {dff['High'].max()}")
        st.write(dff.head())

    # dynamic body width
    width_ms=0 if len(dff)<2 else \
        0.8*dff["Date"].diff().dt.total_seconds().dropna().median()*1000

    ctrace=go.Candlestick(
        x=dff["Date"],open=dff["Open"],high=dff["High"],
        low=dff["Low"],close=dff["Close"],width=width_ms,
        increasing=dict(line=dict(color="#00c800",width=1),
                        fillcolor="rgba(0,200,0,0.8)"),
        decreasing=dict(line=dict(color="#dc2828",width=1),
                        fillcolor="rgba(220,40,40,0.8)"),
        showlegend=False
    )
    fig.add_trace(ctrace)

    if debug: st.write("Candlestick trace dict:", ctrace.to_plotly_json())

    y0,y1=dff["Low"].min(),dff["High"].max()
    half=win//2
    fig.update_xaxes(range=[
        dff["Date"].iloc[max(len(dff)//2-half,0)],
        dff["Date"].iloc[min(len(dff)//2+half,len(dff)-1)]
    ])

    for d in pd.to_datetime(dff["Date"].dt.date.unique()):
        fig.add_shape(type="line",x0=d,x1=d,y0=y0,y1=y1,
                      line=dict(color="white",dash="dash"),layer="below")
    for i,cfg in enumerate(SESSION.values()):
        for d in pd.to_datetime(dff["Date"].dt.date.unique()):
            sx,ex=sess_bounds(cfg,datetime.combine(d,time(0),tzinfo=ZoneInfo("UTC")))
            y0p,y1p=1-(i+1)*BAR_H,1-i*BAR_H
            fig.add_shape(type="rect",x0=sx,x1=ex,xref="x",
                          y0=y0p,y1=y1p,yref="paper",fillcolor=cfg["color"],
                          line=dict(width=0),layer="above")
    return fig

# ───── Sidebar
with st.sidebar:
    csv=st.file_uploader("CSV",type="csv")
    if csv and st.button("Add CSV"): load_csv(csv,"data")
    keys=list(st.session_state.dataframes.keys())
    if not keys: st.info("Upload CSV"); st.stop()
    tf_now=st.radio("TF",keys,index=0)
    win=st.slider("Window",50,1000,200)
    debug=st.checkbox("Debug",value=False)

centre=pd.to_datetime(st.session_state.center_time or datetime.utcnow())
fig=make_fig(tf_now,centre,win,debug)

plotly_events(fig,key="plot",click_event=True,
              override_height=700,override_width="100%")
