#!/usr/bin/env python3
import glob, os, time
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, dcc, html, Output, Input

SNAP_ROOT = "./snapshots"  # same folder as the writer

def latest_parquet():
    files = sorted(glob.glob(os.path.join(SNAP_ROOT, "**", "adsb_mr_*_10min.parquet"), recursive=True))
    return files[-1] if files else None

def build_fig(df: pd.DataFrame):
    # Expect columns: t_rx, hex, flight, lat, lon, altitude, speed, track, vert_rate, ...
    # Group by aircraft (prefer callsign, fallback to hex), draw line per aircraft
    traces = []
    if df.empty:
        return go.Figure(data=[go.Scatter3d(x=[],y=[],z=[],mode="markers",name="no data")])

    # Clean/sort
    df = df.dropna(subset=["lat","lon","altitude"]).sort_values(["flight","hex","t_rx"])
    df["name"] = df["flight"].fillna("")  # callsign when present
    df.loc[df["name"].eq(""), "name"] = df["hex"]  # fallback to hex

    for name, g in df.groupby("name", sort=False):
        traces.append(go.Scatter3d(
            x=g["lon"], y=g["lat"], z=g["altitude"],
            mode="lines",
            name=str(name),
            line=dict(width=4)
        ))

    fig = go.Figure(data=traces)
    fig.update_layout(
        uirevision="stay",
        scene=dict(
            xaxis_title="Longitude",
            yaxis_title="Latitude",
            zaxis_title="Altitude (ft)",
        ),
        margin=dict(l=0,r=0,t=30,b=0),
        showlegend=True,
        title="ADS-B 3D Tracks (Latest 10-minute Snapshot)"
    )
    return fig

app = Dash(__name__)
app.layout = html.Div([
    html.H3("ADS-B 3D (latest 10-min snapshot)"),
    html.Div(id="fileinfo"),
    dcc.Graph(id="g3d", style={"height": "78vh", "width": "100%"}),
    dcc.Interval(id="tick", interval=15_000, n_intervals=0)
])

@app.callback(
    [Output("g3d","figure"), Output("fileinfo","children")],
    [Input("tick","n_intervals")]
)
def refresh(_):
    path = latest_parquet()
    if not path:
        return build_fig(pd.DataFrame()), "No snapshot file found yet."
    try:
        df = pd.read_parquet(path)
        info = f"Using: {os.path.basename(path)}"
        return build_fig(df), info
    except Exception as e:
        return build_fig(pd.DataFrame()), f"Error reading {path}: {e}"

if __name__ == "__main__":
    # Run on the Pi; we’ll forward this port to the Mac
    app.run(host="0.0.0.0", port=8050, debug=False)
