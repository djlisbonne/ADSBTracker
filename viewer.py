#!/usr/bin/env python3
import glob, os, time
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import math
from dash import Dash, dcc, html, Output, Input

SNAP_ROOT = "./snapshots"  # same folder as the writer
KT_TO_MPS  = 0.514444
R_EARTH    = 6371000.0

def latest_parquet():
    files = sorted(glob.glob(os.path.join(SNAP_ROOT, "**", "adsb_mr_*_10min.parquet"), recursive=True))
    return files[-1] if files else None

def forward_point(lat_deg, lon_deg, speed_kt, track_deg, dt_s, alt_ft, vert_rate_fpm=None, alt_override_ft=None):
    if any(v is None for v in [lat_deg, lon_deg, speed_kt, track_deg]):
        return None, None, None
    lat = math.radians(float(lat_deg))
    lon = math.radians(float(lon_deg))
    brg = math.radians(float(track_deg))
    d   = float(speed_kt) * KT_TO_MPS * float(dt_s)
    ang = d / R_EARTH

    sin_lat2 = math.sin(lat)*math.cos(ang) + math.cos(lat)*math.sin(ang)*math.cos(brg)
    lat2 = math.asin(max(-1.0, min(1.0, sin_lat2)))
    y = math.sin(brg) * math.sin(ang) * math.cos(lat)
    x = math.cos(ang) - math.sin(lat) * sin_lat2
    lon2 = lon + math.atan2(y, x)
    lon2 = (lon2 + math.pi) % (2*math.pi) - math.pi

    if alt_override_ft is not None:
        alt2_ft = float(alt_override_ft)
    elif (vert_rate_fpm is not None) and (alt_ft is not None):
        alt2_ft = float(alt_ft) + float(vert_rate_fpm) * (float(dt_s) / 60.0)
    else:
        alt2_ft = alt_ft

    return math.degrees(lat2), math.degrees(lon2), alt2_ft

def predicted_tracks_over_time(df: pd.DataFrame, horizon_s=15.0, use_vert_rate=True, kf_lookup=None):
    """
    For each aircraft and each real sample at time t, compute a prediction at t+horizon_s.
    Returns a DataFrame with one row per *prediction*:
      name, t_rx (source time), t_pred (prediction time),
      pred_lat, pred_lon, pred_alt_ft
    If kf_lookup is provided (dict name -> AircraftKF), uses its predict_next().
    """
    if df.empty:
        return pd.DataFrame(columns=["name","t_rx","t_pred","pred_lat","pred_lon","pred_alt_ft"])

    d = df.dropna(subset=["lat","lon"]).copy()
    d["name"] = d["flight"].fillna("").str.strip()
    d.loc[d["name"].eq(""), "name"] = d["hex"].fillna("UNKNOWN")
    d = d.sort_values(["name","t_rx"])

    rows = []
    for name, g in d.groupby("name", sort=False):
        for _, r in g.iterrows():
            # choose kinematics source
            spd_kt   = r.get("speed")
            trackdeg = r.get("track")
            alt_ft   = r.get("altitude")
            alt_override = None

            # If you have a Kalman filter per aircraft, use it for the horizon prediction
            if kf_lookup and name in kf_lookup:
                alt_p, spd_p, trk_p = kf_lookup[name].predict_next(horizon_s)
                if alt_p is not None:   alt_override = alt_p
                if spd_p is not None:   spd_kt = spd_p
                if trk_p is not None:   trackdeg = trk_p

            lat2, lon2, alt2 = forward_point(
                lat_deg=r["lat"], lon_deg=r["lon"],
                speed_kt=spd_kt, track_deg=trackdeg,
                dt_s=horizon_s,
                alt_ft=alt_ft,
                vert_rate_fpm=(r.get("vert_rate") if use_vert_rate else None),
                alt_override_ft=alt_override
            )
            if lat2 is None:
                continue
            rows.append({
                "name": name,
                "t_rx": float(r["t_rx"]),
                "t_pred": float(r["t_rx"]) + float(horizon_s),
                "pred_lat": lat2,
                "pred_lon": lon2,
                "pred_alt_ft": alt2
            })
    pred = pd.DataFrame(rows)
    return pred

def prediction_errors(df_real: pd.DataFrame, df_pred: pd.DataFrame, max_dt_s=3.0):
    # prepare real with time index per aircraft
    real = df_real.copy()
    real["name"] = real["flight"].fillna("").str.strip()
    real.loc[real["name"].eq(""), "name"] = real["hex"].fillna("UNKNOWN")
    real = real[["name","t_rx","lat","lon","altitude"]].sort_values(["name","t_rx"])

    errs = []
    for name, gp in df_pred.groupby("name"):
        gr = real[real["name"] == name]
        if gr.empty: 
            continue
        # for each predicted point, find nearest real time
        t_real = gr["t_rx"].to_numpy()
        for _, p in gp.iterrows():
            idx = np.searchsorted(t_real, p["t_pred"])
            cand = []
            if idx > 0:            cand.append(idx-1)
            if idx < len(t_real):  cand.append(idx)
            # pick closest in time
            best = None; best_dt = None
            for j in cand:
                dt = abs(t_real[j] - p["t_pred"])
                if best is None or dt < best_dt:
                    best, best_dt = j, dt
            if best is None or best_dt > max_dt_s:
                continue
            rr = gr.iloc[best]
            errs.append({
                "name": name,
                "t_pred": p["t_pred"],
                "dt_s": best_dt,
                "alt_err_ft": (p["pred_alt_ft"] - rr["altitude"]),
                "lat_err_deg": (p["pred_lat"] - rr["lat"]),
                "lon_err_deg": (p["pred_lon"] - rr["lon"]),
            })
    return pd.DataFrame(errs)

def latest_predictions(df, horizon_s=15.0, use_vert_rate=True, kf_lookup=None):
    """
    For each aircraft, take its latest row and produce one predicted point horizon_s seconds ahead.
    If kf_lookup is provided (dict name->AircraftKF), it will use KF predictions for (speed, track, alt).
    Returns DataFrame with: name, lat, lon, alt_ft, pred_lat, pred_lon, pred_alt_ft
    """
    if df.empty:
        return pd.DataFrame(columns=["name","lat","lon","alt_ft","pred_lat","pred_lon","pred_alt_ft"])

    d = df.dropna(subset=["lat","lon"]).copy()
    d["name"] = d["flight"].fillna("").str.strip()
    d.loc[d["name"].eq(""), "name"] = d["hex"].fillna("UNKNOWN")

    idx = d.groupby("name", sort=False)["t_rx"].idxmax()
    g   = d.loc[idx, ["name","lat","lon","altitude","speed","track","vert_rate"]].copy()

    rows = []
    for _, r in g.iterrows():
        # default: raw kinematics
        spd_kt   = r.get("speed")
        trackdeg = r.get("track")
        alt_ft   = r.get("altitude")
        alt_override = None

        # if you have a Kalman filter, override the kinematics here:
        if kf_lookup and r["name"] in kf_lookup:
            alt_pred_ft, spd_pred_kt, track_pred_deg = kf_lookup[r["name"]].predict_next(horizon_s)
            if alt_pred_ft is not None:
                alt_override = alt_pred_ft
            if spd_pred_kt is not None:
                spd_kt = spd_pred_kt
            if track_pred_deg is not None:
                trackdeg = track_pred_deg

        lat2, lon2, alt2 = forward_point(
            lat_deg=r["lat"],
            lon_deg=r["lon"],
            speed_kt=spd_kt,
            track_deg=trackdeg,
            dt_s=horizon_s,
            alt_ft=alt_ft,
            vert_rate_fpm=(r.get("vert_rate") if use_vert_rate else None),
            alt_override_ft=alt_override
        )
        if lat2 is None:  # missing inputs
            continue

        rows.append({
            "name": r["name"],
            "lat": r["lat"], "lon": r["lon"], "alt_ft": alt_ft,
            "pred_lat": lat2, "pred_lon": lon2, "pred_alt_ft": alt2
        })

    return pd.DataFrame(rows)

def build_fig(df: pd.DataFrame, horizon_s=15.0):
    # --- Actual tracks ---
    traces = []
    df = df.dropna(subset=["lat","lon","altitude"]).sort_values(["flight","hex","t_rx"])
    df["name"] = df["flight"].fillna("").str.strip()
    df.loc[df["name"].eq(""), "name"] = df["hex"].fillna("UNKNOWN")

    # optional: human-readable time for tooltip
    t = pd.to_datetime(df["t_rx"], unit="s", utc=True).dt.tz_convert("America/Los_Angeles")
    df["t_str"] = t.dt.strftime("%Y-%m-%d %H:%M:%S %Z")

    for name, g in df.groupby("name", sort=False):
        custom = np.column_stack([
            g["name"].astype(str),
            g["altitude"].astype(float),
            g["speed"].astype(float),
            g["t_str"].astype(str),
        ])
        traces.append(go.Scatter3d(
            x=g["lon"], y=g["lat"], z=g["altitude"],
            mode="lines+markers",
            marker=dict(size=2),
            name=str(name),
            line=dict(width=4),
            customdata=custom,
            hovertemplate=(
                "Flight: %{customdata[0]}<br>"
                "Alt: %{customdata[1]} ft<br>"
                "Speed: %{customdata[2]} kts<br>"
                "Time: %{customdata[3]}<extra></extra>"
            )
        ))

    # --- Predicted tracks over time ---
    pred = predicted_tracks_over_time(df, horizon_s=horizon_s, use_vert_rate=True, kf_lookup=None)
    if not pred.empty:
        # order each predicted series by prediction time (t_pred)
        for name, g in pred.groupby("name", sort=False):
            g = g.sort_values("t_pred")
            custom = np.column_stack([
                g["name"].astype(str),
                g["pred_alt_ft"].astype(float),
                g["t_pred"].astype(str),
            ])
            traces.append(go.Scatter3d(
                x=g["pred_lon"], y=g["pred_lat"], z=g["pred_alt_ft"],
                mode="lines+markers",
                marker=dict(size=2),
                name=f"{name} (pred)",
                line=dict(width=4),
                customdata=custom,
                hovertemplate=(
                    "Flight: %{customdata[0]}<br>"
                    "Pred. Alt: %{customdata[1]} ft<br>"
                    "Pred. Time: %{customdata[3]}<extra></extra>"
                )
            ))

    fig = go.Figure(data=traces)
    fig.update_layout(
        scene=dict(
            xaxis_title="Longitude",
            yaxis_title="Latitude",
            zaxis_title="Altitude (ft)",
        ),
        margin=dict(l=0,r=0,t=30,b=0),
        showlegend=True,
        title=f"Actual vs Predicted Tracks (+{int(horizon_s)}s)",
        uirevision="stay"
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
        flights_to_keep = ["N130JM", "N7165G", "N231PN"]
        df = df[df["flight"].isin(flights_to_keep)]
        info = f"Using: {os.path.basename(path)}"
        return build_fig(df), info
    except Exception as e:
        return build_fig(pd.DataFrame()), f"Error reading {path}: {e}"

if __name__ == "__main__":
    # Run on the Pi; we’ll forward this port to the Mac
    app.run(host="0.0.0.0", port=8050, debug=False)
