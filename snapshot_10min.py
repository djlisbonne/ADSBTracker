#!/usr/bin/env python3
import os, time, signal
from datetime import datetime, timezone
import requests, pandas as pd

# --- config ---
URL = "http://127.0.0.1:8080/data.json"  # MalcolmRobb JSON (LIST)
OUT_ROOT = "./snapshots"                 # where 10-min files go
DURATION_SEC = 10 * 60                   # 10 minutes
POLL_HZ = 1.0                            # 1 sample/sec is plenty
COMPRESSION = "snappy"                   # parquet compression
# -------------

UTC = timezone.utc

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def start_window():
    t0 = time.time()
    start_dt = datetime.fromtimestamp(t0, tz=UTC).replace(second=0, microsecond=0)
    # floor to nearest 10-minute boundary
    floored_minute = (start_dt.minute // 10) * 10
    start_dt = start_dt.replace(minute=floored_minute)
    return t0, start_dt

def out_path_for(start_dt):
    y, m, d, H, M = start_dt.year, start_dt.month, start_dt.day, start_dt.hour, start_dt.minute
    dirpath = os.path.join(OUT_ROOT, f"{y:04d}", f"{m:02d}", f"{d:02d}", f"{H:02d}")
    ensure_dir(dirpath)
    return os.path.join(dirpath, f"adsb_mr_{start_dt.strftime('%Y-%m-%dT%H-%M-%SZ')}_10min.parquet")

def fetch_list():
    # MalcolmRobb returns a LIST of aircraft dicts
    return requests.get(URL, timeout=2).json()

def main():
    rows = []
    t0, start_dt = start_window()
    end_time = t0 + DURATION_SEC
    period = 1.0 / POLL_HZ

    stopping = False
    def _stop(*_):
        nonlocal stopping
        stopping = True
    signal.signal(signal.SIGINT, _stop)

    while not stopping and time.time() < end_time:
        loop_t = time.time()
        try:
            data = fetch_list()
            t_rx = time.time()
            if isinstance(data, list):
                for ac in data:
                    lat, lon = ac.get("lat"), ac.get("lon")
                    if lat is None or lon is None or ac.get("speed") > 150 or ac.get("altitude") > 5000:
                        continue
                    rows.append({
                        "t_rx": t_rx,
                        "hex": ac.get("hex"),
                        "flight": (ac.get("flight") or "").strip() or None,
                        "lat": lat, "lon": lon,
                        "altitude": ac.get("altitude"),
                        "speed": ac.get("speed"),
                        "track": ac.get("track"),
                        "vert_rate": ac.get("vert_rate"),
                        "rssi": ac.get("rssi"),
                        "seen": ac.get("seen"),
                        "seen_pos": ac.get("seen_pos"),
                        "messages": ac.get("messages"),
                        "category": ac.get("category"),
                    })
        except Exception as e:
            print("[poll error]", e)

        # simple pacing
        dt = time.time() - loop_t
        if dt < period:
            time.sleep(period - dt)

    # Write one parquet file for this 10-minute window
    if rows:
        df = pd.DataFrame(rows)
        out = out_path_for(start_dt)
        tmp = out + ".tmp"
        df.to_parquet(tmp, compression=COMPRESSION, index=False)
        os.replace(tmp, out)
        print(f"[WROTE] {out} rows={len(df)}")
    else:
        print("[INFO] No rows collected; nothing written.")

if __name__ == "__main__":
    main()
