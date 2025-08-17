#!/usr/bin/env python3
import time, requests

PI = "127.0.0.1"  # set to your Pi's IP if you run this from your Mac
URL = f"http://{PI}:8080/data.json"  # MalcolmRobb: returns a LIST

def main():
    while True:
        try:
            data = requests.get(URL, timeout=2).json()
            if not isinstance(data, list):
                print("Unexpected JSON shape (expected list).")
                time.sleep(0.5)
                continue

            for ac in data:
                ident = ac.get("flight") or ac.get("hex")
                lat   = ac.get("lat");  lon = ac.get("lon")
                alt   = ac.get("altitude")
                spd   = ac.get("speed"); trk = ac.get("track")
                vr    = ac.get("vert_rate")
                if lat is not None and lon is not None:
                    print(f"{ident}: lat={lat} lon={lon} alt_ft={alt} "
                          f"gs_kt={spd} track={trk} vr_fpm={vr}")
        except Exception as e:
            print("poll error:", e)
        time.sleep(0.5)

if __name__ == "__main__":
    main()
