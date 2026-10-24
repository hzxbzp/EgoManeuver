"""Label-first mining over PAI: run the (no-map) label factory on every egomotion.offline clip
and write a per-clip index used to decide which videos to download.

Usage: python data/mine_pai.py [--workers 32]
Output: $VEGO_DATA/labels/pai_index_v0.parquet
"""

import argparse
import glob
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from egokin.data.sources.pai import LEFT_HAND_TRAFFIC, read_egomotion_zip, track_from_egomotion
from egokin.labels.factory import label_clip
from egokin.taxonomy import Lat, Long

ROOT = os.environ.get("VEGO_DATA", "./vego_data")

# Rough country centroids (lat, lon) for a solar-elevation estimate; refined later from frame brightness.
COUNTRY_LATLON = {
    "United States": (39.0, -98.0), "Germany": (51.0, 10.0), "France": (46.5, 2.5), "Italy": (42.8, 12.5),
    "Sweden": (60.0, 15.0), "Spain": (40.2, -3.7), "Portugal": (39.6, -8.0), "Greece": (39.0, 22.0),
    "Austria": (47.5, 14.5), "Finland": (62.0, 26.0), "Croatia": (45.1, 15.5), "Netherlands": (52.2, 5.5),
    "Belgium": (50.6, 4.6), "Switzerland": (46.8, 8.2), "Denmark": (56.0, 10.0), "Norway": (61.0, 9.0),
    "Poland": (52.0, 19.0), "Czechia": (49.8, 15.5), "Czech Republic": (49.8, 15.5), "Hungary": (47.2, 19.4),
    "Slovenia": (46.1, 14.8), "Slovakia": (48.7, 19.5), "Romania": (45.9, 24.9), "Bulgaria": (42.7, 25.5),
    "Luxembourg": (49.8, 6.1), "United Kingdom": (54.0, -2.0), "Ireland": (53.2, -8.0), "Canada": (56.0, -96.0),
}


def solar_elevation_deg(lat, month, hour):
    """Approximate solar elevation at local solar time `hour` on the 15th of `month`."""
    doy = (month - 1) * 30.4 + 15
    decl = np.radians(23.44) * np.sin(2 * np.pi * (284 + doy) / 365)
    ha = np.radians(15.0 * (hour + 0.5 - 12.0))
    la = np.radians(lat)
    return float(np.degrees(np.arcsin(np.sin(la) * np.sin(decl) + np.cos(la) * np.cos(decl) * np.cos(ha))))


def mine_chunk(zip_path):
    rows = []
    for cid, df in read_egomotion_zip(zip_path).items():
        try:
            tr = track_from_egomotion(df)
        except Exception:
            continue
        rows.append((cid, tr))
    return zip_path, rows


def summarize(cid, tr, rht):
    lab = label_clip(tr, right_hand_traffic=rht)
    row = {"clip_id": cid, "n_frames": len(tr), "v_mean": float(np.abs(tr.v).mean()), "v_max": float(np.abs(tr.v).max())}
    for c in Long:
        row[f"long_{c.name.lower()}"] = float(np.mean(lab.long == c))
    row["hard_brake_frames"] = int(lab.hard_brake.sum())
    for c in Lat:
        row[f"ev_{c.name.lower()}"] = 0
    for e in lab.events:
        row[f"ev_{Lat(e.kind).name.lower()}"] += 1
    return row


def process(zip_path, meta):
    _, items = mine_chunk(zip_path)
    out = []
    for cid, tr in items:
        country = meta.at[cid, "country"] if cid in meta.index else None
        rht = country not in LEFT_HAND_TRAFFIC
        row = summarize(cid, tr, rht)
        row["chunk"] = int(os.path.basename(zip_path).split("_")[-1].split(".")[0])
        if country is not None:
            month, hour = int(meta.at[cid, "month"]), int(meta.at[cid, "hour_of_day"])
            lat = COUNTRY_LATLON.get(country, (45.0, 0.0))[0]
            elev = solar_elevation_deg(lat, month, hour)
            row.update(country=country, month=month, hour=hour, sun_elev_est=elev,
                       light_est="night" if elev < -6 else ("dusk" if elev < 6 else "day"), rht=rht)
        out.append(row)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=32)
    a = ap.parse_args()
    meta = pd.read_parquet(f"{ROOT}/raw/pai/metadata/data_collection.parquet")
    zips = sorted(glob.glob(f"{ROOT}/raw/pai/labels/egomotion.offline/*.zip"))
    rows = []
    with ProcessPoolExecutor(a.workers) as ex:
        for i, out in enumerate(ex.map(process, zips, [meta] * len(zips), chunksize=4)):
            rows += out
            if (i + 1) % 200 == 0:
                print(f"{i + 1}/{len(zips)} chunks, {len(rows)} clips", flush=True)
    df = pd.DataFrame(rows)
    os.makedirs(f"{ROOT}/labels", exist_ok=True)
    df.to_parquet(f"{ROOT}/labels/pai_index_v0.parquet")
    print(f"wrote {len(df)} clips from {len(zips)} chunks")


if __name__ == "__main__":
    main()
