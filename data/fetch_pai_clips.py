"""Select PAI clips from the label-first index, fetch only their front-wide videos, and write
unified clips (video + labels + meta). Raw videos are deleted right after transcoding.

Usage: python data/fetch_pai_clips.py --budget 20000 [--workers 12]
Selection (batch v0): every clip with a rare event (U-turn, reverse, hard brake, pull over/out or
borrow candidates), a large share of metadata-night clips, and a country-stratified remainder.
"""

import argparse
import glob
import os
import shutil
import tempfile
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from egokin.data.clipstore import clip_dir, light_from_luma, save_labels, sky_luma
from egokin.data.sources.pai import fetch_clip_files, read_egomotion_zip, track_from_egomotion
from egokin.data.transcode import transcode_video
from egokin.labels.factory import label_clip

ROOT = os.environ.get("VEGO_DATA", "./vego_data")


def select(df: pd.DataFrame, budget: int, seed: int = 0) -> pd.DataFrame:
    """All rare-class clips (U-turn, reverse, pull over/out candidates); capped samples of borrow
    candidates, hard-brake clips and metadata-night clips; country-stratified remainder."""
    rng = np.random.default_rng(seed)

    def sample(mask, k):
        idx = df.index[mask & ~df.index.isin(list(pick))]
        return set(rng.choice(idx, size=min(len(idx), k), replace=False)) if len(idx) else set()

    pick = set(df.index[(df.ev_u_turn > 0) | (df.long_reverse > 0) | (df.ev_pull_over > 0) | (df.ev_pull_out > 0)])
    pick |= sample((df.ev_borrow_left > 0) | (df.ev_borrow_right > 0), 3000)
    pick |= sample(df.hard_brake_frames > 0, 3000)
    pick |= sample(df.light_est == "night", 6000)
    rest = df[~df.index.isin(list(pick))]
    left = max(0, budget - len(pick))
    if left:
        # country-stratified: equalize toward the long tail of countries
        w = 1.0 / rest.country.map(rest.country.value_counts()).to_numpy() ** 0.5
        pick |= set(rng.choice(rest.index, size=min(left, len(rest)), replace=False, p=w / w.sum()))
    return df.loc[sorted(pick)]


def process_chunk(args):
    chunk, rows = args
    ego = read_egomotion_zip(f"{ROOT}/raw/pai/labels/egomotion.offline/egomotion.offline.chunk_{chunk:04d}.zip")
    done = 0
    tmp = tempfile.mkdtemp(dir=f"{ROOT}/cache")
    try:
        for r in rows:
            cid = r["clip_id"]
            if os.path.exists(os.path.join(clip_dir("pai", cid), "meta.json")) or cid not in ego:
                continue
            try:
                files = fetch_clip_files(chunk, cid, tmp)
                mp4 = next(f for f in files if f.endswith(".mp4"))
                ts = pd.read_parquet(next(f for f in files if f.endswith(".parquet")))["timestamp"].to_numpy() * 1e-6
                tr = track_from_egomotion(ego[cid])
                lab = label_clip(tr, right_hand_traffic=bool(r.get("rht", True)))
                os.makedirs(clip_dir("pai", cid), exist_ok=True)
                out = os.path.join(clip_dir("pai", cid), "video.mp4")
                n = transcode_video(mp4, ts, float(tr.t[-1]), out)
                luma = sky_luma(out)
                meta = dict(dataset="pai", clip_id=cid, chunk=chunk, camera="camera_front_wide_120fov",
                            camera_model="ftheta_120deg", country=r.get("country"), rht=bool(r.get("rht", True)),
                            month=r.get("month"), hour=r.get("hour"), light_prior=r.get("light_est"),
                            sky_luma=round(luma, 1), light=light_from_luma(luma), split="train_only_license")
                save_labels("pai", cid, tr, lab, meta, n)
                done += 1
            except Exception as e:  # keep going; a clip can be retried on the next run
                print(f"chunk {chunk} clip {cid}: {type(e).__name__}: {e}", flush=True)
            finally:
                for f in glob.glob(os.path.join(tmp, cid + "*")):
                    os.remove(f)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return chunk, done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=20000)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--max-chunks", type=int, default=0, help="smoke test: only the first N chunks")
    a = ap.parse_args()
    df = pd.read_parquet(f"{ROOT}/labels/pai_index_v0.parquet").set_index("clip_id", drop=False)
    sel = select(df, a.budget)
    sel.to_parquet(f"{ROOT}/labels/pai_selection_v0.parquet")
    print(f"selected {len(sel)} clips, light prior {sel.light_est.value_counts().to_dict()}", flush=True)
    by_chunk = defaultdict(list)
    for _, r in sel.iterrows():
        by_chunk[int(r.chunk)].append(r.to_dict())
    total = 0
    with ProcessPoolExecutor(a.workers) as ex:
        jobs = sorted(by_chunk.items())[: a.max_chunks or None]
        for i, (chunk, n) in enumerate(ex.map(process_chunk, jobs)):
            total += n
            if (i + 1) % 50 == 0:
                print(f"{i + 1}/{len(by_chunk)} chunks, {total} clips written", flush=True)
    print(f"done: {total} clips", flush=True)


if __name__ == "__main__":
    main()
