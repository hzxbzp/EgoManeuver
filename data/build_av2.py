"""Build unified AV2 clips: fetch ring_front_center images, crop the portrait frame to a 16:9 band
around the horizon, transcode, label with the current map labeler, then delete the raw images.

Usage: python data/build_av2.py [--workers 6]
"""

import argparse
import glob
import json
import os
import shutil
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np

from egokin.data.clipstore import ROOT, clip_dir, light_from_luma, save_labels, sky_luma
from egokin.data.sources.av2 import city_name, download_camera, lead_queue_mask, load_lanegraph, load_track
from egokin.data.transcode import transcode_images
from egokin.labels.factory import label_clip


def work(log_dir):
    split, log_id = log_dir.rstrip("/").split("/")[-2:]
    if os.path.exists(os.path.join(clip_dir("av2", log_id), "meta.json")):
        return log_id, "skip"
    try:
        imgs = sorted(download_camera(split, log_id, f"{ROOT}/raw/av2", workers=8))
        tr, t0 = load_track(log_dir)
        g, xw = load_lanegraph(log_dir)
        lab = label_clip(tr, True, g, xw, lead_queue=lead_queue_mask(log_dir, tr, t0))
        ft = np.array([int(os.path.basename(p)[:-4]) * 1e-9 - t0 for p in imgs])
        h, w = cv2.imread(imgs[0]).shape[:2]
        band = int(w * 9 / 16)
        y0 = int(h * 0.30)
        os.makedirs(clip_dir("av2", log_id), exist_ok=True)
        out = os.path.join(clip_dir("av2", log_id), "video.mp4")
        n = transcode_images(imgs, ft, float(min(ft[-1], tr.t[-1])), out, crop_box=(0, y0, w, y0 + band))
        luma = sky_luma(out)
        meta = dict(dataset="av2", clip_id=log_id, split=split, city=city_name(log_dir), rht=True,
                    camera="ring_front_center_crop16x9", sky_luma=round(luma, 1), light=light_from_luma(luma))
        save_labels("av2", log_id, tr, lab, meta, n)
        return log_id, "ok"
    except Exception as e:
        return log_id, f"error {type(e).__name__}: {e}"
    finally:
        shutil.rmtree(os.path.join(log_dir, "sensors"), ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    logs = sorted(glob.glob(f"{ROOT}/raw/av2/*/*/"))
    stats = {}
    with ProcessPoolExecutor(a.workers) as ex:
        for i, (log_id, status) in enumerate(ex.map(work, logs)):
            key = status.split(" ")[0]
            stats[key] = stats.get(key, 0) + 1
            if key == "error":
                print(log_id, status, flush=True)
            if (i + 1) % 50 == 0:
                print(f"{i + 1}/{len(logs)} {stats}", flush=True)
    print("done", stats, flush=True)


if __name__ == "__main__":
    main()
