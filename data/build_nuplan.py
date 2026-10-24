"""Build nuPlan clips: stream a camera TAR archive keeping only CAM_F0, then per log segment fetch its DB,
label with the map (incl. road-relative state), transcode to 10 fps, and delete the images.

Usage: python data/build_nuplan.py --split train --archives 0,1,2,3 [--workers 8]
"""

import argparse
import collections
import glob
import os
import shutil
import subprocess
from concurrent.futures import ProcessPoolExecutor

from egokin.data.clipstore import ROOT, clip_dir, light_from_luma, save_labels, sky_luma
from egokin.data.sources import nuplan as npl
from egokin.data.transcode import transcode_images
from egokin.labels.factory import label_clip

TMP = f"{ROOT}/cache/nuplan_tmp"
SPLIT_MAP = {"train": "train", "val": "val", "test": "test"}


def build_log(args):
    log_dir, split, index = args
    log = os.path.basename(log_dir.rstrip("/"))
    if os.path.exists(os.path.join(clip_dir("nuplan", log), "meta.json")):
        return log, "skip"
    if log not in index:
        return log, "no_db"
    try:
        loc, epsg, ego, img, tags = npl.read_log(npl.fetch_db(log, index))
        tr, ft, t0 = npl.track_and_frames(ego, img)
        root = os.path.dirname(log_dir.rstrip("/"))
        paths = [os.path.join(root, f) for f in img.filename_jpg]
        ok = [os.path.exists(p) for p in paths]
        if sum(ok) < 0.95 * len(paths):
            return log, f"images_missing({sum(ok)}/{len(paths)})"
        g, xw, carparks = npl.lanegraph(loc, epsg)
        lab = label_clip(tr, loc not in npl.LEFT_HAND, g, xw)
        os.makedirs(clip_dir("nuplan", log), exist_ok=True)
        out = os.path.join(clip_dir("nuplan", log), "video.mp4")
        keep = [i for i, o in enumerate(ok) if o]
        n = transcode_images([paths[i] for i in keep], ft[keep], float(min(ft[-1], tr.t[-1])), out)
        luma = sky_luma(out)
        tag_counts = collections.Counter(tags.type)
        meta = dict(dataset="nuplan", clip_id=log, split=SPLIT_MAP[split], split_final=SPLIT_MAP[split], location=loc,
                    epsg=epsg, rht=loc not in npl.LEFT_HAND, sky_luma=round(luma, 1), light=light_from_luma(luma),
                    scenario_tags={k: int(v) for k, v in tag_counts.most_common()})
        save_labels("nuplan", log, tr, lab, meta, n)
        return log, "ok"
    except Exception as e:
        return log, f"error {type(e).__name__}: {e}"


def stream_archive(split, k):
    dst = os.path.join(TMP, f"{split}_{k}")
    os.makedirs(dst, exist_ok=True)
    done = os.path.join(dst, ".done")
    if not os.path.exists(done):
        cmd = f"curl -s -L --retry 5 '{npl.camera_archive_url(split, k)}' | tar -x -C '{dst}' --wildcards '*/CAM_F0/*'"
        subprocess.run(cmd, shell=True, check=True)
        open(done, "w").close()
    return dst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    ap.add_argument("--archives", default="0")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    index = npl.db_index()
    print(f"db index: {len(index)} log segments", flush=True)
    for k in [int(x) for x in a.archives.split(",")]:
        dst = stream_archive(a.split, k)
        logs = sorted(glob.glob(f"{dst}/*/*/"))
        print(f"archive {a.split}_{k}: {len(logs)} log segments with CAM_F0", flush=True)
        stats = collections.Counter()
        with ProcessPoolExecutor(a.workers) as ex:
            for log, status in ex.map(build_log, [(d, a.split, index) for d in logs]):
                stats[status.split("(")[0].split(" ")[0]] += 1
                if status.startswith("error"):
                    print(log, status, flush=True)
        print(f"archive {a.split}_{k} done: {dict(stats)}", flush=True)
        shutil.rmtree(dst, ignore_errors=True)


if __name__ == "__main__":
    main()
