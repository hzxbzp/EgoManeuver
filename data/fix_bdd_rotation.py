"""Re-transcode BDD clips with the display rotation applied.

The BDD100K .mov files store 720x1280 frames with a 90-degree display matrix; the first build ignored it, so
clip videos were sideways and squashed ~3x. Labels (GPS/IMU + BDD-X) do not depend on the video and are kept;
the new video keeps the old frame count and time base (frame k = clip time k / 10 s).

Usage: python data/fix_bdd_rotation.py [--workers 8]
"""

import argparse
import json
import os
import shutil
import tempfile
from concurrent.futures import ProcessPoolExecutor

import av
import numpy as np

from egokin.data.clipstore import ROOT, clip_dir
from egokin.data.sources import bdd
from egokin.data.transcode import transcode_video

VERSION = "v1_rotation"
_RZ = None


def fetch(split, cid, out_dir):
    """bdd.fetch_video with one RemoteZip handle per worker (reading the 100k-entry directory is the slow part)."""
    global _RZ
    from remotezip import RemoteZip

    member = f"bdd100k/videos/100k/{split}/{cid}.mov"
    for attempt in range(3):
        try:
            _RZ = _RZ or RemoteZip(f"{bdd.MIRROR}/bdd100k_videos.zip")
            _RZ.extract(member, out_dir)
            return os.path.join(out_dir, member)
        except Exception:
            _RZ = None
            if attempt == 2:
                raise


def work(cid):
    d = clip_dir("bdd", cid)
    mp = os.path.join(d, "meta.json")
    meta = json.load(open(mp))
    if meta.get("video_version") == VERSION:
        return cid, "skip"
    tmp = tempfile.mkdtemp(dir=f"{ROOT}/cache")
    try:
        mov = fetch(meta["split"], cid, tmp)
        with av.open(mov) as c:
            s = c.streams.video[0]
            fps = float(s.average_rate)
            nfr = s.frames or int(float(c.duration / 1e6) * fps)
            rot = int(getattr(next(c.decode(video=0)), "rotation", 0))
        with av.open(os.path.join(d, "video.mp4")) as c:
            n_old = c.streams.video[0].frames or sum(1 for _ in c.decode(video=0))
        status = "same"
        if rot % 360:
            new = os.path.join(d, "video.rot.mp4")
            n = transcode_video(mov, np.arange(nfr) / fps, (n_old - 1) / 10.0, new)
            if n != n_old:
                os.remove(new)
                return cid, f"error frame count {n} != {n_old}"
            os.replace(new, os.path.join(d, "video.mp4"))
            status = "fixed"
        meta.update(video_version=VERSION, src_rotation=rot, src_size=[s.width, s.height])
        json.dump(meta, open(mp + ".tmp", "w"), indent=1, default=str)
        os.replace(mp + ".tmp", mp)
        return cid, f"{status} rot={rot}"
    except Exception as e:
        return cid, f"error {type(e).__name__}: {e}"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    ids = sorted(c for c in os.listdir(f"{ROOT}/clips/bdd") if os.path.exists(os.path.join(clip_dir("bdd", c), "meta.json")))
    stats = {}
    with ProcessPoolExecutor(a.workers) as ex:
        for i, (cid, st) in enumerate(ex.map(work, ids)):
            stats[st] = stats.get(st, 0) + 1
            if st.startswith("error"):
                print(cid, st, flush=True)
            if (i + 1) % 200 == 0:
                print(i + 1, dict(sorted(stats.items(), key=lambda kv: -kv[1])[:6]), flush=True)
    print("done", stats, flush=True)


if __name__ == "__main__":
    main()
