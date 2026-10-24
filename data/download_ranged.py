"""Resumable parallel HTTP range download (re-resolves redirects on every retry).

Usage: python data/download_ranged.py URL OUT_FILE [--parts 16]
"""

import argparse
import os
import sys
import threading
import time

import requests

CHUNK = 8 << 20
UA = {"User-Agent": "curl/7.81.0"}  # some portals reject the default python-requests agent


def total_size(url: str) -> int:
    r = requests.get(url, headers={**UA, "Range": "bytes=0-0"}, allow_redirects=True, timeout=60)
    r.raise_for_status()
    return int(r.headers["Content-Range"].split("/")[-1])


def fetch_part(url, path, start, end, progress, idx):
    """Download bytes [start, end] into `path`, resuming from its current size."""
    for attempt in range(1000):
        have = os.path.getsize(path) if os.path.exists(path) else 0
        progress[idx] = have
        if start + have > end:
            return
        try:
            with requests.get(url, headers={**UA, "Range": f"bytes={start + have}-{end}"}, stream=True,
                              allow_redirects=True, timeout=120) as r:
                if r.status_code != 206:
                    raise IOError(f"HTTP {r.status_code}")
                with open(path, "ab") as f:
                    for buf in r.iter_content(CHUNK):
                        f.write(buf)
                        progress[idx] += len(buf)
        except Exception as e:  # network hiccup or expired redirect: retry with a fresh redirect
            print(f"part {idx} attempt {attempt}: {e}", file=sys.stderr, flush=True)
            time.sleep(min(60, 2 + attempt * 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("out")
    ap.add_argument("--parts", type=int, default=16)
    a = ap.parse_args()

    size = total_size(a.url)
    step = -(-size // a.parts)
    ranges = [(i * step, min(size, (i + 1) * step) - 1) for i in range(a.parts)]
    paths = [f"{a.out}.part{i:02d}" for i in range(a.parts)]
    progress = [0] * a.parts
    threads = [threading.Thread(target=fetch_part, args=(a.url, p, s, e, progress, i), daemon=True)
               for i, (p, (s, e)) in enumerate(zip(paths, ranges))]
    for t in threads:
        t.start()
    t0, last = time.time(), 0
    while any(t.is_alive() for t in threads):
        time.sleep(30)
        done = sum(progress)
        print(f"{done / 1e9:7.1f} / {size / 1e9:.1f} GB  {(done - last) / 30e6:6.1f} MB/s  "
              f"elapsed {(time.time() - t0) / 60:5.1f} min", flush=True)
        last = done
    for p, (s, e) in zip(paths, ranges):
        assert os.path.getsize(p) == e - s + 1, f"incomplete {p}"
    with open(a.out + ".tmp", "wb") as out:
        for p in paths:
            with open(p, "rb") as f:
                while buf := f.read(CHUNK * 8):
                    out.write(buf)
    assert os.path.getsize(a.out + ".tmp") == size
    os.replace(a.out + ".tmp", a.out)
    for p in paths:
        os.remove(p)
    print("complete", a.out, size, flush=True)


if __name__ == "__main__":
    main()
