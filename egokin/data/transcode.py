"""Unified clip videos: 10 fps, 512x288 H.264, frame k shown at clip time k / fps.

Frames are picked by timestamp (nearest source frame to each grid time), so video frame k is
aligned with kinematics/label frame k produced at the same rate from the same clip origin.
"""

import av
import numpy as np

OUT_FPS = 10
OUT_SIZE = (512, 288)  # (w, h), 16:9; training crops/zooms from this


def grid_indices(frame_times_s: np.ndarray, duration_s: float, fps: float = OUT_FPS) -> np.ndarray:
    """Index of the nearest source frame for each output grid time 0, 1/fps, ... <= duration."""
    grid = np.arange(0.0, duration_s + 1e-6, 1.0 / fps)
    idx = np.searchsorted(frame_times_s, grid).clip(1, len(frame_times_s) - 1)
    left_closer = (grid - frame_times_s[idx - 1]) < (frame_times_s[idx] - grid)
    return np.where(left_closer, idx - 1, idx)


def _crop_resize(img: np.ndarray, crop_box, size):
    import cv2

    if crop_box is not None:
        x0, y0, x1, y1 = crop_box
        img = img[y0:y1, x0:x1]
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA)


def _encoder(dst: str, fps: int, size, crf: int):
    out = av.open(dst, "w")
    st = out.add_stream("libx264", rate=fps)
    st.width, st.height = size
    st.pix_fmt = "yuv420p"
    st.options = {"crf": str(crf), "preset": "veryfast", "g": str(fps * 2)}
    return out, st


def _write(out, st, rgb_frames):
    for img in rgb_frames:
        for pkt in st.encode(av.VideoFrame.from_ndarray(np.ascontiguousarray(img), format="rgb24")):
            out.mux(pkt)
    for pkt in st.encode():
        out.mux(pkt)
    out.close()


def transcode_video(src: str, frame_times_s: np.ndarray, duration_s: float, dst: str,
                    crop_box=None, size=OUT_SIZE, fps=OUT_FPS, crf=23) -> int:
    """Video file -> unified clip. frame_times_s: source frame times in clip time (s)."""
    pick = grid_indices(np.asarray(frame_times_s, float), duration_s, fps)
    wanted = {}
    for k, src_idx in enumerate(pick):
        wanted.setdefault(int(src_idx), []).append(k)
    frames = [None] * len(pick)
    with av.open(src) as c:
        c.streams.video[0].thread_type = "AUTO"
        for i, fr in enumerate(c.decode(video=0)):
            if i in wanted:
                img = fr.to_ndarray(format="rgb24")
                k = int(round(getattr(fr, "rotation", 0) / 90.0)) % 4  # display matrix (CCW degrees), e.g. BDD100K phones
                if k:
                    img = np.ascontiguousarray(np.rot90(img, k))
                img = _crop_resize(img, crop_box, size)
                for k in wanted[i]:
                    frames[k] = img
            if i >= pick[-1]:
                break
    last = next(f for f in frames if f is not None)
    frames = [last := (f if f is not None else last) for f in frames]  # pad if the video ends early
    out, st = _encoder(dst, fps, size, crf)
    _write(out, st, frames)
    return len(frames)


def transcode_images(paths: list[str], frame_times_s: np.ndarray, duration_s: float, dst: str,
                     crop_box=None, size=OUT_SIZE, fps=OUT_FPS, crf=23) -> int:
    """Image sequence -> unified clip."""
    import cv2

    pick = grid_indices(np.asarray(frame_times_s, float), duration_s, fps)
    cache = {}

    def load(i):
        if i not in cache:
            cache.clear()
            cache[i] = _crop_resize(cv2.cvtColor(cv2.imread(paths[i]), cv2.COLOR_BGR2RGB), crop_box, size)
        return cache[i]

    out, st = _encoder(dst, fps, size, crf)
    _write(out, st, (load(int(i)) for i in pick))
    return len(pick)
