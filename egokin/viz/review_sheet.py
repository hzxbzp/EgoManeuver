"""Review sheet: key frames + kinematics/label plots for verifying auto-labels by eye."""

import av
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from egokin.labels.factory import ClipLabels
from egokin.labels.kinematics import EgoTrack
from egokin.taxonomy import IGNORE, Lat, Long

LAT_COLORS = {
    Lat.STRAIGHT: "#d9d9d9", Lat.CURVE: "#9ecae1", Lat.TURN_LEFT: "#3182bd", Lat.TURN_RIGHT: "#e6550d",
    Lat.U_TURN: "#756bb1", Lat.LANE_CHANGE_LEFT: "#31a354", Lat.LANE_CHANGE_RIGHT: "#a1d99b",
    Lat.NUDGE_LEFT: "#fd8d3c", Lat.NUDGE_RIGHT: "#fdd0a2", Lat.PULL_OVER: "#de2d26", Lat.PULL_OUT: "#fc9272",
}
LONG_COLORS = {
    Long.STATIONARY: "#252525", Long.START: "#41ab5d", Long.ACCELERATE: "#a1d99b", Long.CRUISE: "#d9d9d9",
    Long.DECELERATE: "#fdae6b", Long.DECEL_TO_STOP: "#e6550d", Long.REVERSE: "#756bb1",
}


def read_frames(video_path, times_s: np.ndarray, frame_times_s: np.ndarray, width: int = 384) -> list:
    """Frames closest to `times_s` (clip time, s). `video_path` is a video file or a list of image files."""
    want = {int(np.argmin(np.abs(frame_times_s - t))): i for i, t in enumerate(times_s)}
    out = [None] * len(times_s)
    if isinstance(video_path, (list, tuple)):
        import cv2

        for k, i in want.items():
            img = cv2.cvtColor(cv2.imread(video_path[k]), cv2.COLOR_BGR2RGB)
            out[i] = cv2.resize(img, (width, int(img.shape[0] * width / img.shape[1])))
        return out
    with av.open(video_path) as c:
        for k, frame in enumerate(c.decode(video=0)):
            if k in want:
                h = int(frame.height * width / frame.width)
                out[want[k]] = frame.reformat(width=width, height=h).to_ndarray(format="rgb24")
            if k >= max(want):
                break
    return out


def _band(ax, labels, t, colors, y0, y1):
    for lab, col in colors.items():
        m = labels == int(lab)
        if m.any():
            ax.fill_between(t, y0, y1, where=m, color=col, step="mid", linewidth=0)


def render(video_path, frame_times_s, tr: EgoTrack, lab: ClipLabels, out_png, title="",
           window=None, n_frames=6, v_lat=None):
    t = tr.t
    t0, t1 = (t[0], t[-1]) if window is None else (max(t[0], window[0] - 1.0), min(t[-1], window[1] + 1.0))
    times = np.linspace(t0, t1, n_frames)
    frames = read_frames(video_path, times, frame_times_s)

    fig = plt.figure(figsize=(3.2 * n_frames, 9.0))
    gs = fig.add_gridspec(3, n_frames, height_ratios=[1.0, 1.0, 1.1])
    for i, (img, ts) in enumerate(zip(frames, times)):
        ax = fig.add_subplot(gs[0, i])
        if img is not None:
            ax.imshow(img)
        k = int(np.argmin(np.abs(t - ts)))
        lat_name = Lat(lab.lat[k]).name if lab.lat[k] != IGNORE else "-"
        ax.set_title(f"t={ts:.1f}s  {Long(lab.long[k]).name}\n{lat_name}", fontsize=8)
        ax.axis("off")

    ax = fig.add_subplot(gs[1, : n_frames // 2])
    ax.plot(t, tr.v, "k-", lw=1.2, label="v [m/s]")
    ax.plot(t, tr.a * 3, "b-", lw=0.8, alpha=0.6, label="a x3 [m/s2]")
    ylo, yhi = min(tr.v.min(), (tr.a * 3).min()) - 1, max(tr.v.max(), (tr.a * 3).max()) + 1
    _band(ax, lab.long, t, LONG_COLORS, ylo, ylo + 0.12 * (yhi - ylo))
    ax.set_ylim(ylo, yhi); ax.axvspan(t0, t1, color="y", alpha=0.08); ax.legend(fontsize=7, loc="upper right")
    ax.set_title("longitudinal (band = A label)", fontsize=9)

    ax = fig.add_subplot(gs[1, n_frames // 2:])
    ax.plot(t, np.degrees(tr.yaw_rate), "k-", lw=1.0, label="yaw rate [deg/s]")
    if v_lat is not None:
        ax.plot(t, v_lat * 10, "g-", lw=1.0, label="v_lat x10 [m/s]")
        ax.plot(t, np.cumsum(v_lat) / tr.hz, "m-", lw=0.8, alpha=0.7, label="lat disp [m]")
    lo, hi = ax.get_ylim()
    _band(ax, lab.lat, t, LAT_COLORS, lo, lo + 0.12 * (hi - lo))
    ax.axvspan(t0, t1, color="y", alpha=0.08); ax.legend(fontsize=7, loc="upper right")
    ax.set_title("lateral (band = B label)", fontsize=9)

    ax = fig.add_subplot(gs[2, : n_frames // 2])
    ax.scatter(tr.x, tr.y, c=lab.lat, s=3, cmap="tab20", vmin=-1, vmax=19)
    w = (t >= t0) & (t <= t1)
    ax.plot(tr.x[w], tr.y[w], "r-", lw=2, alpha=0.5)
    ax.plot(tr.x[0], tr.y[0], "go")
    ax.set_aspect("equal", adjustable="datalim"); ax.set_title("BEV trajectory (start = green)", fontsize=9)

    ax = fig.add_subplot(gs[2, n_frames // 2:])
    ax.axis("off")
    txt = "\n".join(f"{Lat(e.kind).name:18s} {e.start / lab.hz:5.1f}-{e.end / lab.hz:5.1f}s disp={e.disp:+.2f}m conf={e.conf:.1f} {e.note}"
                    for e in lab.events)
    ax.text(0, 1, txt or "(no lateral events)", va="top", family="monospace", fontsize=8)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_png, dpi=80)
    plt.close(fig)
