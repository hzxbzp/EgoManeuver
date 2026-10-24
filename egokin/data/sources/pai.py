"""NVIDIA PhysicalAI-Autonomous-Vehicles (PAI) adapter.

License note: PAI is used for training only. Its license treats benchmarking/performance
results relating to the dataset as NVIDIA confidential, so PAI clips never enter the
reported test sets. Copies must be tracked and deleted 12 months after first download.
"""

import io
import zipfile

import numpy as np
import pandas as pd

from egokin.labels.kinematics import EgoTrack, track_from_poses, yaw_from_quat

REPO = "nvidia/PhysicalAI-Autonomous-Vehicles"

# Countries in PAI that drive on the left (others are right-hand traffic).
LEFT_HAND_TRAFFIC = {"United Kingdom", "Ireland", "Malta", "Cyprus", "Japan", "Australia", "New Zealand"}


def egomotion_offline_path(chunk: int) -> str:
    return f"labels/egomotion.offline/egomotion.offline.chunk_{chunk:04d}.zip"


def camera_path(chunk: int, camera: str = "camera_front_wide_120fov") -> str:
    return f"camera/{camera}/{camera}.chunk_{chunk:04d}.zip"


def read_egomotion_zip(zip_path: str) -> dict[str, pd.DataFrame]:
    """clip_id -> egomotion dataframe (works for both egomotion and egomotion.offline zips)."""
    out = {}
    with zipfile.ZipFile(zip_path) as z:
        for name in z.namelist():
            if name.endswith(".parquet"):
                out[name.split(".")[0]] = pd.read_parquet(io.BytesIO(z.read(name)))
    return out


def track_from_egomotion(df: pd.DataFrame) -> EgoTrack:
    t = df["timestamp"].to_numpy(np.float64) * 1e-6
    yaw = yaw_from_quat(df["qx"].to_numpy(), df["qy"].to_numpy(), df["qz"].to_numpy(), df["qw"].to_numpy())
    return track_from_poses(t, df["x"].to_numpy(), df["y"].to_numpy(), yaw)


def fetch_clip_files(chunk: int, clip_id: str, out_dir: str, camera: str = "camera_front_wide_120fov") -> list[str]:
    """Download one clip's video + timestamps out of a 2 GB chunk zip via HTTP range requests."""
    import os

    from huggingface_hub import get_token, hf_hub_url
    from remotezip import RemoteZip

    url = hf_hub_url(REPO, camera_path(chunk, camera), repo_type="dataset")
    headers = {"Authorization": f"Bearer {get_token()}"}
    os.makedirs(out_dir, exist_ok=True)
    out = []
    with RemoteZip(url, headers=headers) as z:
        for name in z.namelist():
            if name.startswith(clip_id) and (name.endswith(".mp4") or name.endswith(".timestamps.parquet")):
                z.extract(name, out_dir)
                out.append(os.path.join(out_dir, name))
    return out
