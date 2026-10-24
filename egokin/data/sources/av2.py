"""Argoverse 2 Sensor dataset adapter (public S3 bucket, CC BY-NC-SA 4.0)."""

import os
import re
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import requests

from egokin.labels.kinematics import EgoTrack, track_from_poses, yaw_from_quat

BUCKET = "https://argoverse.s3.us-east-1.amazonaws.com"
PREFIX = "datasets/av2/sensor"
_NS = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
META_FILES = ("city_SE3_egovehicle.feather", "annotations.feather",
              "calibration/egovehicle_SE3_sensor.feather", "calibration/intrinsics.feather")


def _list(prefix: str, delimiter: str | None = None):
    """Yield (keys, common_prefixes) pages of an S3 listing."""
    token = None
    while True:
        params = {"list-type": "2", "prefix": prefix, "max-keys": "1000"}
        if delimiter:
            params["delimiter"] = delimiter
        if token:
            params["continuation-token"] = token
        r = requests.get(BUCKET, params=params, timeout=60)
        r.raise_for_status()
        root = ET.fromstring(r.content)
        keys = [(c.find("s3:Key", _NS).text, int(c.find("s3:Size", _NS).text)) for c in root.findall("s3:Contents", _NS)]
        prefixes = [p.find("s3:Prefix", _NS).text for p in root.findall("s3:CommonPrefixes", _NS)]
        yield keys, prefixes
        if root.find("s3:IsTruncated", _NS).text != "true":
            break
        token = root.find("s3:NextContinuationToken", _NS).text


def list_logs(split: str) -> list[str]:
    out = []
    for _, prefixes in _list(f"{PREFIX}/{split}/", delimiter="/"):
        out += [p.rstrip("/").split("/")[-1] for p in prefixes]
    return out


def list_files(split: str, log_id: str, sub: str = "") -> list[tuple[str, int]]:
    out = []
    for keys, _ in _list(f"{PREFIX}/{split}/{log_id}/{sub}"):
        out += keys
    return out


def _get(key: str, dst: str, session: requests.Session | None = None) -> str:
    if os.path.exists(dst):
        return dst
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    r = (session or requests).get(f"{BUCKET}/{key}", timeout=120)
    r.raise_for_status()
    tmp = dst + ".part"
    with open(tmp, "wb") as f:
        f.write(r.content)
    os.replace(tmp, dst)
    return dst


def download_log_meta(split: str, log_id: str, root: str) -> str:
    """Poses, vector map, calibration and 3D box tracks for one log (~0.6 MB)."""
    base = f"{PREFIX}/{split}/{log_id}"
    dst = os.path.join(root, split, log_id)
    for f in META_FILES:
        try:
            _get(f"{base}/{f}", os.path.join(dst, f))
        except requests.HTTPError:
            pass  # e.g. annotations are withheld for the test split
    for key, _ in list_files(split, log_id, "map/"):
        if "log_map_archive" in key:
            _get(key, os.path.join(dst, "map", key.split("/")[-1]))
    return dst


def download_camera(split: str, log_id: str, root: str, camera: str = "ring_front_center", workers: int = 16) -> list[str]:
    files = list_files(split, log_id, f"sensors/cameras/{camera}/")
    dst = os.path.join(root, split, log_id, "sensors", "cameras", camera)
    with requests.Session() as s, ThreadPoolExecutor(workers) as ex:
        return list(ex.map(lambda k: _get(k[0], os.path.join(dst, k[0].split("/")[-1]), s), files))


def city_name(log_dir: str) -> str:
    m = next(f for f in os.listdir(os.path.join(log_dir, "map")) if f.startswith("log_map_archive"))
    return re.search(r"____([A-Z]+)_city", m).group(1)


def load_lanegraph(log_dir: str):
    """Vector map of one log as a LaneGraph (plus pedestrian crossing polygons)."""
    from pathlib import Path

    from av2.map.lane_segment import LaneType
    from av2.map.map_api import ArgoverseStaticMap
    from shapely.geometry import Polygon

    from egokin.labels.lanegraph import Lane, LaneGraph

    mp = next(f for f in os.listdir(os.path.join(log_dir, "map")) if f.startswith("log_map_archive"))
    avm = ArgoverseStaticMap.from_json(Path(os.path.join(log_dir, "map", mp)))
    lanes = {}
    for lid, seg in avm.vector_lane_segments.items():
        left, right = seg.left_lane_boundary.xyz[:, :2], seg.right_lane_boundary.xyz[:, :2]
        poly = Polygon(np.vstack([left, right[::-1]])).buffer(0)
        lanes[lid] = Lane(
            id=lid, centerline=avm.get_lane_segment_centerline(lid)[:, :2], polygon=poly,
            successors=list(seg.successors), predecessors=list(seg.predecessors),
            left_neighbor=seg.left_neighbor_id, right_neighbor=seg.right_neighbor_id,
            is_intersection=bool(seg.is_intersection), drivable_by_car=seg.lane_type != LaneType.BIKE)
    drivable = [Polygon(da.xyz[:, :2]).buffer(0) for da in avm.vector_drivable_areas.values()]
    crossings = [Polygon(np.vstack([pc.edge1.xyz[:, :2], pc.edge2.xyz[::-1, :2]])).buffer(0)
                 for pc in avm.vector_pedestrian_crossings.values()]
    return LaneGraph(lanes, drivable), crossings


def load_track(log_dir: str) -> tuple[EgoTrack, float]:
    """Ego track in the city frame (meters) and the absolute start time [s]."""
    df = pd.read_feather(os.path.join(log_dir, "city_SE3_egovehicle.feather"))
    t = df["timestamp_ns"].to_numpy(np.float64) * 1e-9
    yaw = yaw_from_quat(df["qx"].to_numpy(), df["qy"].to_numpy(), df["qz"].to_numpy(), df["qw"].to_numpy())
    tr = track_from_poses(t, df["tx_m"].to_numpy(), df["ty_m"].to_numpy(), yaw)
    return tr, float(t.min())


def lead_queue_mask(log_dir: str, tr: EgoTrack, t0: float, ahead=(2.0, 20.0), half_width=1.8) -> np.ndarray | None:
    """Per-frame: a vehicle box center lies in the ego corridor ahead (AV2 boxes are in the ego frame)."""
    path = os.path.join(log_dir, "annotations.feather")
    if not os.path.exists(path):
        return None  # test split: annotations withheld
    a = pd.read_feather(path)
    veh = a[a["category"].isin(["REGULAR_VEHICLE", "LARGE_VEHICLE", "BUS", "BOX_TRUCK", "TRUCK", "TRUCK_CAB",
                                "VEHICULAR_TRAILER", "ARTICULATED_BUS", "SCHOOL_BUS", "MOTORCYCLE"])]
    hit = veh[(veh.tx_m > ahead[0]) & (veh.tx_m < ahead[1]) & (veh.ty_m.abs() < half_width)]
    times = np.unique(hit["timestamp_ns"].to_numpy()) * 1e-9 - t0
    mask = np.zeros(len(tr), bool)
    if len(times):
        k = np.clip(np.round(times * tr.hz).astype(int), 0, len(tr) - 1)
        mask[k] = True
        mask = np.convolve(mask, np.ones(3), mode="same") > 0  # bridge the 10 Hz sweep grid
    return mask
