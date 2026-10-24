"""nuScenes adapter: CAM_FRONT sweeps (~12 Hz), ego poses per image, and the map expansion as a
LaneGraph. The map expansion has no lane adjacency, so left/right neighbors are inferred
geometrically (parallel lanes whose polygons touch on that side). Singapore drives on the left.
"""

import glob
import os

import numpy as np

from egokin.labels.kinematics import EgoTrack, track_from_poses, yaw_from_quat
from egokin.labels.lanegraph import Lane, LaneGraph

ROOT = os.environ.get("VEGO_DATA", "./vego_data")
RAW = f"{ROOT}/raw/nuscenes"
LEFT_HAND = {"singapore-onenorth", "singapore-queenstown", "singapore-hollandvillage"}

_nusc = None


def nusc():
    global _nusc
    if _nusc is None:
        from nuscenes.nuscenes import NuScenes

        _nusc = NuScenes(version="v1.0-trainval", dataroot=f"{RAW}/meta", verbose=False)
    return _nusc


def image_index() -> dict[str, str]:
    """basename -> path over the streamed CAM_FRONT parts."""
    out = {}
    for p in glob.glob(f"{RAW}/part_*/*/CAM_FRONT/*.jpg"):
        out[os.path.basename(p)] = p
    return out


def scene_frames(scene: dict) -> list[dict]:
    """All CAM_FRONT sample_data records (keyframes + sweeps) of a scene, in time order."""
    n = nusc()
    first = n.get("sample", scene["first_sample_token"])
    sd = n.get("sample_data", first["data"]["CAM_FRONT"])
    while sd["prev"]:
        sd = n.get("sample_data", sd["prev"])
    out = []
    end_t = n.get("sample", scene["last_sample_token"])["timestamp"]
    while sd is not None and sd["timestamp"] <= end_t + 1e5:
        out.append(sd)
        sd = n.get("sample_data", sd["next"]) if sd["next"] else None
    return out


def scene_track(frames: list[dict]) -> tuple[EgoTrack, np.ndarray]:
    """Ego track (global map frame) on the 10 Hz grid from the first image; image times (s)."""
    n = nusc()
    t = np.array([f["timestamp"] for f in frames], float) * 1e-6
    poses = [n.get("ego_pose", f["ego_pose_token"]) for f in frames]
    xy = np.array([p["translation"][:2] for p in poses])
    q = np.array([p["rotation"] for p in poses])  # w, x, y, z
    yaw = yaw_from_quat(q[:, 1], q[:, 2], q[:, 3], q[:, 0])
    return track_from_poses(t, xy[:, 0], xy[:, 1], yaw), t - t[0]


def scene_location(scene: dict) -> str:
    return nusc().get("log", scene["log_token"])["location"]


_graphs = {}


def lanegraph(location: str):
    """LaneGraph + pedestrian crossings for one map (cached)."""
    if location in _graphs:
        return _graphs[location]
    from nuscenes.map_expansion import arcline_path_utils
    from nuscenes.map_expansion.map_api import NuScenesMap
    from shapely.strtree import STRtree

    m = NuScenesMap(dataroot=f"{RAW}/meta", map_name=location)
    lanes = {}
    for rec, inter in [(r, False) for r in m.lane] + [(r, True) for r in m.lane_connector]:
        tok = rec["token"]
        path = m.arcline_path_3.get(tok)
        if not path or not rec.get("polygon_token"):
            continue
        cl = np.array(arcline_path_utils.discretize_lane(path, resolution_meters=1.0))[:, :2]
        if len(cl) < 2:
            continue
        lanes[tok] = Lane(id=tok, centerline=cl, polygon=m.extract_polygon(rec["polygon_token"]).buffer(0),
                          successors=m.get_outgoing_lane_ids(tok), predecessors=m.get_incoming_lane_ids(tok),
                          is_intersection=inter)
    _infer_neighbors(lanes, STRtree)
    drivable = [m.extract_polygon(p) for rec in m.drivable_area for p in rec["polygon_tokens"]]
    crossings = [m.extract_polygon(r["polygon_token"]) for r in m.ped_crossing]
    _graphs[location] = (LaneGraph(lanes, drivable), crossings)
    return _graphs[location]


def _infer_neighbors(lanes: dict, STRtree, max_gap=0.5, max_angle_deg=20):
    """Left/right same-direction neighbors of non-intersection lanes from geometry."""
    ids = [k for k, l in lanes.items() if not l.is_intersection]
    tree = STRtree([lanes[k].polygon for k in ids])
    for k in ids:
        a = lanes[k]
        mid = a.centerline[len(a.centerline) // 2]
        _, _, hdg = a.project(*mid)
        best = {1: (np.inf, None), -1: (np.inf, None)}
        for j in tree.query(a.polygon.buffer(max_gap)):
            b = lanes[ids[j]]
            if b.id == a.id:
                continue
            _, d_mid, hb = b.project(*mid)  # where a's midpoint lies relative to b
            if abs(np.degrees((hb - hdg + np.pi) % (2 * np.pi) - np.pi)) > max_angle_deg:
                continue
            if a.polygon.distance(b.polygon) > max_gap:
                continue
            side = -1 if d_mid > 0 else 1  # a is left of b => b is a's right neighbor
            if abs(d_mid) < best[side][0] and 2.0 < abs(d_mid) < 6.0:
                best[side] = (abs(d_mid), b.id)
        a.left_neighbor, a.right_neighbor = best[1][1], best[-1][1]
