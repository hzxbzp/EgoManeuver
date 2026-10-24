"""nuPlan adapter: GeoPackage maps -> LaneGraph, per-log SQLite DBs -> ego track + CAM_F0 image times.

Sensor blobs (*.zip on S3) are actually TAR archives (<archive>/<log_segment>/<CAM>/<token>.jpg), so
CAM_F0 is obtained by streaming an archive through a filtering tar (see data/build_nuplan.py).
The per-log DB archives are real zips (random access): only the needed .db files are fetched.
Singapore (sg-one-north) drives on the left.
"""

import glob
import os
import sqlite3

import numpy as np
import pandas as pd

from egokin.labels.kinematics import EgoTrack, track_from_poses, yaw_from_quat
from egokin.labels.lanegraph import Lane, LaneGraph

ROOT = os.environ.get("VEGO_DATA", "./vego_data")
RAW = f"{ROOT}/raw/nuplan"
S3 = "https://motional-nuplan.s3.ap-northeast-1.amazonaws.com/public/nuplan-v1.1"
LEFT_HAND = {"sg-one-north"}
DB_ZIPS = ["nuplan-v1.1_train_boston.zip", "nuplan-v1.1_train_pittsburgh.zip", "nuplan-v1.1_train_singapore.zip"] + \
          [f"nuplan-v1.1_train_vegas_{i}.zip" for i in range(1, 7)] + ["nuplan-v1.1_val.zip", "nuplan-v1.1_test.zip"]


def camera_archive_url(split: str, k: int) -> str:
    return f"{S3}/sensor_blobs/{split}_set/nuplan-v1.1_{split}_camera_{k}.zip"


def db_index() -> dict[str, tuple[str, str]]:
    """log segment name -> (db zip, member path), from the zip central directories (cached)."""
    cache = f"{RAW}/db_index.parquet"
    if os.path.exists(cache):
        df = pd.read_parquet(cache)
        return {r.log: (r.zip, r.member) for r in df.itertuples()}
    from remotezip import RemoteZip

    rows = []
    for zname in DB_ZIPS:
        with RemoteZip(f"{S3}/{zname}") as z:
            rows += [(os.path.basename(n)[:-3], zname, n) for n in z.namelist() if n.endswith(".db")]
    df = pd.DataFrame(rows, columns=["log", "zip", "member"])
    df.to_parquet(cache)
    return {r.log: (r.zip, r.member) for r in df.itertuples()}


def fetch_db(log: str, index: dict) -> str:
    from remotezip import RemoteZip

    zname, member = index[log]
    out = os.path.join(RAW, "db", member)
    if not os.path.exists(out):
        with RemoteZip(f"{S3}/{zname}") as z:
            z.extract(member, os.path.join(RAW, "db"))
    return out


def read_log(db_path: str):
    """(location, epsg, ego_pose df, CAM_F0 image df [timestamp, filename_jpg], scenario tags df)."""
    c = sqlite3.connect(db_path)
    # the map is keyed by map_version (Las Vegas logs say location 'las_vegas', map 'us-nv-las-vegas-strip')
    loc = c.execute("select map_version from log").fetchone()[0]
    epsg = int(c.execute("select epsg from ego_pose limit 1").fetchone()[0])
    ego = pd.read_sql("select timestamp, x, y, qw, qx, qy, qz from ego_pose order by timestamp", c)
    img = pd.read_sql("select i.timestamp, i.filename_jpg from image i join camera k on i.camera_token = k.token "
                      "where k.channel = 'CAM_F0' order by i.timestamp", c)
    tags = pd.read_sql("select t.type, l.timestamp from scenario_tag t join lidar_pc l on t.lidar_pc_token = l.token", c)
    c.close()
    return loc, epsg, ego, img, tags


def track_and_frames(ego: pd.DataFrame, img: pd.DataFrame) -> tuple[EgoTrack, np.ndarray, float]:
    """Ego track on the 10 Hz grid starting at the first CAM_F0 image; image times relative to it (s)."""
    t_img = img["timestamp"].to_numpy(np.float64) * 1e-6
    t0 = t_img[0]
    e = ego[(ego.timestamp * 1e-6 >= t0 - 1.0) & (ego.timestamp * 1e-6 <= t_img[-1] + 1.0)]
    yaw = yaw_from_quat(e.qx.to_numpy(), e.qy.to_numpy(), e.qz.to_numpy(), e.qw.to_numpy())
    te = e.timestamp.to_numpy(np.float64) * 1e-6
    # start the track at the first image so that label frame k == video frame k
    keep = te >= t0
    tr = track_from_poses(te[keep], e.x.to_numpy()[keep], e.y.to_numpy()[keep], yaw[keep])
    return tr, t_img - t0, t0


_graphs = {}


def lanegraph(location: str, epsg: int):
    """LaneGraph + crosswalks + carpark polygons of one nuPlan map (cached), reprojected from the
    GeoPackage's lon/lat to the UTM zone of the ego poses (`epsg` from the log's ego_pose table)."""
    key = (location, epsg)
    if key in _graphs:
        return _graphs[key]
    import pyogrio
    from shapely.ops import unary_union

    gpkg = sorted(glob.glob(f"{RAW}/nuplan-maps-v1.0/{location}/*/map.gpkg"))[-1]
    rd = lambda layer, **kw: pyogrio.read_dataframe(gpkg, layer=layer, **kw).to_crs(epsg=epsg)
    lanes_df = rd("lanes_polygons")
    base = rd("baseline_paths")
    conn = rd("lane_connectors", fid_as_index=True)
    conn_poly = rd("gen_lane_connectors_scaled_width_polygons")

    cl_lane = {int(r.lane_fid): np.asarray(r.geometry.coords)[:, :2] for r in base[base.lane_fid.notna()].itertuples()}
    cl_conn = {int(r.lane_connector_fid): np.asarray(r.geometry.coords)[:, :2] for r in base[base.lane_connector_fid.notna()].itertuples()}
    poly_conn = {int(r.lane_connector_fid): r.geometry for r in conn_poly.itertuples()}

    lanes = {}
    for r in lanes_df.itertuples():
        fid = int(r.lane_fid)
        if fid in cl_lane and len(cl_lane[fid]) >= 2:
            lanes[f"L{fid}"] = Lane(id=f"L{fid}", centerline=cl_lane[fid], polygon=r.geometry.buffer(0))
    for fid, r in conn.iterrows():
        fid = int(fid)
        if fid in cl_conn and fid in poly_conn and len(cl_conn[fid]) >= 2:
            lanes[f"C{fid}"] = Lane(id=f"C{fid}", centerline=cl_conn[fid], polygon=poly_conn[fid].buffer(0),
                                    is_intersection=not pd.isna(r.intersection_fid))
    for fid, r in conn.iterrows():
        c, a, b = f"C{int(fid)}", f"L{int(r.exit_lane_fid)}", f"L{int(r.entry_lane_fid)}"
        if c in lanes and a in lanes and b in lanes:
            lanes[a].successors.append(c)
            lanes[c].predecessors.append(a)
            lanes[c].successors.append(b)
            lanes[b].predecessors.append(c)
    # left/right neighbors: adjacent lane_index within a lane group, side from geometry
    for _, grp in lanes_df.groupby("lane_group_fid"):
        ids = [f"L{int(f)}" for f in grp.sort_values("lane_index").lane_fid if f"L{int(f)}" in lanes]
        for a, b in zip(ids[:-1], ids[1:]):
            la, lb = lanes[a], lanes[b]
            mid = la.centerline[len(la.centerline) // 2]
            _, d_b, _ = lb.project(*mid)  # where a lies relative to b (left > 0)
            if d_b > 0:
                la.right_neighbor, lb.left_neighbor = b, a
            else:
                la.left_neighbor, lb.right_neighbor = b, a
    drivable = list(rd("generic_drivable_areas").geometry) + [l.polygon for l in lanes.values()]
    drivable = [unary_union([g.buffer(0) for g in drivable])]
    crossings = list(rd("crosswalks").geometry)
    carparks = list(rd("carpark_areas").geometry)
    _graphs[key] = (LaneGraph(lanes, drivable), crossings, carparks)
    return _graphs[key]
