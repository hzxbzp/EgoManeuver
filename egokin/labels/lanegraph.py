"""Dataset-agnostic lane graph used by the map-based labeler.

Each source (AV2, nuScenes, nuPlan, Waymo, nuReasoning) converts its map into this form.
Coordinates are planar meters in the same frame as the ego track.
"""

from dataclasses import dataclass, field

import numpy as np
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union
from shapely.strtree import STRtree


@dataclass
class Lane:
    id: int
    centerline: np.ndarray  # (N, 2), in driving direction
    polygon: Polygon
    successors: list = field(default_factory=list)
    predecessors: list = field(default_factory=list)
    left_neighbor: int | None = None  # same-direction lane to the left
    right_neighbor: int | None = None
    is_intersection: bool = False
    drivable_by_car: bool = True

    def __post_init__(self):
        d = np.diff(self.centerline, axis=0)
        seg = np.hypot(d[:, 0], d[:, 1])
        self._s = np.concatenate([[0.0], np.cumsum(seg)])
        self._hdg = np.arctan2(d[:, 1], d[:, 0])

    @property
    def length(self) -> float:
        return float(self._s[-1])

    def heading_change(self) -> float:
        """Signed heading change start -> end in radians (left turn > 0)."""
        return float(np.unwrap(self._hdg)[-1] - self._hdg[0])

    def project(self, x: float, y: float):
        """(s, d, heading): arc length, signed lateral offset (left > 0) and lane heading at the projection."""
        p = np.array([x, y])
        a, b = self.centerline[:-1], self.centerline[1:]
        ab = b - a
        t = np.clip(np.einsum("ij,ij->i", p - a, ab) / np.maximum(np.einsum("ij,ij->i", ab, ab), 1e-9), 0, 1)
        proj = a + ab * t[:, None]
        dist = np.hypot(*(p - proj).T)
        i = int(np.argmin(dist))
        cross = ab[i, 0] * (p[1] - a[i, 1]) - ab[i, 1] * (p[0] - a[i, 0])
        return self._s[i] + t[i] * np.hypot(*ab[i]), float(np.sign(cross) * dist[i]), float(self._hdg[i])


class LaneGraph:
    def __init__(self, lanes: dict[int, Lane], drivable: list[Polygon]):
        self.lanes = {k: v for k, v in lanes.items() if v.drivable_by_car}
        self._ids = list(self.lanes)
        self._tree = STRtree([self.lanes[i].polygon for i in self._ids])
        self.drivable = unary_union(drivable) if drivable else None

    def candidates(self, x: float, y: float, buffer: float = 0.3) -> list[int]:
        pt = Point(x, y)
        hits = self._tree.query(pt.buffer(buffer))
        return [self._ids[i] for i in hits if self.lanes[self._ids[i]].polygon.distance(pt) <= buffer]

    def local_edge(self, xs, ys, margin: float = 60.0):
        """Drivable-area boundary clipped to the trajectory's bounding box (fast per-frame distances)."""
        from shapely.geometry import box

        if self.drivable is None:
            return None
        b = box(min(xs) - margin, min(ys) - margin, max(xs) + margin, max(ys) + margin)
        return self.drivable.boundary.intersection(b)

    def dist_to_edge(self, x: float, y: float) -> float:
        """Distance from a point to the drivable-area boundary (inf if no drivable area)."""
        if self.drivable is None:
            return float("inf")
        return float(self.drivable.boundary.distance(Point(x, y)))

    def relation(self, a: int, b: int) -> str:
        """How lane b is reached from lane a: same / succ / pred / left / right / left_succ / right_succ / other."""
        if a == b:
            return "same"
        la = self.lanes[a]
        if b in la.successors:
            return "succ"
        if b in la.predecessors:
            return "pred"
        if b == la.left_neighbor:
            return "left"
        if b == la.right_neighbor:
            return "right"
        for s in la.successors:
            ls = self.lanes.get(s)
            if ls is None:
                continue
            if b == ls.left_neighbor:
                return "left_succ"
            if b == ls.right_neighbor:
                return "right_succ"
            if b in ls.successors:
                return "succ"
        for p in la.predecessors:
            lp = self.lanes.get(p)
            if lp is not None and b in (lp.left_neighbor, lp.right_neighbor):
                return "left_succ" if b == lp.left_neighbor else "right_succ"
        return "other"
