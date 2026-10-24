"""Label factory entry point: ego track (+ optional map) -> per-frame A/B labels and events."""

from dataclasses import dataclass, field

import numpy as np

from egokin.labels.kinematics import EgoTrack
from egokin.labels.lateral_nomap import LatEvent, label_lateral_nomap
from egokin.labels.longitudinal import label_longitudinal
from egokin.labels.segments import to_segments

LABEL_VERSION = "v0"


@dataclass
class ClipLabels:
    hz: float
    long: np.ndarray  # Long per frame
    lat: np.ndarray  # Lat per frame (IGNORE where undefined)
    strong: np.ndarray  # |a| above the intensity threshold
    hard_brake: np.ndarray
    events: list[LatEvent]
    source: str  # "nomap" or "map"
    context: dict = field(default_factory=dict)  # name -> bool per frame (map-derived when available)
    version: str = LABEL_VERSION

    def segments(self):
        return {"long": to_segments(self.long), "lat": to_segments(self.lat)}


def label_clip(track: EgoTrack, right_hand_traffic: bool = True, lanegraph=None, crossings=None,
               lead_queue=None) -> ClipLabels:
    long_lab, strong, hard = label_longitudinal(track)
    context = {}
    if lanegraph is not None:
        from shapely.geometry import Point

        from egokin.labels.lateral_map import label_lateral_map

        lat = label_lateral_map(track, long_lab, lanegraph, right_hand_traffic, lead_queue=lead_queue)
        source = "map"
        context["intersection"] = lat.in_intersection
        if crossings:
            context["crosswalk"] = np.array([any(c.distance(Point(x, y)) < 2.0 for c in crossings)
                                             for x, y in zip(track.x, track.y)])
    else:
        lat = label_lateral_nomap(track, long_lab, right_hand_traffic)
        source = "nomap"
    out = ClipLabels(track.hz, long_lab, lat.labels, strong, hard, lat.events, source, context)
    out.v_lat = lat.v_lat
    out.road = getattr(lat, "road", None)  # map-based road-relative state, None without a map
    return out
