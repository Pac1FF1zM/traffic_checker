from __future__ import annotations

import sys
from math import isclose
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.baseline import _times_to_intervals, collision_risk
from src.geometry import crossed_line, merge_intervals, point_in_polygon
from src.temporal import fuse_risk, prepare_frame
from src.tracking import Detection, Track


def make_track(track_id: int, points: list[tuple[float, float, float]]) -> Track:
    track = Track(track_id, 2, (0.0, 0.0, 10.0, 10.0), 1.0, points[-1][0])
    track.history = [(t, x, y, track.bbox) for t, x, y in points]
    return track


def test_geometry() -> None:
    square = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
    assert point_in_polygon((0.5, 0.5), square)
    assert not point_in_polygon((1.5, 0.5), square)
    assert crossed_line((0.2, 0.4), (0.8, 0.4), [[0.5, 0.0], [0.5, 1.0]])


def test_interval_postprocessing() -> None:
    assert merge_intervals([(0.0, 1.0), (1.2, 2.0)], gap=0.25) == [(0.0, 2.0)]
    intervals = _times_to_intervals([0.0, 0.2, 0.4, 2.0], 0.2, 0.3, 0.3)
    assert len(intervals) == 1 and intervals[0][0] == 0.0 and isclose(intervals[0][1], 0.6)


def test_ttc_risk_separates_approaching_and_parallel_tracks() -> None:
    approaching_a = make_track(1, [(0.0, 0.2, 0.5), (1.0, 0.3, 0.5)])
    approaching_b = make_track(2, [(0.0, 0.8, 0.5), (1.0, 0.7, 0.5)])
    parallel_b = make_track(3, [(0.0, 0.8, 0.5), (1.0, 0.9, 0.5)])
    assert collision_risk([approaching_a, approaching_b]) > 0.2
    assert collision_risk([approaching_a, parallel_b]) == 0.0


def test_strict_ttc_rejects_a_tracker_identity_jump() -> None:
    unstable = make_track(
        1,
        [(0.0, 0.20, 0.50), (0.2, 0.22, 0.50), (0.4, 0.24, 0.50),
         (0.6, 0.26, 0.50), (0.8, 0.28, 0.50), (1.0, 0.55, 0.50)],
    )
    other = make_track(
        2,
        [(0.0, 0.80, 0.50), (0.2, 0.78, 0.50), (0.4, 0.76, 0.50),
         (0.6, 0.74, 0.50), (0.8, 0.72, 0.50), (1.0, 0.70, 0.50)],
    )
    assert collision_risk(
        [unstable, other], min_history=6, min_track_age=0.6, max_instant_speed=0.45
    ) == 0.0


def test_temporal_preprocessing_and_fusion() -> None:
    import numpy as np

    frame = np.zeros((32, 48, 3), dtype=np.uint8)
    prepared = prepare_frame(frame)
    assert prepared.shape == (3, 224, 224)
    assert prepared.dtype == np.float32
    assert isclose(fuse_risk(0.5, 0.5), 0.75)
    assert fuse_risk(-1.0, 2.0) == 1.0


if __name__ == "__main__":
    test_geometry()
    test_interval_postprocessing()
    test_ttc_risk_separates_approaching_and_parallel_tracks()
    test_strict_ttc_rejects_a_tracker_identity_jump()
    test_temporal_preprocessing_and_fusion()
    print("baseline unit tests: OK")
