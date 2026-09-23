from __future__ import annotations

from dataclasses import dataclass, field
from math import hypot

from .geometry import BBox, box_iou, center


@dataclass(frozen=True)
class Detection:
    bbox: BBox
    cls_id: int
    confidence: float


@dataclass
class Track:
    track_id: int
    cls_id: int
    bbox: BBox
    confidence: float
    last_seen: float
    missed: int = 0
    history: list[tuple[float, float, float, BBox]] = field(default_factory=list)

    def add(self, detection: Detection, t_sec: float, frame_width: int, frame_height: int) -> None:
        self.bbox = detection.bbox
        self.confidence = detection.confidence
        self.last_seen = t_sec
        self.missed = 0
        cx, cy = center(detection.bbox)
        self.history.append((t_sec, cx / max(frame_width, 1), cy / max(frame_height, 1), detection.bbox))
        if len(self.history) > 300:
            self.history = self.history[-300:]

    def velocity(self, lookback_sec: float = 1.0) -> tuple[float, float]:
        if len(self.history) < 2:
            return (0.0, 0.0)
        latest = self.history[-1]
        earliest = self.history[0]
        for sample in reversed(self.history[:-1]):
            earliest = sample
            if latest[0] - sample[0] >= lookback_sec:
                break
        dt = latest[0] - earliest[0]
        if dt <= 1e-6:
            return (0.0, 0.0)
        return ((latest[1] - earliest[1]) / dt, (latest[2] - earliest[2]) / dt)

    def speed(self, lookback_sec: float = 1.0) -> float:
        vx, vy = self.velocity(lookback_sec)
        return hypot(vx, vy)


class CentroidTracker:
    """Small deterministic tracker for a baseline, using IoU + centre distance."""

    def __init__(self, max_missed: int = 8, min_iou: float = 0.05, max_distance: float = 0.12):
        self.max_missed = max_missed
        self.min_iou = min_iou
        self.max_distance = max_distance
        self.next_id = 1
        self.tracks: dict[int, Track] = {}
        self.finished: list[Track] = []

    def update(
        self,
        detections: list[Detection],
        t_sec: float,
        frame_width: int,
        frame_height: int,
    ) -> list[Track]:
        candidates: list[tuple[float, int, int]] = []
        for track_id, track in self.tracks.items():
            tcx, tcy = center(track.bbox)
            for det_idx, det in enumerate(detections):
                if det.cls_id != track.cls_id:
                    continue
                dcx, dcy = center(det.bbox)
                dist = hypot((dcx - tcx) / max(frame_width, 1), (dcy - tcy) / max(frame_height, 1))
                iou = box_iou(track.bbox, det.bbox)
                if iou >= self.min_iou or dist <= self.max_distance:
                    score = iou - 0.35 * dist
                    candidates.append((score, track_id, det_idx))

        used_tracks: set[int] = set()
        used_detections: set[int] = set()
        for _, track_id, det_idx in sorted(candidates, reverse=True):
            if track_id in used_tracks or det_idx in used_detections:
                continue
            self.tracks[track_id].add(detections[det_idx], t_sec, frame_width, frame_height)
            used_tracks.add(track_id)
            used_detections.add(det_idx)

        for track_id in list(self.tracks):
            if track_id not in used_tracks:
                self.tracks[track_id].missed += 1
                if self.tracks[track_id].missed > self.max_missed:
                    self.finished.append(self.tracks.pop(track_id))

        for det_idx, detection in enumerate(detections):
            if det_idx in used_detections:
                continue
            track = Track(
                track_id=self.next_id,
                cls_id=detection.cls_id,
                bbox=detection.bbox,
                confidence=detection.confidence,
                last_seen=t_sec,
            )
            track.add(detection, t_sec, frame_width, frame_height)
            self.tracks[self.next_id] = track
            self.next_id += 1

        return [t for t in self.tracks.values() if t.missed == 0]

    def all_tracks(self) -> list[Track]:
        return [*self.finished, *self.tracks.values()]
