from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any, Iterable

import cv2
import numpy as np

from .geometry import box_iou, crossed_line, merge_intervals, point_in_polygon
from .temporal import TemporalRiskModel, create_temporal_model, fuse_risk
from .tracking import CentroidTracker, Detection, Track

PROJECT_ROOT = Path(__file__).resolve().parents[1]
COCO_PERSON = 0
COCO_VEHICLES = {1, 2, 3, 5, 7}  # bicycle, car, motorcycle, bus, truck
COCO_ROAD_USERS = {COCO_PERSON, *COCO_VEHICLES}


DEFAULT_CONFIG: dict[str, Any] = {
    "model_path": "weights/yolo11n.pt",
    "device": "auto",
    "imgsz": 640,
    "confidence": 0.30,
    "nms_iou": 0.55,
    "frame_stride": 3,
    "risk_frame_stride": 5,
    "road_polygon": [],
    "crosswalk_polygon": [],
    "intersection_polygon": [],
    "traffic_light_roi": [],
    "stop_line": [],
    "solid_lines": [],
    "lanes": [],
    "stopped_speed": 0.004,
    "moving_speed": 0.008,
    "stopped_min_sec": 10.0,
    "wrong_way_min_sec": 1.5,
    "jaywalking_min_sec": 0.5,
    "congestion_min_vehicles": 6,
    "congestion_speed": 0.006,
    "congestion_min_sec": 5.0,
    "near_miss_threshold": 0.72,
    "near_miss_min_sec": 0.35,
    "enable_near_miss_heuristic": True,
    "enable_accident_heuristic": False,
    "accident_iou": 0.18,
    "red_light_event_sec": 3.0,
    "event_merge_gap_sec": 0.75,
    "temporal_enabled": True,
    "temporal_required": True,
    "temporal_source_path": "third_party/simple_tad/run_inference_simple.py",
    "temporal_checkpoint_path": "weights/simpletad_ft-dota_dapt-vm1-s_auroc.pth",
    "temporal_arch": "videomae_small",
    "temporal_device": "auto",
    "temporal_window_frames": 16,
    "temporal_view_fps": 10.0,
    "temporal_inference_stride": 4,
    "temporal_accident_threshold": 0.75,
    "temporal_accident_min_sec": 0.4,
}


def load_config() -> dict[str, Any]:
    configured = os.getenv("WIUT_SCENE_CONFIG")
    candidates = [Path(configured)] if configured else []
    candidates += [PROJECT_ROOT / "configs" / "scene.json", PROJECT_ROOT / "configs" / "scene.example.json"]
    config = dict(DEFAULT_CONFIG)
    for path in candidates:
        if path and path.is_file():
            config.update(json.loads(path.read_text(encoding="utf-8")))
            config["_source"] = str(path)
            break
    return config


class YoloDetector:
    _MODEL_CACHE: dict[str, Any] = {}

    def __init__(self, config: dict[str, Any]):
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError("ultralytics is required; run `pip install -r requirements.txt`") from exc

        model_path = Path(str(config["model_path"]))
        if not model_path.is_absolute():
            model_path = PROJECT_ROOT / model_path
        if not model_path.is_file():
            raise FileNotFoundError(
                f"YOLO weights not found at {model_path}. Run `python scripts/download_weights.py`."
            )
        cache_key = str(model_path.resolve())
        if cache_key not in self._MODEL_CACHE:
            self._MODEL_CACHE[cache_key] = YOLO(str(model_path))
        self.model = self._MODEL_CACHE[cache_key]
        self.imgsz = int(config["imgsz"])
        self.conf = float(config["confidence"])
        self.iou = float(config["nms_iou"])
        requested_device = str(config.get("device", "auto"))
        if requested_device == "auto":
            try:
                import torch

                requested_device = "0" if torch.cuda.is_available() else "cpu"
            except ImportError:
                requested_device = "cpu"
        self.device = requested_device

    def __call__(self, frame: np.ndarray) -> list[Detection]:
        result = self.model.predict(
            source=frame,
            imgsz=self.imgsz,
            conf=self.conf,
            iou=self.iou,
            classes=sorted(COCO_ROAD_USERS),
            device=self.device,
            verbose=False,
        )[0]
        if result.boxes is None:
            return []
        xyxy = result.boxes.xyxy.detach().cpu().numpy()
        cls = result.boxes.cls.detach().cpu().numpy().astype(int)
        conf = result.boxes.conf.detach().cpu().numpy()
        return [
            Detection(tuple(map(float, box)), int(class_id), float(score))
            for box, class_id, score in zip(xyxy, cls, conf)
        ]


def _sample_velocity(history: list[tuple[float, float, float, Any]], index: int, lookback: float = 1.0) -> tuple[float, float]:
    if index <= 0:
        return (0.0, 0.0)
    current = history[index]
    previous = history[index - 1]
    for candidate in reversed(history[:index]):
        previous = candidate
        if current[0] - candidate[0] >= lookback:
            break
    dt = current[0] - previous[0]
    if dt <= 1e-6:
        return (0.0, 0.0)
    return ((current[1] - previous[1]) / dt, (current[2] - previous[2]) / dt)


def collision_risk(
    tracks: Iterable[Track],
    horizon: float = 5.0,
    *,
    miss_threshold: float = 0.12,
    min_history: int = 2,
    min_track_age: float = 0.0,
    max_instant_speed: float | None = None,
    ttc_scale: float = 3.0,
    proximity_scale: float = 0.045,
) -> float:
    """Causal TTC score in [0, 1] from the latest state of active tracks."""
    stable_tracks: list[Track] = []
    for track in tracks:
        if track.cls_id not in COCO_ROAD_USERS or len(track.history) < max(2, min_history):
            continue
        if track.history[-1][0] - track.history[0][0] < min_track_age:
            continue
        if max_instant_speed is not None:
            latest, previous = track.history[-1], track.history[-2]
            dt = latest[0] - previous[0]
            if dt <= 1e-6:
                continue
            instant_speed = math.hypot(latest[1] - previous[1], latest[2] - previous[2]) / dt
            if instant_speed > max_instant_speed:
                continue
        stable_tracks.append(track)
    tracks = stable_tracks
    best = 0.0
    for i, left in enumerate(tracks):
        lx, ly = left.history[-1][1:3]
        lvx, lvy = left.velocity(1.0)
        for right in tracks[i + 1 :]:
            rx, ry = right.history[-1][1:3]
            rvx, rvy = right.velocity(1.0)
            px, py = rx - lx, ry - ly
            vx, vy = rvx - lvx, rvy - lvy
            v2 = vx * vx + vy * vy
            if v2 < 1e-7:
                continue
            ttc = -(px * vx + py * vy) / v2
            if not 0.0 < ttc <= horizon:
                continue
            miss = math.hypot(px + vx * ttc, py + vy * ttc)
            if miss > miss_threshold:
                continue
            approach = math.exp(-ttc / max(ttc_scale, 1e-6))
            proximity = math.exp(-miss / max(proximity_scale, 1e-6))
            best = max(best, approach * proximity)
    return float(min(1.0, max(0.0, best)))


def _times_to_intervals(times: list[float], sample_period: float, gap: float, minimum: float) -> list[tuple[float, float]]:
    if not times:
        return []
    raw: list[tuple[float, float]] = []
    start = previous = times[0]
    for t_sec in times[1:]:
        if t_sec - previous > max(gap, sample_period * 1.8):
            raw.append((start, previous + sample_period))
            start = t_sec
        previous = t_sec
    raw.append((start, previous + sample_period))
    return merge_intervals(raw, gap=gap, minimum=minimum)


def _signal_is_red(frame: np.ndarray, roi: list[list[float]]) -> bool:
    if len(roi) != 4:
        return False
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = roi
    x1, x2 = sorted((max(0, min(width, int(x1 * width))), max(0, min(width, int(x2 * width)))))
    y1, y2 = sorted((max(0, min(height, int(y1 * height))), max(0, min(height, int(y2 * height)))))
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return False
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, np.array([0, 90, 80]), np.array([10, 255, 255]))
    mask2 = cv2.inRange(hsv, np.array([170, 90, 80]), np.array([180, 255, 255]))
    red_ratio = float(np.count_nonzero(mask1 | mask2)) / float(mask1.size)
    return red_ratio >= 0.015


@dataclass
class FrameState:
    t_sec: float
    red: bool
    track_ids: tuple[int, ...]


class EventAnalyzer:
    def __init__(self, config: dict[str, Any], fps: float, width: int, height: int):
        stride = max(1, int(config["frame_stride"]))
        max_missed = max(3, round(float(fps) / stride * 1.5))
        self.config = config
        self.fps = fps
        self.width = width
        self.height = height
        self.sample_period = stride / max(fps, 1e-6)
        self.tracker = CentroidTracker(
            max_missed=max_missed,
            min_iou=float(config.get("tracker_min_iou", 0.05)),
            max_distance=float(config.get("tracker_max_distance", 0.12)),
        )
        self.states: list[FrameState] = []
        self.near_miss_times: list[float] = []
        self.accident_times: list[float] = []
        self.failure_to_yield_times: list[float] = []
        self.congestion_times: list[float] = []

    def record_temporal_risk(self, t_sec: float, score: float) -> None:
        if score >= float(self.config["temporal_accident_threshold"]):
            self.accident_times.append(t_sec)

    def update(self, frame: np.ndarray, detections: list[Detection], t_sec: float) -> None:
        active = self.tracker.update(detections, t_sec, self.width, self.height)
        red = _signal_is_red(frame, self.config["traffic_light_roi"])
        self.states.append(FrameState(t_sec, red, tuple(t.track_id for t in active)))

        risk = collision_risk(active)
        if self.config["enable_near_miss_heuristic"] and risk >= float(self.config["near_miss_threshold"]):
            self.near_miss_times.append(t_sec)

        if self.config["enable_accident_heuristic"]:
            road_users = [t for t in active if t.cls_id in COCO_ROAD_USERS]
            if any(
                box_iou(a.bbox, b.bbox) >= float(self.config["accident_iou"])
                and a.speed() + b.speed() >= 0.02
                for i, a in enumerate(road_users)
                for b in road_users[i + 1 :]
            ):
                self.accident_times.append(t_sec)

        vehicles = [t for t in active if t.cls_id in COCO_VEHICLES]
        if len(vehicles) >= int(self.config["congestion_min_vehicles"]):
            if median(t.speed() for t in vehicles) <= float(self.config["congestion_speed"]):
                self.congestion_times.append(t_sec)

        crosswalk = self.config["crosswalk_polygon"]
        if crosswalk:
            people_on_crosswalk = any(
                t.cls_id == COCO_PERSON and point_in_polygon(t.history[-1][1:3], crosswalk) for t in active
            )
            vehicles_on_crosswalk = any(
                t.cls_id in COCO_VEHICLES and point_in_polygon(t.history[-1][1:3], crosswalk) for t in active
            )
            if people_on_crosswalk and vehicles_on_crosswalk:
                self.failure_to_yield_times.append(t_sec)

    def _stopped_intervals(self, track: Track) -> list[tuple[float, float]]:
        if track.cls_id not in COCO_VEHICLES or len(track.history) < 3:
            return []
        intervals: list[tuple[float, float]] = []
        run_start: float | None = None
        last_t = track.history[0][0]
        for i in range(1, len(track.history)):
            t_sec = track.history[i][0]
            vx, vy = _sample_velocity(track.history, i)
            speed = math.hypot(vx, vy)
            if speed <= float(self.config["stopped_speed"]):
                run_start = last_t if run_start is None else run_start
            elif speed >= float(self.config["moving_speed"]):
                if run_start is not None and last_t - run_start >= float(self.config["stopped_min_sec"]):
                    intervals.append((run_start, t_sec))
                run_start = None
            last_t = t_sec
        if run_start is not None and last_t - run_start >= float(self.config["stopped_min_sec"]):
            intervals.append((run_start, last_t + self.sample_period))
        return intervals

    def _track_rule_times(self, track: Track) -> tuple[list[float], list[float]]:
        wrong_way: list[float] = []
        jaywalking: list[float] = []
        road = self.config["road_polygon"]
        crosswalk = self.config["crosswalk_polygon"]
        lanes = self.config["lanes"]
        for i, sample in enumerate(track.history):
            t_sec, x, y, _ = sample
            if track.cls_id == COCO_PERSON and road and point_in_polygon((x, y), road):
                if not crosswalk or not point_in_polygon((x, y), crosswalk):
                    jaywalking.append(t_sec)
            if track.cls_id in COCO_VEHICLES and i > 0:
                vx, vy = _sample_velocity(track.history, i)
                speed = math.hypot(vx, vy)
                for lane in lanes:
                    direction = lane.get("direction", [0.0, 0.0])
                    polygon = lane.get("polygon", [])
                    if polygon and point_in_polygon((x, y), polygon) and speed >= 0.008:
                        norm = math.hypot(float(direction[0]), float(direction[1])) or 1.0
                        alignment = (vx * float(direction[0]) + vy * float(direction[1])) / (speed * norm)
                        if alignment < -0.45:
                            wrong_way.append(t_sec)
                        break
        return wrong_way, jaywalking

    def _line_events(self, track: Track, duration: float) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
        red_light: list[tuple[float, float]] = []
        solid_crossing: list[tuple[float, float]] = []
        if track.cls_id not in COCO_VEHICLES:
            return red_light, solid_crossing
        red_by_time = {round(s.t_sec, 3): s.red for s in self.states}
        for previous, current in zip(track.history, track.history[1:]):
            a, b = previous[1:3], current[1:3]
            t_sec = current[0]
            if crossed_line(a, b, self.config["stop_line"]) and red_by_time.get(round(t_sec, 3), False):
                red_light.append((t_sec, min(duration, t_sec + float(self.config["red_light_event_sec"]))))
            if any(crossed_line(a, b, line) for line in self.config["solid_lines"]):
                solid_crossing.append((max(0.0, previous[0]), min(duration, t_sec + self.sample_period)))
        return red_light, solid_crossing

    def events(self, duration: float) -> list[list]:
        by_label: dict[str, list[tuple[float, float]]] = {}
        wrong_times: list[float] = []
        jaywalking_times: list[float] = []
        for track in self.tracker.all_tracks():
            by_label.setdefault("stopped_vehicle", []).extend(self._stopped_intervals(track))
            wrong, jaywalking = self._track_rule_times(track)
            wrong_times.extend(wrong)
            jaywalking_times.extend(jaywalking)
            red, solid = self._line_events(track, duration)
            by_label.setdefault("red_light", []).extend(red)
            by_label.setdefault("solid_line_crossing", []).extend(solid)

        gap = float(self.config["event_merge_gap_sec"])
        by_label["wrong_way"] = _times_to_intervals(
            sorted(wrong_times), self.sample_period, gap, float(self.config["wrong_way_min_sec"])
        )
        by_label["jaywalking"] = _times_to_intervals(
            sorted(jaywalking_times), self.sample_period, gap, float(self.config["jaywalking_min_sec"])
        )
        by_label["near_miss"] = _times_to_intervals(
            self.near_miss_times, self.sample_period, gap, float(self.config["near_miss_min_sec"])
        )
        temporal_period = float(self.config["temporal_inference_stride"]) / max(
            float(self.config["temporal_view_fps"]), 1e-6
        )
        accident_period = max(self.sample_period, temporal_period)
        by_label["accident"] = _times_to_intervals(
            self.accident_times,
            accident_period,
            gap,
            float(self.config["temporal_accident_min_sec"]),
        )
        by_label["failure_to_yield"] = _times_to_intervals(
            self.failure_to_yield_times, self.sample_period, gap, 0.4
        )
        by_label["congestion"] = _times_to_intervals(
            self.congestion_times, self.sample_period, gap, float(self.config["congestion_min_sec"])
        )

        output: list[list] = []
        for label, intervals in by_label.items():
            for start, end in merge_intervals(intervals, gap=gap):
                start, end = max(0.0, start), min(duration, end)
                if end > start:
                    output.append([round(start, 3), round(end, 3), label])
        return sorted(output, key=lambda event: (event[0], event[2]))


class OnlineRiskEstimator:
    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.detector: YoloDetector | None = None
        self.tracker: CentroidTracker | None = None
        self.frame_index = 0
        self.last_score = 0.0
        self.width = 1
        self.height = 1
        self.temporal: TemporalRiskModel | None = None
        self.last_ttc_score = 0.0
        self.last_temporal_score = 0.0

    def reset(self, meta: dict[str, Any]) -> None:
        self.detector = YoloDetector(self.config)
        stride = max(1, int(self.config["risk_frame_stride"]))
        fps = float(meta.get("fps", 25.0))
        self.tracker = CentroidTracker(max_missed=max(3, round(fps / stride)))
        self.frame_index = 0
        self.last_score = 0.0
        self.width = int(meta.get("width", 1))
        self.height = int(meta.get("height", 1))
        self.temporal = create_temporal_model(self.config)
        self.last_ttc_score = 0.0
        self.last_temporal_score = 0.0

    def step(self, frame: np.ndarray, t_sec: float) -> float:
        if self.detector is None or self.tracker is None:
            raise RuntimeError("reset(meta) must be called before step(frame, t_sec)")
        stride = max(1, int(self.config["risk_frame_stride"]))
        if self.temporal is not None:
            temporal_score = self.temporal.step(frame, t_sec)
            if temporal_score is not None:
                self.last_temporal_score = temporal_score
        if self.frame_index % stride == 0:
            detections = self.detector(frame)
            tracks = self.tracker.update(detections, t_sec, self.width, self.height)
            self.last_ttc_score = collision_risk(tracks)
        raw = fuse_risk(self.last_ttc_score, self.last_temporal_score)
        # Rise quickly for a real threat, decay slowly to avoid fragmented alarms.
        alpha = 0.65 if raw > self.last_score else 0.12
        self.last_score = alpha * raw + (1.0 - alpha) * self.last_score
        self.frame_index += 1
        return float(min(1.0, max(0.0, self.last_score)))


def analyze_video(video_path: str, config: dict[str, Any]) -> list[list]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = n_frames / fps if fps > 0 else 0.0
    detector = YoloDetector(config)
    analyzer = EventAnalyzer(config, fps, width, height)
    temporal = create_temporal_model(config)
    stride = max(1, int(config["frame_stride"]))
    frame_index = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if temporal is not None:
                temporal_score = temporal.step(frame, frame_index / fps)
                if temporal_score is not None:
                    analyzer.record_temporal_risk(frame_index / fps, temporal_score)
            if frame_index % stride == 0:
                analyzer.update(frame, detector(frame), frame_index / fps)
            frame_index += 1
    finally:
        cap.release()
    actual_duration = frame_index / fps if fps > 0 else duration
    return analyzer.events(actual_duration)
