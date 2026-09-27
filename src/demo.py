"""Single-pass video analysis used by the local live-demo API."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import cv2

from .baseline import EventAnalyzer, YoloDetector, collision_risk
from .temporal import create_temporal_model, fuse_risk


class AdaptiveTemporalCalibrator:
    """Turn a scene-biased classifier score into a causal change score.

    A dashcam-trained classifier can sit near one for an entire fixed-camera
    clip.  The slowly adapting baseline treats that steady value as the scene's
    normal appearance and only surfaces a sufficiently large upward change.
    """

    def __init__(
        self,
        *,
        warmup_samples: int = 4,
        margin: float = 0.10,
        scale: float = 0.35,
        alpha_up: float = 0.02,
        alpha_down: float = 0.12,
    ) -> None:
        self.warmup_samples = max(1, int(warmup_samples))
        self.margin = max(0.0, float(margin))
        self.scale = max(1e-6, float(scale))
        self.alpha_up = min(1.0, max(0.0, float(alpha_up)))
        self.alpha_down = min(1.0, max(0.0, float(alpha_down)))
        self.baseline: float | None = None
        self.samples = 0

    def update(self, score: float) -> float:
        score = min(1.0, max(0.0, float(score)))
        if self.baseline is None:
            self.baseline = score
            self.samples = 1
            return 0.0

        excess = max(0.0, score - self.baseline - self.margin)
        calibrated = min(1.0, excess / self.scale)
        if self.samples < self.warmup_samples:
            calibrated = 0.0

        alpha = self.alpha_up if score > self.baseline else self.alpha_down
        self.baseline += alpha * (score - self.baseline)
        self.samples += 1
        return calibrated


def fuse_fixed_camera_risk(ttc_score: float, temporal_change: float) -> float:
    """Require cross-source agreement while keeping each source visible.

    A single unsupported signal is capped at a non-alarm level.  Agreement
    between changing appearance and converging trajectories can still produce
    a high score.
    """

    ttc = min(1.0, max(0.0, float(ttc_score)))
    temporal = min(1.0, max(0.0, float(temporal_change)))
    return float(max(0.25 * ttc, 0.25 * temporal, ttc * temporal))


def analyze_video_detailed(video_path: str | Path, config: dict[str, Any]) -> dict[str, Any]:
    """Return events and a causal risk timeline without processing the video twice."""
    source = str(video_path)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError("The uploaded file could not be opened as a video.")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    frame_count_hint = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    detector = YoloDetector(config)
    analyzer = EventAnalyzer(config, fps, width, height)
    temporal = create_temporal_model(config)
    fixed_camera_mode = bool(config.get("demo_fixed_camera_mode", False))
    calibrator = (
        AdaptiveTemporalCalibrator(
            warmup_samples=int(config.get("fixed_camera_warmup_samples", 4)),
            margin=float(config.get("fixed_camera_temporal_margin", 0.10)),
            scale=float(config.get("fixed_camera_temporal_scale", 0.35)),
        )
        if fixed_camera_mode
        else None
    )
    detection_stride = max(1, int(config["frame_stride"]))
    timeline_period = max(0.25, float(config.get("demo_timeline_period", 0.5)))

    frame_index = 0
    next_timeline_t = 0.0
    last_ttc_score = 0.0
    last_raw_temporal_score = 0.0
    last_temporal_score = 0.0
    smoothed_risk = 0.0
    timeline: list[dict[str, float]] = []
    started = time.perf_counter()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t_sec = frame_index / max(fps, 1e-6)

            if temporal is not None:
                temporal_score = temporal.step(frame, t_sec)
                if temporal_score is not None:
                    last_raw_temporal_score = temporal_score
                    last_temporal_score = (
                        calibrator.update(temporal_score) if calibrator is not None else temporal_score
                    )
                    if not fixed_camera_mode:
                        analyzer.record_temporal_risk(t_sec, last_temporal_score)

            if frame_index % detection_stride == 0:
                detections = detector(frame)
                analyzer.update(frame, detections, t_sec)
                active_tracks = [track for track in analyzer.tracker.tracks.values() if track.missed == 0]
                last_ttc_score = (
                    collision_risk(
                        active_tracks,
                        horizon=float(config.get("fixed_camera_ttc_horizon", 1.75)),
                        miss_threshold=float(config.get("fixed_camera_ttc_miss", 0.04)),
                        min_history=int(config.get("fixed_camera_ttc_min_history", 6)),
                        min_track_age=float(config.get("fixed_camera_ttc_min_age", 0.6)),
                        max_instant_speed=float(config.get("fixed_camera_ttc_max_speed", 0.45)),
                        ttc_scale=float(config.get("fixed_camera_ttc_scale", 1.0)),
                        proximity_scale=float(config.get("fixed_camera_ttc_proximity_scale", 0.018)),
                    )
                    if fixed_camera_mode
                    else collision_risk(active_tracks)
                )

            raw_risk = (
                fuse_fixed_camera_risk(last_ttc_score, last_temporal_score)
                if fixed_camera_mode
                else fuse_risk(last_ttc_score, last_temporal_score)
            )
            if fixed_camera_mode:
                analyzer.record_temporal_risk(t_sec, raw_risk)
            alpha = 0.65 if raw_risk > smoothed_risk else 0.12
            smoothed_risk = alpha * raw_risk + (1.0 - alpha) * smoothed_risk
            if t_sec + 1e-6 >= next_timeline_t:
                timeline.append(
                    {
                        "time": round(t_sec, 3),
                        "risk": round(smoothed_risk, 6),
                        "visual_risk": round(last_temporal_score, 6),
                        "raw_visual_score": round(last_raw_temporal_score, 6),
                        "ttc_risk": round(last_ttc_score, 6),
                    }
                )
                next_timeline_t += timeline_period
            frame_index += 1
    finally:
        cap.release()

    duration = frame_index / max(fps, 1e-6)
    elapsed = time.perf_counter() - started
    events = [
        {"start": start, "end": end, "label": label}
        for start, end, label in analyzer.events(duration)
    ]
    max_point = max(timeline, key=lambda point: point["risk"], default={"risk": 0.0, "time": 0.0})
    average_risk = sum(point["risk"] for point in timeline) / max(len(timeline), 1)
    max_ttc_risk = max((point["ttc_risk"] for point in timeline), default=0.0)
    max_visual_change = max((point["visual_risk"] for point in timeline), default=0.0)
    max_raw_visual_score = max((point["raw_visual_score"] for point in timeline), default=0.0)

    return {
        "video": {
            "duration": round(duration, 3),
            "fps": round(fps, 3),
            "frames": frame_index,
            "reported_frames": frame_count_hint,
            "width": width,
            "height": height,
        },
        "summary": {
            "max_risk": round(float(max_point["risk"]), 6),
            "max_risk_time": round(float(max_point["time"]), 3),
            "average_risk": round(average_risk, 6),
            "max_ttc_risk": round(max_ttc_risk, 6),
            "max_visual_change": round(max_visual_change, 6),
            "max_raw_visual_score": round(max_raw_visual_score, 6),
            "event_count": len(events),
            "processing_seconds": round(elapsed, 3),
            "realtime_factor": round(elapsed / max(duration, 1e-6), 3),
        },
        "events": events,
        "timeline": timeline,
        "model": {
            "name": (
                "VideoMAE-S + YOLO11n/TTC (fixed-camera calibrated)"
                if fixed_camera_mode
                else "VideoMAE-S + YOLO11n/TTC"
            ),
            "causal": True,
            "fixed_camera_calibration": fixed_camera_mode,
            "threshold": float(config["temporal_accident_threshold"]),
        },
    }
