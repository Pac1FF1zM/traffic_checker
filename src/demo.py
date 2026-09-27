"""Single-pass video analysis used by the local live-demo API."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import cv2

from .baseline import EventAnalyzer, YoloDetector, collision_risk
from .temporal import create_temporal_model, fuse_risk


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
    detection_stride = max(1, int(config["frame_stride"]))
    timeline_period = max(0.25, float(config.get("demo_timeline_period", 0.5)))

    frame_index = 0
    next_timeline_t = 0.0
    last_ttc_score = 0.0
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
                    last_temporal_score = temporal_score
                    analyzer.record_temporal_risk(t_sec, temporal_score)

            if frame_index % detection_stride == 0:
                detections = detector(frame)
                analyzer.update(frame, detections, t_sec)
                active_tracks = [track for track in analyzer.tracker.tracks.values() if track.missed == 0]
                last_ttc_score = collision_risk(active_tracks)

            raw_risk = fuse_risk(last_ttc_score, last_temporal_score)
            alpha = 0.65 if raw_risk > smoothed_risk else 0.12
            smoothed_risk = alpha * raw_risk + (1.0 - alpha) * smoothed_risk
            if t_sec + 1e-6 >= next_timeline_t:
                timeline.append(
                    {
                        "time": round(t_sec, 3),
                        "risk": round(smoothed_risk, 6),
                        "visual_risk": round(last_temporal_score, 6),
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
            "event_count": len(events),
            "processing_seconds": round(elapsed, 3),
            "realtime_factor": round(elapsed / max(duration, 1e-6), 3),
        },
        "events": events,
        "timeline": timeline,
        "model": {
            "name": "VideoMAE-S + YOLO11n/TTC",
            "causal": True,
            "threshold": float(config["temporal_accident_threshold"]),
        },
    }
