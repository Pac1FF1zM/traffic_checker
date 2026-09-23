"""WIUT Hackathon baseline implementing the organizers' exact interface."""
from __future__ import annotations

import numpy as np

from src.baseline import OnlineRiskEstimator, analyze_video, load_config

CLASSES: list[str] = [
    "accident",
    "near_miss",
    "red_light",
    "wrong_way",
    "illegal_u_turn",
    "stopped_vehicle",
    "jaywalking",
    "failure_to_yield",
    "illegal_turn",
    "solid_line_crossing",
    "stop_line",
    "congestion",
    "road_obstacle",
    "fire_smoke",
]


def detect_events(video_path: str) -> list[list]:
    return analyze_video(video_path, load_config())


class RiskEstimator:
    """Causal accident-risk estimate; it never opens the video itself."""

    def __init__(self) -> None:
        self._impl = OnlineRiskEstimator(load_config())

    def reset(self, meta: dict) -> None:
        self._impl.reset(meta)

    def step(self, frame: np.ndarray, t_sec: float) -> float:
        return self._impl.step(frame, t_sec)
