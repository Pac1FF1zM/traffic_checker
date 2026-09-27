from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.demo import AdaptiveTemporalCalibrator, fuse_fixed_camera_risk


def test_steady_high_domain_bias_is_suppressed() -> None:
    calibrator = AdaptiveTemporalCalibrator(warmup_samples=3)
    calibrated = [calibrator.update(0.99) for _ in range(8)]
    assert max(calibrated) == 0.0


def test_large_change_after_normal_background_is_retained() -> None:
    calibrator = AdaptiveTemporalCalibrator(warmup_samples=3, margin=0.10, scale=0.35)
    for _ in range(6):
        calibrator.update(0.20)
    assert calibrator.update(0.95) > 0.99


def test_one_source_cannot_create_a_high_fixed_camera_alarm() -> None:
    assert fuse_fixed_camera_risk(1.0, 0.0) == 0.25
    assert fuse_fixed_camera_risk(0.0, 1.0) == 0.25
    assert fuse_fixed_camera_risk(1.0, 1.0) == 1.0


if __name__ == "__main__":
    test_steady_high_domain_bias_is_suppressed()
    test_large_change_after_normal_background_is_retained()
    test_one_source_cannot_create_a_high_fixed_camera_alarm()
    print("demo calibration tests: OK")
