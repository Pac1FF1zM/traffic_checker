"""Local GPU API and static-site host for the WIUT traffic live demo."""
from __future__ import annotations

import os
import tempfile
import threading
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from src.baseline import load_config
from src.demo import analyze_video_detailed


ROOT = Path(__file__).resolve().parent
DIST = ROOT / "website" / "dist"
MAX_UPLOAD_BYTES = int(os.getenv("WIUT_DEMO_MAX_UPLOAD_MB", "300")) * 1024 * 1024
ALLOWED_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
ANALYSIS_LOCK = threading.Lock()

app = FastAPI(title="WIUT Traffic Vision Demo", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


def demo_config() -> dict:
    config = load_config()
    checkpoint = os.getenv("WIUT_DEMO_CHECKPOINT", "").strip()
    if checkpoint:
        config["temporal_checkpoint_path"] = checkpoint
    config["temporal_arch"] = "videomae_small"
    config["temporal_enabled"] = True
    config["temporal_required"] = True
    # The deployment target is a stationary road camera, while the available
    # VideoMAE checkpoint was trained on dashcam clips.  Use conservative
    # causal calibration and require temporal/TTC agreement for a high alarm.
    config["demo_fixed_camera_mode"] = True
    config["fixed_camera_warmup_samples"] = 4
    config["fixed_camera_temporal_margin"] = 0.15
    config["fixed_camera_temporal_scale"] = 0.35
    config["temporal_accident_threshold"] = 0.65
    config["temporal_accident_min_sec"] = 1.0
    config["enable_near_miss_heuristic"] = False
    config["tracker_min_iou"] = 0.10
    config["tracker_max_distance"] = 0.045
    config["fixed_camera_ttc_horizon"] = 1.75
    config["fixed_camera_ttc_miss"] = 0.04
    config["fixed_camera_ttc_min_history"] = 6
    config["fixed_camera_ttc_min_age"] = 0.6
    config["fixed_camera_ttc_max_speed"] = 0.45
    config["fixed_camera_ttc_scale"] = 1.0
    config["fixed_camera_ttc_proximity_scale"] = 0.018
    return config


def checkpoint_status() -> tuple[bool, str]:
    configured = Path(str(demo_config()["temporal_checkpoint_path"]))
    resolved = configured if configured.is_absolute() else ROOT / configured
    return resolved.is_file(), str(resolved)


@app.get("/api/health")
def health() -> dict:
    ready, checkpoint = checkpoint_status()
    try:
        import torch

        cuda = bool(torch.cuda.is_available())
        device = torch.cuda.get_device_name(0) if cuda else "CPU"
    except ImportError:
        cuda = False
        device = "PyTorch unavailable"
    return {"status": "ready" if ready else "checkpoint_missing", "checkpoint": checkpoint, "cuda": cuda, "device": device}


@app.post("/api/analyze")
def analyze(file: UploadFile = File(...)) -> dict:
    ready, checkpoint = checkpoint_status()
    if not ready:
        raise HTTPException(503, f"Final checkpoint is not available at {checkpoint}")
    suffix = Path(file.filename or "video.mp4").suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(415, "Upload MP4, AVI, MOV, MKV, or WebM video.")
    if not ANALYSIS_LOCK.acquire(blocking=False):
        raise HTTPException(409, "The GPU is already analyzing another video. Try again shortly.")

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(prefix="wiut-demo-", suffix=suffix, delete=False) as temporary:
            temporary_path = Path(temporary.name)
            copied = 0
            while chunk := file.file.read(1024 * 1024):
                copied += len(chunk)
                if copied > MAX_UPLOAD_BYTES:
                    raise HTTPException(413, "Video exceeds the configured upload limit.")
                temporary.write(chunk)
        result = analyze_video_detailed(temporary_path, demo_config())
        result["filename"] = Path(file.filename or "video").name
        return result
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(500, f"Analysis failed: {exc}") from exc
    finally:
        file.file.close()
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        ANALYSIS_LOCK.release()


if DIST.is_dir():
    app.mount("/", StaticFiles(directory=DIST, html=True), name="website")
