"""Run the locked WIUT comparison without viewing or adapting to sample frames.

Run this from the DirectML environment on the current Windows laptop:

    .venv\\dml\\Scripts\\python scripts\\run_blind_model_comparison.py \
        --videos data\\wiut_blind\\*.mp4 --out tmp\\wiut_blind_results.json

The script intentionally saves raw diagnostics only. It never creates labels,
chooses thresholds, renders frames, or treats X3D's Kinetics head as accident
risk.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.temporal import TemporalRiskModel


PROTOCOL = ROOT / "experiments" / "wiut_blind_protocol.json"
SPECS = {
    "videomae_small": ROOT / "weights" / "simpletad_ft-dota_dapt-vm1-s_auroc.pth",
    "videomae_base": ROOT / "weights" / "simpletad_ft-dota_dapt-vm1-b_auroc.pth",
}


class TubeletConv2d(torch.nn.Module):
    """Exact Conv3D(kT=2,sT=2) replacement for the DirectML backend."""

    def __init__(self, conv: torch.nn.Conv3d):
        super().__init__()
        out_channels, in_channels, kernel_t, kernel_h, kernel_w = conv.weight.shape
        self.kernel_t = kernel_t
        self.stride_t = conv.stride[0]
        self.weight = torch.nn.Parameter(
            conv.weight.detach().reshape(
                out_channels, in_channels * kernel_t, kernel_h, kernel_w
            ),
            requires_grad=False,
        )
        self.bias = (
            None
            if conv.bias is None
            else torch.nn.Parameter(conv.bias.detach(), requires_grad=False)
        )
        self.stride_hw = conv.stride[1:]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, _, height, width = x.shape
        clips = x.unfold(2, self.kernel_t, self.stride_t)
        count = clips.shape[2]
        clips = clips.permute(0, 2, 1, 5, 3, 4).reshape(
            batch * count, channels * self.kernel_t, height, width
        )
        out = F.conv2d(clips, self.weight, self.bias, stride=self.stride_hw)
        return out.reshape(
            batch, count, out.shape[1], out.shape[2], out.shape[3]
        ).permute(0, 2, 1, 3, 4)


def sha256(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk_bytes):
            digest.update(block)
    return digest.hexdigest()


def metadata(path: Path) -> dict[str, Any]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video: {path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.release()
    if fps <= 0 or frame_count <= 0:
        raise RuntimeError(f"invalid video metadata: {path}")
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
        "fps": fps,
        "frame_count": frame_count,
        "duration_sec": frame_count / fps,
        "width": width,
        "height": height,
    }


def window_starts(duration: float, window: float, count: int) -> list[float]:
    if duration <= window or count <= 1:
        return [0.0]
    last = duration - window
    candidates = np.linspace(0.0, last, count)
    return sorted({round(float(value), 3) for value in candidates})


def sampled_frames(
    path: Path, start_sec: float, duration_sec: float, sample_fps: float
) -> Iterable[tuple[float, np.ndarray]]:
    """Decode sequentially and yield causal samples without rendering frames."""
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video: {path}")
    capture.set(cv2.CAP_PROP_POS_MSEC, start_sec * 1000.0)
    end_sec = start_sec + duration_sec
    next_sample = start_sec
    step = 1.0 / sample_fps
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            current = float(capture.get(cv2.CAP_PROP_POS_MSEC)) / 1000.0
            if current > end_sec + 1e-6:
                break
            if current + 1e-6 < next_sample:
                continue
            yield current, frame
            while next_sample <= current + 1e-6:
                next_sample += step
    finally:
        capture.release()


def dml_device() -> tuple[torch.device, bool]:
    if torch.cuda.is_available():
        return torch.device("cuda:0"), False
    try:
        import torch_directml

        return torch_directml.device(), True
    except (ImportError, RuntimeError):
        return torch.device("cpu"), False


def temporal_config(arch: str, device: torch.device) -> dict[str, Any]:
    return {
        "temporal_enabled": True,
        "temporal_required": True,
        "temporal_arch": arch,
        "temporal_checkpoint_path": str(SPECS[arch]),
        "temporal_source_path": str(
            ROOT / "third_party" / "simple_tad" / "run_inference_simple.py"
        ),
        "temporal_device": str(device),
        "temporal_window_frames": 16,
        "temporal_view_fps": 10.0,
        "temporal_inference_stride": 4,
    }


def run_temporal(
    arch: str,
    videos: list[tuple[Path, dict[str, Any], list[float]]],
    window_seconds: float,
) -> dict[str, Any]:
    device, is_dml = dml_device()
    TemporalRiskModel._MODEL_CACHE.clear()
    load_started = time.perf_counter()
    model = TemporalRiskModel(temporal_config(arch, device))
    if is_dml:
        model.model.patch_embed.proj = TubeletConv2d(model.model.patch_embed.proj)
    load_seconds = time.perf_counter() - load_started
    windows: list[dict[str, Any]] = []
    inference_count = 0
    processed_seconds = 0.0
    run_started = time.perf_counter()
    for path, info, starts in videos:
        for start in starts:
            actual_duration = min(window_seconds, info["duration_sec"] - start)
            model.reset()
            samples: list[dict[str, float]] = []
            for timestamp, frame in sampled_frames(path, start, actual_duration, 10.0):
                score = model.step(frame, timestamp - start)
                if score is not None:
                    samples.append(
                        {"t_sec": round(timestamp, 6), "risk": round(score, 8)}
                    )
                    inference_count += 1
            windows.append(
                {
                    "video": path.name,
                    "start_sec": start,
                    "duration_sec": actual_duration,
                    "samples": samples,
                }
            )
            processed_seconds += actual_duration
    wall_seconds = time.perf_counter() - run_started
    scores = [s["risk"] for w in windows for s in w["samples"]]
    result = {
        "arch": arch,
        "device": str(device),
        "checkpoint": str(SPECS[arch].relative_to(ROOT)),
        "load_seconds": load_seconds,
        "wall_seconds": wall_seconds,
        "processed_video_seconds": processed_seconds,
        "wall_per_video_second": wall_seconds / processed_seconds,
        "inference_count": inference_count,
        "risk_summary": {},
        "windows": windows,
    }
    if scores:
        array = np.asarray(scores, dtype=np.float64)
        result["risk_summary"] = {
            "min": float(array.min()),
            "p25": float(np.percentile(array, 25)),
            "median": float(np.median(array)),
            "p75": float(np.percentile(array, 75)),
            "max": float(array.max()),
            "mean": float(array.mean()),
            "std": float(array.std()),
        }
    del model
    TemporalRiskModel._MODEL_CACHE.clear()
    gc.collect()
    return result


def x3d_frame(frame: np.ndarray) -> torch.Tensor:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    height, width = rgb.shape[:2]
    scale = 182.0 / min(height, width)
    resized = cv2.resize(
        rgb,
        (round(width * scale), round(height * scale)),
        interpolation=cv2.INTER_LINEAR,
    )
    height, width = resized.shape[:2]
    top = (height - 182) // 2
    left = (width - 182) // 2
    crop = resized[top : top + 182, left : left + 182]
    tensor = torch.from_numpy(crop.copy()).permute(2, 0, 1).float().div_(255.0)
    return tensor.sub_(0.45).div_(0.225)


def run_x3d(
    videos: list[tuple[Path, dict[str, Any], list[float]]], window_seconds: float
) -> dict[str, Any]:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    load_started = time.perf_counter()
    model = torch.hub.load(
        "facebookresearch/pytorchvideo",
        "x3d_s",
        pretrained=True,
        trust_repo=True,
    ).to(device).eval()
    load_seconds = time.perf_counter() - load_started
    windows: list[dict[str, Any]] = []
    processed_seconds = 0.0
    inference_count = 0
    run_started = time.perf_counter()
    with torch.no_grad():
        for path, info, starts in videos:
            for start in starts:
                actual_duration = min(window_seconds, info["duration_sec"] - start)
                frames: deque[torch.Tensor] = deque(maxlen=13)
                samples: list[dict[str, Any]] = []
                sampled_index = 0
                for timestamp, frame in sampled_frames(path, start, actual_duration, 5.0):
                    frames.append(x3d_frame(frame))
                    if len(frames) < 13 or (sampled_index - 12) % 2:
                        sampled_index += 1
                        continue
                    clip = torch.stack(tuple(frames), dim=1).unsqueeze(0).to(device)
                    output = model(clip)[0]
                    probabilities = (
                        output
                        if abs(float(output.sum()) - 1.0) < 1e-3
                        else torch.softmax(output, dim=0)
                    )
                    clipped = probabilities.clamp_min(1e-12)
                    entropy = float(-(clipped * clipped.log()).sum())
                    top_probability, top_index = probabilities.max(dim=0)
                    samples.append(
                        {
                            "t_sec": round(timestamp, 6),
                            "top1_index": int(top_index),
                            "top1_probability": round(float(top_probability), 8),
                            "entropy": round(entropy, 8),
                        }
                    )
                    inference_count += 1
                    sampled_index += 1
                windows.append(
                    {
                        "video": path.name,
                        "start_sec": start,
                        "duration_sec": actual_duration,
                        "samples": samples,
                    }
                )
                processed_seconds += actual_duration
    wall_seconds = time.perf_counter() - run_started
    return {
        "arch": "x3d_s",
        "task": "Kinetics-400 diagnostics only; not accident risk",
        "device": str(device),
        "load_seconds": load_seconds,
        "wall_seconds": wall_seconds,
        "processed_video_seconds": processed_seconds,
        "wall_per_video_second": wall_seconds / processed_seconds,
        "inference_count": inference_count,
        "windows": windows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--videos", nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--window-seconds", type=float, default=60.0)
    parser.add_argument("--windows-per-video", type=int, default=3)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=("videomae_small", "videomae_base", "x3d_s"),
        default=("videomae_small", "videomae_base", "x3d_s"),
    )
    args = parser.parse_args()
    if args.window_seconds <= 0 or args.windows_per_video <= 0:
        raise SystemExit("window duration and count must be positive")

    paths = [Path(item).resolve() for item in args.videos]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise SystemExit(f"missing videos: {missing}")
    if not PROTOCOL.is_file():
        raise SystemExit(f"locked protocol not found: {PROTOCOL}")

    prepared: list[tuple[Path, dict[str, Any], list[float]]] = []
    for path in paths:
        info = metadata(path)
        starts = window_starts(
            info["duration_sec"], args.window_seconds, args.windows_per_video
        )
        prepared.append((path, info, starts))
        print(
            f"metadata: {path.name} duration={info['duration_sec']:.2f}s "
            f"windows={starts}",
            flush=True,
        )

    output: dict[str, Any] = {
        "protocol": json.loads(PROTOCOL.read_text(encoding="utf-8")),
        "videos": [info for _, info, _ in prepared],
        "results": {},
    }
    for model_name in args.models:
        print(f"running: {model_name}", flush=True)
        if model_name == "x3d_s":
            result = run_x3d(prepared, args.window_seconds)
        else:
            result = run_temporal(model_name, prepared, args.window_seconds)
        output["results"][model_name] = result
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(output, indent=2), encoding="utf-8")
        print(
            f"finished: {model_name} wall={result['wall_seconds']:.2f}s "
            f"factor={result['wall_per_video_second']:.3f}x",
            flush=True,
        )

    print(f"saved: {args.out.resolve()}")


if __name__ == "__main__":
    main()
