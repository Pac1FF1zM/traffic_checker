"""Optional causal Simple-TAD temporal risk backend.

The upstream model consumes the latest 16 frames and returns probabilities for
normal/risky traffic. Imports are lazy so the detector baseline still works
when the optional submodule or checkpoint is absent.
"""
from __future__ import annotations

import importlib.util
import sys
import warnings
from collections import deque
from pathlib import Path
from typing import Any

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGENET_MEAN = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)[:, None, None]
IMAGENET_STD = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)[:, None, None]


def fuse_risk(ttc_score: float, temporal_score: float) -> float:
    """Probabilistic OR: either geometry or appearance can raise the risk."""
    left = min(1.0, max(0.0, float(ttc_score)))
    right = min(1.0, max(0.0, float(temporal_score)))
    return float(1.0 - (1.0 - left) * (1.0 - right))


def prepare_frame(frame: np.ndarray, size: int = 224) -> np.ndarray:
    """Match the preprocessing used by Simple-TAD's official inference code."""
    resized = cv2.resize(frame, (size, size), interpolation=cv2.INTER_CUBIC)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    chw = rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
    return np.ascontiguousarray((chw - IMAGENET_MEAN) / IMAGENET_STD)


class TemporalRiskModel:
    """Sliding-window Simple-TAD model that only sees current/past frames."""

    _MODEL_CACHE: dict[tuple[str, str, str], Any] = {}

    def __init__(self, config: dict[str, Any]):
        self.window = int(config.get("temporal_window_frames", 16))
        if self.window != 16:
            raise ValueError("the released Simple-TAD checkpoint requires temporal_window_frames=16")
        self.stride = max(1, int(config.get("temporal_inference_stride", 4)))
        self.view_fps = float(config.get("temporal_view_fps", 10.0))
        if self.view_fps <= 0:
            raise ValueError("temporal_view_fps must be positive")
        self.arch = str(config.get("temporal_arch", "videomae_base")).lower()
        if self.arch not in {"videomae_small", "videomae_base"}:
            raise ValueError(
                "temporal_arch must be 'videomae_small' or 'videomae_base'"
            )

        checkpoint = Path(str(config["temporal_checkpoint_path"]))
        source = Path(str(config["temporal_source_path"]))
        self.checkpoint = checkpoint if checkpoint.is_absolute() else PROJECT_ROOT / checkpoint
        self.source = source if source.is_absolute() else PROJECT_ROOT / source
        if not self.checkpoint.is_file():
            raise FileNotFoundError(
                f"Simple-TAD checkpoint not found at {self.checkpoint}; "
                "run `python scripts/download_temporal_weights.py`"
            )
        if not self.source.is_file():
            raise FileNotFoundError(
                f"Simple-TAD source not found at {self.source}; "
                "run `git submodule update --init --recursive`"
            )

        import torch

        requested = str(config.get("temporal_device", config.get("device", "auto")))
        if requested == "auto":
            requested = "cuda:0" if torch.cuda.is_available() else "cpu"
        elif requested.isdigit():
            requested = f"cuda:{requested}"
        self.device = torch.device(requested)
        cache_key = (
            str(self.source.resolve()),
            str(self.checkpoint.resolve()),
            f"{self.device}:{self.arch}",
        )
        if cache_key not in self._MODEL_CACHE:
            self._MODEL_CACHE[cache_key] = self._load_model(torch)
        self.model = self._MODEL_CACHE[cache_key]
        self.frames: deque[np.ndarray] = deque(maxlen=self.window)
        self.frame_index = 0
        self.last_score = 0.0
        self.last_sample_t: float | None = None

    def _load_model(self, torch: Any) -> Any:
        module_name = "_traffic_checker_simple_tad_inference"
        module = sys.modules.get(module_name)
        if module is None:
            spec = importlib.util.spec_from_file_location(module_name, self.source)
            if spec is None or spec.loader is None:
                raise RuntimeError(f"cannot import Simple-TAD from {self.source}")
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            previous_bytecode_setting = sys.dont_write_bytecode
            sys.dont_write_bytecode = True
            try:
                spec.loader.exec_module(module)
            finally:
                sys.dont_write_bytecode = previous_bytecode_setting

        factory_name = {
            "videomae_small": "get_video_vit_small",
            "videomae_base": "get_video_vit_base",
        }[self.arch]
        model = getattr(module, factory_name)(with_flash=False)
        try:
            state = torch.load(self.checkpoint, map_location="cpu", weights_only=True)
        except TypeError:  # torch < 2.0
            state = torch.load(self.checkpoint, map_location="cpu")
        for key in ("model", "state_dict", "module"):
            if isinstance(state, dict) and key in state and isinstance(state[key], dict):
                state = state[key]
                break
        if not isinstance(state, dict):
            raise RuntimeError("unsupported Simple-TAD checkpoint format")
        state = {str(k).removeprefix("module."): v for k, v in state.items()}
        model.load_state_dict(state, strict=True)
        return model.to(self.device).eval()

    def reset(self) -> None:
        self.frames.clear()
        self.frame_index = 0
        self.last_score = 0.0
        self.last_sample_t = None

    def step(self, frame: np.ndarray, t_sec: float | None = None) -> float | None:
        """Return a new score when inference runs, otherwise ``None``."""
        if t_sec is not None and self.last_sample_t is not None:
            if t_sec - self.last_sample_t < (1.0 / self.view_fps) - 1e-6:
                return None
        if t_sec is not None:
            self.last_sample_t = t_sec
        self.frames.append(prepare_frame(frame))
        current = self.frame_index
        self.frame_index += 1
        if len(self.frames) < self.window or (current - (self.window - 1)) % self.stride:
            return None

        import torch

        clip = np.stack(tuple(self.frames), axis=1)  # C, T, H, W
        tensor = torch.from_numpy(clip).unsqueeze(0).to(self.device, non_blocking=True)
        # DirectML cannot create some view version counters for inference
        # tensors; no_grad remains read-only while working on that backend.
        with torch.no_grad():
            output = self.model(tensor)
        scores = output[0]
        looks_like_probability = bool(
            torch.all(scores >= 0).item()
            and torch.all(scores <= 1).item()
            and abs(float(scores.sum().detach().cpu().item()) - 1.0) <= 1e-3
        )
        probabilities = scores if looks_like_probability else torch.softmax(scores, dim=0)
        self.last_score = float(probabilities[1].detach().cpu().item())
        return self.last_score


def create_temporal_model(config: dict[str, Any]) -> TemporalRiskModel | None:
    """Build the optional model, or warn and retain the detector baseline."""
    if not bool(config.get("temporal_enabled", False)):
        return None
    try:
        return TemporalRiskModel(config)
    except (FileNotFoundError, ImportError, ModuleNotFoundError, RuntimeError) as exc:
        if bool(config.get("temporal_required", False)):
            raise
        warnings.warn(f"temporal backend disabled: {exc}", RuntimeWarning, stacklevel=2)
        return None
