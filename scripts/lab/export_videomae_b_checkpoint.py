"""Export a training checkpoint into the pure state dict used by inference."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[2]
SIMPLE_TAD = ROOT / "third_party" / "simple_tad"
sys.path.insert(0, str(ROOT / "training_compat"))
sys.path.insert(0, str(SIMPLE_TAD))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def extract_state_dict(payload: Any) -> dict[str, torch.Tensor]:
    if not isinstance(payload, dict):
        raise TypeError("checkpoint must contain a dictionary")
    for key in ("model", "state_dict"):
        nested = payload.get(key)
        if isinstance(nested, dict):
            payload = nested
            break
    state = {
        (key[7:] if key.startswith("module.") else key): value
        for key, value in payload.items()
        if isinstance(value, torch.Tensor)
    }
    if not state:
        raise ValueError("checkpoint does not contain tensor parameters")
    return state


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    source = args.checkpoint.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not source.is_file():
        raise SystemExit(f"checkpoint not found: {source}")
    if source == output:
        raise SystemExit("source and output paths must differ")

    payload = torch.load(source, map_location="cpu", weights_only=False)
    state = extract_state_dict(payload)

    from run_inference_simple import get_video_vit_base

    model = get_video_vit_base(with_flash=False)
    model.load_state_dict(state, strict=True)
    cpu_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}

    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cpu_state, output)
    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "architecture": "VideoMAE-B / vit_base_patch16_224",
        "num_classes": 2,
        "num_frames": 16,
        "view_fps": 10,
        "input_size": 224,
        "source": str(source),
        "source_sha256": sha256(source),
        "output": str(output),
        "output_sha256": sha256(output),
        "strict_load_verified": True,
    }
    metadata_path = output.with_suffix(output.suffix + ".metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
