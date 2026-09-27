"""Run the upstream trainer as a single Windows process without NCCL."""
from __future__ import annotations

import gc
import os
import runpy
import sys
from pathlib import Path

import torch.distributed as dist


ROOT = Path(__file__).resolve().parents[2]
UPSTREAM = ROOT / "third_party" / "simple_tad"
COMPAT = ROOT / "training_compat"
sys.path.insert(0, str(COMPAT))
sys.path.insert(0, str(UPSTREAM))

_barrier = dist.barrier


def safe_barrier(*args, **kwargs):
    if dist.is_available() and dist.is_initialized():
        return _barrier(*args, **kwargs)
    return None


dist.barrier = safe_barrier

# The upstream loop clears Python and CUDA caches before every micro-batch.
# PyTorch's caching allocator is faster with expandable segments; retain the
# conservative behavior as an opt-in fallback for tight VRAM cases.
if os.environ.get("TRAFFIC_DISABLE_PER_BATCH_CACHE_CLEAR") == "1":
    gc.collect = lambda *_args, **_kwargs: 0
    torch.cuda.empty_cache = lambda: None

runpy.run_path(str(UPSTREAM / "run_frame_finetuning.py"), run_name="__main__")
