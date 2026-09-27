"""Evaluate Simple-TAD on DADA's held-out test list, never the tuning validation list."""
from __future__ import annotations

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

import datasets_frame  # noqa: E402
from dada import FrameClsDataset_DADA  # noqa: E402


def build_heldout_dataset(is_train, test_mode, args):
    if is_train or not test_mode or not args.data_set.startswith("DADA2K"):
        raise RuntimeError("held-out wrapper may only construct the DADA test dataset")
    split = os.environ.get("DADA_HELDOUT_SPLIT", "DADA2K_my_split/test.txt")
    sampling_rate = args.sampling_rate_val if args.sampling_rate_val > 0 else args.sampling_rate
    dataset = FrameClsDataset_DADA(
        anno_path=split,
        data_path=args.data_path,
        mode="test",
        view_len=args.num_frames,
        view_step=sampling_rate,
        orig_fps=30,
        target_fps=args.view_fps,
        num_segment=1,
        test_num_segment=args.test_num_segment,
        test_num_crop=1,
        num_crop=1,
        keep_aspect_ratio=True,
        crop_size=args.input_size,
        short_side_size=args.short_side_size,
        args=args,
    )
    if args.nb_classes != 2:
        raise ValueError("held-out DADA evaluation expects two classes")
    return dataset, 2


datasets_frame.build_frame_dataset = build_heldout_dataset


# Upstream final_test has a single-process branch that references its output
# tensor before assignment. Present the Windows process as a world-size-one
# group and provide local collective shims, avoiding NCCL while preserving the
# exact predictions and metrics that a one-rank distributed run would produce.
def local_all_gather_object(output, value, *args, **kwargs):
    output[0] = value


def local_all_gather(output, value, *args, **kwargs):
    output[0].copy_(value)


dist.is_initialized = lambda: True
dist.get_rank = lambda *args, **kwargs: 0
dist.get_world_size = lambda *args, **kwargs: 1
dist.barrier = lambda *args, **kwargs: None
dist.all_reduce = lambda tensor, *args, **kwargs: tensor
dist.all_gather_object = local_all_gather_object
dist.all_gather = local_all_gather


runpy.run_path(str(UPSTREAM / "run_frame_finetuning.py"), run_name="__main__")
