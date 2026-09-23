# WIUT traffic-event baseline

A conservative, runnable baseline for the WIUT Hackathon 2026 Computer Vision
track. The organizers' `run_submission.py` and `evaluate.py` are kept unchanged.

The baseline combines:

- YOLO11n COCO road-user detection;
- deterministic IoU/centroid tracking;
- temporal rules for stopped vehicles, congestion and TTC near-misses;
- optional Simple-TAD DAPT VideoMAE-B risk recognition over the latest 16 frames;
- optional camera-calibrated rules for wrong-way driving, jaywalking,
  failure-to-yield, red-light running and solid-line crossing;
- a causal `RiskEstimator` fusing track TTC and temporal-model risk.

This is a starting point, not a trained final model. `illegal_u_turn`,
`illegal_turn`, `stop_line`, `road_obstacle` and `fire_smoke` deliberately produce
no events yet. The accident contact heuristic is disabled by default because box
overlap is too noisy on a distant CCTV view.

## Setup

Python 3.10+:

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
pip install -r requirements.txt
python scripts/download_weights.py
git submodule update --init --recursive
python scripts/download_temporal_weights.py
```

The temporal download is optional. If its source or checkpoint is absent, the
code emits a warning and automatically uses the YOLO+TTC baseline. Set
`temporal_required: true` after installation to make a missing/broken temporal
backend a hard error before submission.

Ultralytics uses AGPL-3.0 unless covered by an enterprise licence. Check that this
is acceptable for your submission. If not, replace `YoloDetector` with an
Apache-2.0 detector such as an RT-DETR implementation whose code and weights have
compatible terms.

Simple-TAD source and weights use CC BY-NC 4.0. Verify that the hackathon and
the intended use qualify before redistributing a submission bundle. The weights
are intentionally excluded from Git.

## Camera calibration

Copy `configs/scene.example.json` to `configs/scene.json`. Coordinates are
normalised to `[0, 1]`, so the same configuration works at any resolution.

- `road_polygon`: road surface, used for `jaywalking`.
- `crosswalk_polygon`: marked crossing, excluded from `jaywalking` and used for
  `failure_to_yield`.
- `traffic_light_roi`: rectangle `[x1, y1, x2, y2]` containing only the relevant
  signal head.
- `stop_line`: two endpoints; a tracked vehicle crossing it while the ROI is red
  creates `red_light`.
- `solid_lines`: list of line segments.
- `lanes`: polygons with an allowed image-plane direction vector, for example:

```json
{
  "lanes": [
    {
      "polygon": [[0.10, 0.95], [0.42, 0.95], [0.52, 0.35], [0.43, 0.35]],
      "direction": [0.0, -1.0]
    }
  ]
}
```

Set `WIUT_SCENE_CONFIG` to use a different JSON file. Empty geometry fields disable
the corresponding scene-specific rules rather than guessing.

## Run

Part A only, useful while calibrating geometry:

```bash
python run_submission.py --videos samples --out predictions_samples.json --team TEAM --no-risk
python evaluate.py --pred predictions_samples.json --validate-only
```

Full Part A + causal Part B:

```bash
python run_submission.py --videos samples --out predictions_samples.json --team TEAM
```

With team-created labels:

```bash
python evaluate.py --pred predictions_samples.json --gt my_labels.json --per-video
```

## Tests

The tests do not require video or model inference:

```bash
python tests/test_baseline.py
python -m py_compile solution.py src/*.py run_submission.py evaluate.py
```

## What to tune first

1. Label every supplied WIUT clip using the official boundary conventions.
2. Calibrate scene polygons and lines from the fixed camera.
3. Plot track histories and tune speeds in normalised image units per second.
4. Disable rules that produce false positives; macro F1 punishes speculative
   classes.
5. Train a temporal accident/near-miss classifier using the shortlist in
   [DATASETS.md](DATASETS.md) and the staged plan in
   [FINETUNING.md](FINETUNING.md).
6. Measure runtime with the official runner. Part A and Part B share the
   `3 x video duration` budget.

## Repository layout

```text
solution.py                 official interface
src/baseline.py             detector, event rules and causal risk
src/temporal.py             causal Simple-TAD adapter and TTC/model fusion
src/tracking.py             deterministic lightweight tracker
src/geometry.py             geometry and segment post-processing
configs/scene.example.json  camera calibration template
scripts/download_weights.py one-time model download
scripts/download_temporal_weights.py  Simple-TAD checkpoint download
DATASETS.md                 researched dataset shortlist
FINETUNING.md               staged fine-tuning plan for a T4 GPU
third_party/simple_tad      pinned upstream source (git submodule)
run_submission.py           organizer file, unchanged
evaluate.py                 organizer file, unchanged
```
