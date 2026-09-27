# WIUT Hackathon 2026 — official requirement snapshot

Source: the authenticated Computer Vision elimination task at
`https://hackathon.wiut.uz/team/task/`, checked 27 September 2026 (Tashkent).
This file is a working compliance summary; the organizer page remains the
authority.

## Deadline and submission

- Deadline: **Sunday, 27 September 2026, 23:59 Tashkent time**.
- Submit one public Git repository URL, one exact tag or 40-character commit
  hash, one public team website URL, all three T-shirt sizes, and an optional
  reviewer note.
- The tagged commit at the deadline is evaluated; later commits are ignored.
- One submission per three-person team.

## Deliverables

1. A public repository containing code, weights and a one-command entry point.
2. A public website containing the team, approach, sample-video EDA,
   visualizations, results, live upload demo and short technical report.
3. A short technical report; a website section is acceptable.

Required repository shape:

```text
solution.py
run_submission.py             # unchanged organizer file
evaluate.py                   # unchanged organizer file
requirements.txt              # or Dockerfile
weights/                      # weights or download.sh; total <= 5 GB
src/
predictions_samples.json
README.md
```

The clean-machine commands are exactly:

```bash
pip install -r requirements.txt
python run_submission.py --videos /data/test --out predictions.json
```

`weights/download.sh` may be run once with internet before evaluation. The
actual evaluation run is offline.

## Model interface

Part A is mandatory. `detect_events(video_path)` may use random access,
multiple passes or clip sampling and returns:

```python
[[start_sec, end_sec, label], ...]
```

Boundaries must satisfy `0 <= start < end <= duration`; same-class segments may
not overlap. Temporal boundary precision matters because matching uses tIoU.

Part B is bonus. `RiskEstimator.reset(meta)` is called once and
`step(frame, t_sec)` is called for every BGR `uint8` frame in order. It must
return a float in `[0, 1]`: the probability that an `accident` starts in the
next five seconds. It may skip inference internally and repeat the last score,
but it must not open the video, read future frames or reuse non-causal Part A
output.

The 14 exact labels are:

```text
accident, near_miss, red_light, wrong_way, illegal_u_turn,
stopped_vehicle, jaywalking, failure_to_yield, illegal_turn,
solid_line_crossing, stop_line, congestion, road_obstacle, fire_smoke
```

## Data and rules

- Four unlabeled sample videos are provided for EDA, team annotation,
  development and visualizations. They are not the final test.
- Hidden videos use the same fixed CCTV camera, angle, resolution and frame
  rate, but may contain events absent from the samples.
- Public datasets and team-created annotations of the samples are allowed.
- Do not obtain more footage from the same camera by scraping or other means.
- Open weights only. No paid/hosted model APIs at inference.
- Every external dataset and its licence must be listed in README.
- Fix seeds. Two runs on the same machine must agree up to floating-point noise.
- Open-source reuse is allowed with attribution; shared/copied team solutions
  are not.

## Official scoring

Part A:

- greedy one-to-one segment matching at tIoU 0.3, 0.5 and 0.7;
- pooled TP/FP/FN per class;
- `Score_A` is macro mean F1 over present/predicted classes and thresholds;
- predicting a class absent from ground truth adds that class with score 0.

Part B (`accident` only):

- positive horizon `H = 5 s`;
- alarm matching window `W = 10 s`;
- threshold `theta = 0.5`; alarm runs separated by less than 2 s are merged;
- `Score_B = 0.4*AP + 0.4*F1_alarm + 0.2*(mTTA/W)`;
- AP is chance-normalized; accident and near-miss intervals are ignored as
  defined in the organizer's `evaluate.py`.

Combined:

```text
M = 0.7*Score_A + 0.3*Score_B
Elimination = 0.6*M + 0.25*Website + 0.15*Code
```

## Runtime environment

- 1 NVIDIA GPU, 16 GB VRAM, T4 class;
- 8 CPU cores, 32 GB RAM;
- Python 3.10+;
- no internet during evaluation;
- Part A + Part B: at most `3 * video duration` wall-clock per video;
- total model weights: at most 5 GB;
- a timeout, crash or unreadable package scores that video as empty; a package
  that still does not run after one obvious environment fix gets model score 0.

## Website rubric

| Criterion | Weight | Required evidence |
|---|---:|---|
| Live demo | 30% | Upload succeeds, model returns visualized results, no crash |
| Sample visualizations | 20% | Every sample annotated; readable event timelines and risk curves |
| EDA | 15% | Resolution/fps/duration, counts, trajectories, motion/density findings that shaped the solution |
| Approach and report | 15% | Rebuildable pipeline, learned/rule-based split, honest failures |
| Team and portfolio | 10% | Names, roles, contributions, GitHub/LinkedIn/portfolio links |
| Design, UX, extras | 10% | Clean, fast, mobile-friendly; useful extras |

The demo must state accepted size/length, show progress, and return a timeline,
annotated playback or clips, plus risk curve when Part B is implemented. The
site must remain online through judging.

## Code rubric

| Criterion | Weight | Required evidence |
|---|---:|---|
| Runs as submitted | 40% | Both clean-machine commands work with no manual repair |
| Reproducibility | 25% | Weights, seeds, datasets/training scripts and matching sample predictions |
| Structure/readability | 20% | Clear modules, no dead code, notebooks not the only implementation |
| Engineering judgement | 15% | Sampling, batching, caching and measured runtime margin |

## Current repository compliance (27 September 2026)

| Requirement | Status | Evidence / remaining action |
|---|---|---|
| Root `solution.py` with exact API | Pass | `solution.py`; causal `OnlineRiskEstimator` |
| Organizer runner and metric unchanged | Pass | `run_submission.py`, `evaluate.py` |
| Exact 14 labels and valid JSON | Pass | `solution.py`; `evaluate.py --validate-only` |
| Reproducible training and leakage audit | Pass | Windows and T4 scripts; split hashes; tests |
| External-data/licence disclosure | Pass with caveat | README states DADA has no clear standard licence file |
| Offline weights <= 5 GB | **Release blocker** | Put exported final VideoMAE-S and YOLO weights in submitted package/commit; test `weights/download.sh` |
| `predictions_samples.json` | **Release blocker** | Generate on all four organizer samples and validate |
| Camera geometry | **Release blocker** | Commit the same-camera `configs/scene.json` after calibration |
| Public website | Partial | Frontend and local live demo exist; public URL and persistent GPU backend must be verified |
| EDA and every sample visualization | **Website blocker** | Generate from the four samples; do not replace with DADA metrics |
| Team links/portfolios | Partial | Names/roles present; add verified personal links |
| Technical report/failure analysis | Pass | README and website report section |
