# Team404 — fixed-camera traffic event detection

Submission repository for the WIUT Hackathon 2026 Computer Vision elimination
task: **Toyota Traffic Event Detection and Accident Anticipation from a Fixed
Road Camera**.

The system implements both official interfaces:

- **Part A (mandatory):** temporal segments `[start_sec, end_sec, label]` for
  the 14 official traffic-event classes.
- **Part B (bonus):** a causal `RiskEstimator` that receives frames one by one
  and returns `P(accident starts within 5 seconds)` in `[0, 1]`.

`run_submission.py` and `evaluate.py` are the unchanged organizer files. The
official hidden set uses the same fixed camera and angle as the four unlabeled
sample videos. See [HACKATHON_REQUIREMENTS.md](HACKATHON_REQUIREMENTS.md) for the
requirements audit and [SUBMISSION_CHECKLIST.md](SUBMISSION_CHECKLIST.md) for
the release checklist.

## Clean-machine run

Python 3.10+ and an NVIDIA GPU are recommended. Evaluation is offline; run the
weight preparation once while internet access is available.

```bash
python -m venv .venv
# Linux: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
git submodule update --init --recursive
bash weights/download.sh

python run_submission.py --videos /data/test --out predictions.json --team Team404
python evaluate.py --pred predictions.json --validate-only
```

The submitted package should contain the trained VideoMAE-S checkpoint at
`weights/simpletad_ft-dota_dapt-vm1-s_auroc.pth`. If that file is absent,
`weights/download.sh` installs the public Simple-TAD DoTA checkpoint as a
runnable fallback; the fallback is not the model behind the reported DADA
fine-tuning results.

The complete Part A + Part B wall-clock budget is at most three times each
video's duration. The code samples detection frames and caches model instances
to retain margin on the organizer's T4-class 16 GB GPU.

## Architecture

```text
fixed MP4
  ├─ YOLO11n road-user detector ─ centroid/IoU tracker ─ geometry rules
  │                                                └─ causal TTC risk
  └─ causal 16-frame VideoMAE-S ─ anomaly probability ─┬─ event segments
                                                       └─ 5-second risk curve
```

Learned components:

- YOLO11n COCO road-user detector;
- Simple-TAD VideoMAE-S, initialized from its public DoTA checkpoint and
  continued on a leakage-safe DADA-2000 split.

Rule-based components:

- deterministic association and track histories;
- time-to-collision risk;
- stopped vehicle, congestion, wrong-way, pedestrian/crosswalk, red-light and
  line-crossing rules when camera geometry is configured;
- temporal segment merging and minimum-duration filtering.

The Part B implementation is causal: `RiskEstimator.step(frame, t_sec)` uses
only the current frame, retained past frames and track state. It never opens the
video and never reuses Part A output.

## Official output contract

`solution.py` exposes `CLASSES`, `detect_events(video_path)` and
`RiskEstimator.reset/step`. The exact class ids are:

```text
accident, near_miss, red_light, wrong_way, illegal_u_turn,
stopped_vehicle, jaywalking, failure_to_yield, illegal_turn,
solid_line_crossing, stop_line, congestion, road_obstacle, fire_smoke
```

The harness writes one entry per test file:

```json
{
  "team": "Team404",
  "videos": {
    "test_001.mp4": {
      "events": [[12.4, 18.9, "accident"]],
      "risk": [[0.0, 0.01], [0.04, 0.02]]
    }
  }
}
```

Same-class segments never overlap. Risk timestamps are non-decreasing and all
scores are clipped to `[0, 1]`.

## Training data, leakage controls and licences

Only public external data was used for training. The four WIUT sample videos
were not used for model selection, threshold tuning or the reported DADA test.

| Asset | Use | Licence / terms |
|---|---|---|
| DADA-2000 | VideoMAE-S continuation and held-out evaluation | The official repository publishes the benchmark for research but does not provide a clear standard licence file; do not redistribute the videos and verify permission for any use beyond this academic hackathon. |
| Simple-TAD source and public checkpoint | Architecture and initialization | Majority CC BY-NC 4.0; separately identified dependencies retain Apache-2.0, MIT or BSD terms. |
| Ultralytics YOLO11n code and weights | Road-user detection | AGPL-3.0 unless covered by an Ultralytics Enterprise licence. This repository is public; downstream proprietary use requires a separate licence review. |
| COCO | Detector pre-training through YOLO11n | COCO image annotations are CC BY 4.0; individual images retain their source terms. |

The deterministic seed is `42`. The DADA preparation script creates disjoint
source-video splits and records hashes in `split_manifest.json`:

- train: 419 clips;
- validation: 74 clips;
- held-out test: 227 clips;
- pairwise train/validation/test overlap: 0.

Run the leakage and reproducibility checks with:

```bash
python tests/test_training_bundle.py
```

See [WINDOWS_HALF_TRAINING.md](WINDOWS_HALF_TRAINING.md) for the exact Windows
training commands, hyperparameters and one-shot held-out evaluation.

## Verified external-domain result

Checkpoint selection used validation AUROC only. The selected checkpoint was
then evaluated once on 227 untouched DADA-2000 dashcam clips (52,255 windows):

| Metric | Result |
|---|---:|
| AUROC | 84.55% |
| Average precision | 79.46% |
| Accuracy | 79.33% |
| Precision at 0.5 | 84.24% |
| Recall at 0.5 | 52.86% |
| F1 at 0.5 | 64.96% |

These are **not hidden-WIUT scores** and are **not fixed-camera accuracy
claims**. DADA-2000 is dashcam footage, so fixed-camera sample performance must
be assessed separately with team annotations and the organizer's exact metric.

## Camera calibration

Copy `configs/scene.example.json` to `configs/scene.json` and fill normalized
camera geometry: road, crosswalk, intersection, signal ROI, stop line, solid
lines and lane directions. Empty fields disable the corresponding rule rather
than guessing. Set `WIUT_SCENE_CONFIG` to load another file.

## Local team website and live demo

```powershell
python -m pip install -r requirements-demo.txt
Push-Location .\website
npm.cmd install
npm.cmd run build
Pop-Location

$Checkpoint = ".\weights\simpletad_ft-dota_dapt-vm1-s_auroc.pth"
.\scripts\windows\start_live_demo.ps1 -Checkpoint $Checkpoint
```

Open `http://127.0.0.1:8000`. The visitor can upload a video up to 300 MB and
receives event intervals, a causal risk curve and runtime diagnostics. Uploads
are processed locally and deleted after inference. Deployment instructions are
in [LIVE_DEMO.md](LIVE_DEMO.md).

## Tests

```bash
python tests/test_baseline.py
python tests/test_demo_calibration.py
python tests/test_training_bundle.py
python -m py_compile solution.py src/*.py run_submission.py evaluate.py demo_api.py
python evaluate.py --pred predictions_samples.json --validate-only
```

## Honest limitations

- The temporal model was trained on dashcam data; fixed-camera calibration is
  a conservative cross-domain adaptation, not a substitute for labelled CCTV
  training data.
- Scene-dependent classes require the supplied-camera geometry.
- `illegal_u_turn`, `illegal_turn`, `stop_line`, `road_obstacle` and
  `fire_smoke` are not yet emitted by the current rules.
- Sample-video EDA and `predictions_samples.json` must be generated from the
  organizer-provided files before tagging the submission commit.

## Team404

- **Murodkulov Nazarbek Jonibekovich** — captain; model training and evaluation.
- **Davronkulov Abubark Davlatovich** — pipeline integration, testing and deployment.
- **Ruziyev Firdavs Negmurodovich** — website, visualizations and presentation.

## Repository layout

```text
solution.py                    official interface implementation
run_submission.py              organizer harness, unchanged
evaluate.py                    organizer metric, unchanged
src/                           detector, tracking, rules and causal temporal model
configs/                       fixed-camera configuration
weights/download.sh            one-time online weight preparation
scripts/windows/               reproducible Windows training/evaluation
scripts/lab/                   Linux/T4 training and export utilities
website/                       React public presentation and upload UI
HACKATHON_REQUIREMENTS.md       official requirements audit
SUBMISSION_CHECKLIST.md         release and submission checklist
```
