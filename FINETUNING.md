# Fine-tuning plan

The integrated checkpoint is **Simple-TAD DAPT VideoMAE-B**, adapted on
BDD100K + CAP-DATA and fine-tuned for binary traffic anomaly recognition on
DoTA. It is the initial teacher/backbone, not a ready-made 14-class WIUT model.

## Recommended stages

1. **Reproduce inference first.** Initialise the submodule and download the
   checkpoint. Run several normal and dangerous videos, save temporal scores,
   and tune only `temporal_accident_threshold` on a held-out split.
2. **Binary domain adaptation.** Fine-tune the existing 2-class head on DoTA,
   DADA-2000 and any manually labelled WIUT `normal/accident` windows. Keep
   16 frames sampled at 10 FPS so training matches `src/temporal.py`.
3. **WIUT multi-label head.** Replace the 2-way softmax head with 14 sigmoid
   outputs and `BCEWithLogitsLoss`. A window can contain more than one event,
   so ordinary 14-way softmax is the wrong objective.
4. **Hybrid output.** Use the temporal head mainly for `accident`,
   `near_miss`, `road_obstacle` and `fire_smoke`. Keep calibrated geometry and
   tracking for red-light, line, direction, stopping and congestion events.
5. **Calibrate per class.** Select one threshold and minimum duration per class
   on validation data. Optimise event-level macro F1 rather than frame accuracy.

## T4 starting recipe

- input: `3 x 16 x 224 x 224`, sampled at 10 FPS;
- mixed precision: FP16;
- batch size: start at 2, accumulate gradients to an effective batch of 16;
- optimiser: AdamW, backbone LR `1e-5`, head LR `1e-4`, weight decay `0.05`;
- freeze the first 8 transformer blocks for the first 2 epochs;
- use class-balanced sampling and positive-class weights;
- stop on validation event macro F1, not training loss.

The upstream training entry point is
`third_party/simple_tad/run_frame_finetuning.py`; its reproducible configurations
are under `third_party/simple_tad/jobs/finetune`. The DoTA/DADA layouts are
documented in `third_party/simple_tad/DATASET.md`. Do not begin the 14-label run
until the team has produced a manifest containing `video_path`, `start_sec`,
`end_sec`, split and the complete label set for every training window.

## Leakage rules

- split by source video, never by overlapping window;
- never use test clips to choose thresholds;
- retain normal clips and hard negatives from busy intersections;
- for causal Part B, every training window must end at the scored timestamp;
- report results with the untouched organizer evaluator.
