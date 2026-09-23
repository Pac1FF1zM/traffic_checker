# Fine-tuning plan

The integrated checkpoint is **Simple-TAD DAPT VideoMAE-B**, adapted on
BDD100K + CAP-DATA and fine-tuned for binary traffic anomaly recognition on
DoTA. It is the initial teacher/backbone, not a ready-made 14-class WIUT model.

## Recommended stages

1. **Reproduce inference first.** Initialise the submodule and download the
   checkpoint. Verify loading and runtime without changing the model.
2. **Binary domain adaptation.** Continue the existing 2-class head from the
   DoTA checkpoint on DADA-2000. Keep 16 frames sampled at 10 FPS so training
   matches `src/temporal.py`.
3. **WIUT multi-label head.** Replace the 2-way softmax head with 14 sigmoid
   outputs and `BCEWithLogitsLoss`. A window can contain more than one event,
   so ordinary 14-way softmax is the wrong objective.
4. **Hybrid output.** Use the temporal head mainly for `accident`,
   `near_miss`, `road_obstacle` and `fire_smoke`. Keep calibrated geometry and
   tracking for red-light, line, direction, stopping and congestion events.
5. **Calibrate per class.** Select one threshold and minimum duration per class
   on validation data. Optimise event-level macro F1 rather than frame accuracy.

## Selected T4 recipe

- input: `3 x 16 x 224 x 224`, sampled at 10 FPS;
- mixed precision: FP16;
- batch size: start at 2, accumulate gradients to an effective batch of 16;
- optimiser: AdamW, reference LR `5e-4` scaled by effective batch in the
  upstream trainer, weight decay `0.05`, layer decay `0.6`;
- freeze the first 8 transformer blocks for the first 2 epochs;
- train for 2 frozen plus 18 fully unfrozen epochs;
- select the checkpoint by external validation AUROC, not training loss.

The ready-to-run configs, preflight, smoke test, two-stage launcher, evaluation,
and strict checkpoint exporter are documented in [LAB_TRAINING.md](LAB_TRAINING.md).

The upstream training entry point is
`third_party/simple_tad/run_frame_finetuning.py`; its reproducible configurations
are under `third_party/simple_tad/jobs/finetune`. The DoTA/DADA layouts are
documented in `third_party/simple_tad/DATASET.md`. Do not begin the 14-label run
until the team has produced a manifest containing `video_path`, `start_sec`,
`end_sec`, split and the complete label set for every training window.

## Leakage rules

- split by source video, never by overlapping window;
- never use WIUT blind/test clips to train, select checkpoints, choose
  thresholds, or make qualitative model choices;
- retain normal clips and hard negatives from busy intersections;
- for causal Part B, every training window must end at the scored timestamp;
- report results with the untouched organizer evaluator.
