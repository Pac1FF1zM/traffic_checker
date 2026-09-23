# WIUT blind model comparison

Protocol: `wiut_blind_protocol.json` (locked before inference). No WIUT frames
were viewed, annotated, used for training, used for threshold selection, or used
for checkpoint selection. The run used three deterministic 60-second windows
(start, midpoint, end) from the one complete video that Google Drive allowed us
to download.

## Inputs

- `wiut_03.mp4`: 3840x2160, 29.970 fps, 317.8175 s, SHA-256
  `146f8b7d5ced37fdd2cad3129d99d83ad50502f79c96b933da6beacecdd2155d`.
- The other three links were not substituted: IDs 1 and 2 returned Drive quota
  errors; ID 4 returned the same error after the third download completed.

## Results on this laptop

| Model | Head/task | Device | Inferences | Wall time | Time / video second | Raw-score summary |
|---|---|---:|---:|---:|---:|---|
| Simple-TAD VideoMAE-S DAPT DoTA | accident risk | DirectML | 438 | 244.32 s | 1.357x | median 0.0681; max 0.5240 |
| Simple-TAD VideoMAE-B DAPT DoTA | accident risk | DirectML | 438 | 332.87 s | 1.849x | median 0.0536; max 0.4753 |
| X3D-S Kinetics-400 | action diagnostics only | CPU | 432 | 315.72 s | 1.754x | top-1 entropy median 0.4202; 4 unique top-1 classes |

VideoMAE-S and VideoMAE-B scores had Pearson correlation 0.739; their maxima
occurred at the same sampled timestamp. No event threshold was applied to this
unlabeled set. For descriptive context only, neither model crossed the existing
0.75 submission threshold; S had 2/438 samples above 0.5 and B had 0/438, but
these counts are not used for calibration or model selection.

X3D-S must not be compared as an accident detector here: its public checkpoint
has a 400-class Kinetics action head, not an accident-trained binary head. Its
runtime and output stability are useful for planning a later public-DoTA
fine-tuning run, but this blind set cannot establish accuracy or select a winner.

The complete raw output (all timestamps and scores) is in
`tmp/wiut_blind_results.json` and is intentionally ignored by Git because it is
large experiment data.
