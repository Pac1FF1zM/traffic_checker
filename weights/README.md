# Model weights

Run `bash weights/download.sh` once with internet access before offline
evaluation. It prepares YOLO11n, the pinned Simple-TAD source and a VideoMAE-S
checkpoint. A packaged Team404 checkpoint is preserved; when it is missing the
script downloads the public DoTA checkpoint as an explicitly reported fallback.

Ultralytics code and weights use AGPL-3.0 unless you have a separate enterprise
license. Confirm that this is acceptable for the hackathon submission.

## Temporal accident weights (recommended)

```bash
git submodule update --init --recursive
python scripts/download_temporal_weights.py
```

This downloads the **Simple-TAD DAPT VideoMAE-S and VideoMAE-B** checkpoints for
development. The submitted inference path uses VideoMAE-S. The models and
upstream source are CC BY-NC 4.0: appropriate for a non-commercial academic
hackathon, but not for unrestricted commercial use.

The optional blind-comparison script also uses the official Kinetics-400
X3D-S checkpoint from PyTorchVideo. X3D-S does not have an accident-trained
head here, so its outputs are recorded only as efficiency/action diagnostics.
