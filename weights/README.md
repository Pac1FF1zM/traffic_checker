# Model weights

Run `python scripts/download_weights.py` before offline evaluation. The script
downloads the official Ultralytics YOLO11n COCO weights into this directory.
For the final submission either commit/package the resulting file or provide the
organizers with the allowed one-time `weights/download.sh` equivalent.

Ultralytics code and weights use AGPL-3.0 unless you have a separate enterprise
license. Confirm that this is acceptable for the hackathon submission.

## Temporal accident weights (recommended)

```bash
git submodule update --init --recursive
python scripts/download_temporal_weights.py
```

This downloads the **Simple-TAD DAPT VideoMAE-S and VideoMAE-B** checkpoints,
traffic-domain-adapted on BDD100K + CAP-DATA and fine-tuned on DoTA. The models
and upstream source are CC BY-NC 4.0: appropriate for a non-commercial
hackathon, but not for unrestricted commercial use.

The optional blind-comparison script also uses the official Kinetics-400
X3D-S checkpoint from PyTorchVideo. X3D-S does not have an accident-trained
head here, so its outputs are recorded only as efficiency/action diagnostics.
