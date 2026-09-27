# VideoMAE-B training on a university T4

For native Windows training on an 8 GB RTX 4060 Ti with a deterministic,
source-disjoint DADA-2000 half split, use
[WINDOWS_HALF_TRAINING.md](WINDOWS_HALF_TRAINING.md). The Windows profile uses
VideoMAE-S and keeps a separate held-out test split.

This bundle fine-tunes the selected Simple-TAD DAPT VideoMAE-B checkpoint on
an **external** traffic-anomaly dataset. The recommended continuation is
DoTA checkpoint -> DADA-2000. The provided WIUT videos remain a locked blind
set and are never used for training, validation, threshold selection, or early
stopping.

## Fixed experiment profile

| Item | Value |
|---|---|
| GPU | 1 x NVIDIA T4, 16 GB |
| Runtime | Python 3.10, PyTorch 2.5.1, CUDA 12.1 wheels |
| Input | 16 RGB frames, 224 x 224, sampled at 10 FPS |
| Precision | native FP16 AMP |
| Attention | standard PyTorch attention; FlashAttention disabled on T4 |
| Optimizer | AdamW, weight decay 0.05 |
| Effective batch | micro-batch 2 x accumulation 8 x 1 GPU = 16 |
| Schedule | 2 frozen epochs + 18 full epochs |
| Frozen stage | first 8 transformer blocks |
| Reference LR | 5e-4, linearly scaled by upstream code to batch size |
| Regularization | layer decay 0.6, drop path 0.2, grad clip 1.0 |
| Model selection | highest AUROC on external validation split only |
| Seed | 42 |

The architecture and deployment input remain unchanged during tuning. This is
important: increasing training FPS or frame count would make the trained model
incompatible with the current causal inference adapter.

## Machine and disk requirements

- Linux with an NVIDIA driver compatible with CUDA 12.1 wheels;
- Python 3.10 and `python3.10-venv`;
- a T4-class CUDA GPU with at least 14 GiB usable VRAM;
- at least 32 GB system RAM recommended;
- enough local SSD space for the extracted frame archives, environment, and
  checkpoints (budget at least 100 GB beyond the dataset itself).

Weights, datasets, `outputs/`, and generated metadata are ignored by Git.

## 1. Copy and install

```bash
git clone --recurse-submodules https://github.com/Pac1FF1zM/traffic_checker.git
cd traffic_checker
bash scripts/lab/setup_linux_t4.sh
source .venv-lab/bin/activate
```

The setup script installs the official PyTorch 2.5.1 CUDA 12.1 wheels, the
pinned training dependencies, the Simple-TAD submodule, and both temporal
initialization checkpoints. It deliberately does not install FlashAttention.

## 2. Prepare one external dataset

Recommended DADA-2000 layout (the path is configured as `/data/DADA2000`):

```text
/data/DADA2000/
  DADA2K_my_split/
    training.txt
    validation.txt
  annotation/
    full_anno.csv
  frames/
    <clip-id>/images.zip
```

Alternative DoTA layout:

```text
/data/DoTA/
  dataset/
    train_split.txt
    val_split.txt
    annotations/<clip-id>.json
  frames/
    <clip-id>/images.zip
```

Each line in a split file must identify one source clip, and the same source
must not occur in both splits. Edit only `DATA_ROOT` if your mount point is
different:

```bash
nano configs/training/videomae_b_t4_dada.env
```

Do not copy WIUT blind videos into the dataset root. The preflight rejects
known WIUT references and exact train/validation source overlap.

## 3. Run the integration smoke test

First edit the smoke config's `DATA_ROOT` to the same dataset location, then:

```bash
bash scripts/lab/train_videomae_b_t4.sh \
  configs/training/videomae_b_t4_smoke.env 2>&1 | tee smoke.log
```

This executes both training stages with only 128 samples per epoch. It checks
the complete software/data path, but its checkpoint is not a candidate model.

## 4. Train the full profile

Run inside `tmux` so an SSH disconnect does not stop training:

```bash
tmux new -s videomae
source .venv-lab/bin/activate
bash scripts/lab/train_videomae_b_t4.sh \
  configs/training/videomae_b_t4_dada.env 2>&1 | tee train_videomae_b.log
```

The ordinary preflight validates a representative archive sample. For a full
archive integrity scan before training (potentially slow), set:

```bash
PREFLIGHT_FULL_CHECK=1 bash scripts/lab/train_videomae_b_t4.sh \
  configs/training/videomae_b_t4_dada.env
```

Monitor the run with `nvidia-smi` and, in a second terminal:

```bash
source .venv-lab/bin/activate
tensorboard --logdir outputs --bind_all
```

The launcher is restart-safe through upstream auto-resume. Its main outputs
are under `outputs/videomae_b_t4_dada/stage2_full/`:

- `checkpoint-bestauroc.pth`: preferred validation-selected checkpoint;
- `checkpoint-last.pth`: full optimizer state for resuming;
- `log.txt` and TensorBoard events: training evidence;
- `preflight.json`: environment, split hashes, checkpoint hash, and leakage audit.

If CUDA runs out of memory, change `BATCH_SIZE=1` and `UPDATE_FREQ=16`; the
effective batch remains 16. If data loading exhausts RAM or stalls storage,
reduce `NUM_WORKERS` from 4 to 2. Keep 16 frames, 10 FPS, and 224 px fixed.

## 5. Evaluate without touching WIUT blind videos

```bash
bash scripts/lab/eval_videomae_b.sh \
  configs/training/videomae_b_t4_dada.env \
  outputs/videomae_b_t4_dada/stage2_full/checkpoint-bestauroc.pth
```

Use only the external validation metrics to select between the initial and
fine-tuned checkpoints. Do not choose a checkpoint or threshold by watching its
predictions on organizer videos.

## 6. Export for this repository's inference adapter

```bash
python scripts/lab/export_videomae_b_checkpoint.py \
  outputs/videomae_b_t4_dada/stage2_full/checkpoint-bestauroc.pth \
  weights/videomae_b_wiut_candidate.pth
```

The exporter unwraps training state, verifies a strict load into the exact
VideoMAE-B inference architecture, saves a CPU state dict, and writes a SHA-256
metadata sidecar. Point `temporal_checkpoint` in your local
`configs/scene.json` at this exported file only after external validation.

## Reproducibility artifacts to retain

Keep the Git commit hash, both split files, `preflight.json`, the `.env` config,
training log, external validation output, exported checkpoint, and metadata
sidecar together. These files are sufficient to identify the code, data split,
initial weights, settings, and resulting weights without committing large or
licensed artifacts to GitHub.
