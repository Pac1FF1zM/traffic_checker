# Leakage-safe DADA-2000 half training on Windows

This profile targets one RTX 4060 Ti 8 GB and uses VideoMAE-S. It does not use
the WIUT blind videos for training, model selection, calibration, or qualitative
selection.

## Split policy

The split generator preserves the official DADA separation:

1. The official training list is the only source for training and tuning.
2. A deterministic, stratified 50% pool is selected from official training.
3. Ten percent of that selected pool becomes validation; the rest is training.
4. A deterministic, stratified 20% subset of official validation becomes the
   held-out test and is never loaded by the training command. The rest is not
   used, which keeps the selected frame data within the local disk budget.
5. Stratification uses accident category, accident presence, and day/night.
6. Every split is source-video-disjoint and recorded with SHA-256 hashes.

Do not repeatedly evaluate the test split. Run it once, after freezing the
model, checkpoint-selection rule, thresholds, and all hyperparameters.

## Expected dataset layout

```text
DADA2000/
  annotation/full_anno.csv
  DADA2K_my_split/training.txt
  DADA2K_my_split/validation.txt
  frames/<category>/<clip>/images.zip
```

Download the Simple-TAD annotation package from:

```text
https://huggingface.co/tue-mps/simple-tad/resolve/main/datasets/D2K.zip
```

Use the official DADA-2000 source for frame data. Keep only the source clips
listed by the generated `half_pool.txt` plus `test.txt` when disk space is
limited.

## Generate immutable splits

```powershell
python scripts\prepare_dada_half_splits.py --data-root C:\datasets\DADA2000 --fraction 0.5 --validation-fraction 0.1 --test-fraction 0.2 --seed 42
```

Download only the selected RGB clips from the official multi-volume release. The
downloader reads the remote ZIP index and does not store the 117 GB source archive.
Frames are reduced to a 256-pixel short side (the training crop is 224) and packed
directly into the `images.zip` files expected by Simple-TAD. Interrupted runs keep
completed clips and write the current clip atomically.

```powershell
# Two-clip end-to-end check. The reusable remote index is about 460 MiB.
python scripts\download_dada_selected.py --data-root C:\datasets\DADA2000 --short-side 256 --limit-clips 2

# Complete the selected dataset; the first two clips are reused.
python scripts\download_dada_selected.py --data-root C:\datasets\DADA2000 --short-side 256
```

If the official Google Drive volume reports `Quota exceeded`, keep all completed
clips and fill only the missing ones from the independent Hugging Face mirror.
The mirror is a sequential 124 GB gzip stream, so this avoids extra disk usage but
may transfer the whole compressed dataset before every missing clip is found.

```powershell
python scripts\download_dada_missing_hf.py --data-root C:\datasets\DADA2000
```

The command preserves the originals as `official_training.txt` and
`official_validation.txt`, writes `half_training.txt`, `validation.txt`,
`test.txt`, and `selected_dataset_clips.txt`, and creates
`split_manifest.json`. Only clips in `selected_dataset_clips.txt` are needed
locally.

## Smoke test

```powershell
.\scripts\windows\train_videomae_s_half.ps1 -DataRoot C:\datasets\DADA2000 -Smoke
```

The preflight checks package versions, VRAM, checkpoint presence, archive
integrity samples, duplicate sources, pairwise split overlap, and forbidden
WIUT references. Training does not start if any check fails.

## Full run

```powershell
.\scripts\windows\train_videomae_s_half.ps1 -DataRoot C:\datasets\DADA2000
```

Select the checkpoint by validation AUROC only. Do not inspect the held-out test
predictions while changing the model.

## One-shot final test

After every choice is frozen:

```powershell
.\scripts\windows\eval_videomae_s_final_test.ps1 -DataRoot C:\datasets\DADA2000 -FinalTest
```

This explicit gate records the checkpoint hash and test-split hash before
evaluation. Test metrics are for final reporting, not further tuning.
