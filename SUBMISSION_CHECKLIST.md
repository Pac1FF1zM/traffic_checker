# Team404 release checklist

The official deadline is 27 September 2026, 23:59 Tashkent time. Do not tag or
submit until every blocking item below is resolved.

## 1. Package the trained checkpoint

On the Windows machine that contains the completed training run:

```powershell
Set-Location "C:\Users\RUSLANDO\Desktop\traffic_checker"
$Source = ".\outputs\videomae_s_windows_half\deadline_b1_u16_w4_gc1_alloc-fast\stage2_full\checkpoint-bestauroc.pth"
$Target = ".\weights\simpletad_ft-dota_dapt-vm1-s_auroc.pth"

python .\scripts\lab\export_videomae_b_checkpoint.py $Source $Target --arch videomae_small
Get-Item $Target | Select-Object FullName,Length
```

If the exported file is below GitHub's single-file limit, add it explicitly
despite `.gitignore`:

```powershell
git add -f .\weights\simpletad_ft-dota_dapt-vm1-s_auroc.pth
```

If it is too large, upload it to one public archive/release and update
`weights/download.sh` with the exact URL and SHA-256. Do not submit a script
that downloads the public fallback while claiming it is the fine-tuned model.

## 2. Add fixed-camera calibration

Create `configs/scene.json` for the organizer's camera. Verify road,
crosswalk/intersection, signal ROI, stop/solid lines and lane directions.
Commit it; the hidden videos use the same camera and angle.

## 3. Generate organizer-sample predictions

```powershell
python .\run_submission.py `
  --videos "C:\path\to\all-four-organizer-samples" `
  --out .\predictions_samples.json `
  --team Team404

python .\evaluate.py --pred .\predictions_samples.json --validate-only
git add .\predictions_samples.json
```

Confirm all four filenames are present, even when an event list is empty. Do
not tune against hidden test data.

## 4. Final clean-machine checks

```powershell
python -m pip install -r .\requirements.txt
git submodule update --init --recursive
python .\tests\test_baseline.py
python .\tests\test_demo_calibration.py
python .\tests\test_training_bundle.py
python -m py_compile .\solution.py .\src\*.py .\run_submission.py .\evaluate.py .\demo_api.py
python .\evaluate.py --pred .\predictions_samples.json --validate-only
```

Run at least one full sample through the official default command and confirm
`total_sec <= 3 * duration` with margin on the target GPU.

## 5. Website checks

- Public HTTPS URL opens in a private browser window.
- Upload demo is reachable from the public site, not only `127.0.0.1`.
- A two-minute upload shows progress and completes without a crash.
- Every organizer sample has an annotated video, event timeline and risk curve.
- EDA contains measured resolution, fps, duration, counts, trajectories,
  density and motion findings.
- Team names, roles, contributions and verified personal links are present.
- Repository, weights and `predictions_samples.json` are linked.
- DADA numbers are clearly labelled external dashcam results, not WIUT scores.

## 6. Freeze the exact release

```powershell
git status --short
git add README.md HACKATHON_REQUIREMENTS.md SUBMISSION_CHECKLIST.md `
  DATASETS.md solution.py configs weights website src scripts tests requirements*.txt
git commit -m "Prepare WIUT elimination submission"
git tag -a wiut-elimination-v1 -m "WIUT elimination submission"
git push origin codex/website
git push origin wiut-elimination-v1
git rev-parse HEAD
```

Record the public repository URL, exact tag/hash, public website URL and three
T-shirt sizes. Submission is an external irreversible action; inspect the final
commit and website before pressing the organizer's Submit button.
