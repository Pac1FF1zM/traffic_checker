# Local GPU live demo

The React site and Python inference API are served from one local address. The
final checkpoint remains outside Git and is supplied explicitly at startup.

## One-time setup on Windows

From the repository root with the Python 3.11 virtual environment activated:

```powershell
python -m pip install -r requirements-demo.txt
Push-Location .\website
npm install
npm run build
Pop-Location
```

## Start

```powershell
$Checkpoint = ".\outputs\videomae_s_windows_half\deadline_b1_u16_w4_gc1_alloc-fast\stage2_full\checkpoint-bestauroc.pth"
.\scripts\windows\start_live_demo.ps1 -Checkpoint $Checkpoint
```

Open `http://127.0.0.1:8000`. The API accepts MP4, AVI, MOV, MKV, and WebM up
to 300 MB, serializes GPU jobs, and deletes the temporary upload after the
response. Set `WIUT_DEMO_MAX_UPLOAD_MB` before startup to change the limit.

The page reports the fixed held-out metrics from the one-shot test. Do not tune
thresholds from those results. Scene-specific rule events require calibrated
geometry in `configs/scene.json`; temporal accident risk works without it.
