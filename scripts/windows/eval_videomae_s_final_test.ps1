param(
    [Parameter(Mandatory = $true)]
    [string]$DataRoot,
    [string]$Checkpoint = "",
    [switch]$FinalTest
)

$ErrorActionPreference = "Stop"
if (-not $FinalTest) {
    throw "Held-out test is locked. Re-run with -FinalTest only after all model and threshold choices are frozen."
}

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$DataRoot = (Resolve-Path $DataRoot).Path
$Python = Join-Path $RepoRoot ".venv\Scripts\python.exe"
$Runner = Join-Path $RepoRoot "scripts\windows\eval_simple_tad_heldout.py"
if (-not $Checkpoint) {
    $Checkpoint = Join-Path $RepoRoot "outputs\videomae_s_windows_half\stage2_full\checkpoint-bestauroc.pth"
}
$Checkpoint = (Resolve-Path $Checkpoint).Path
$TestSplit = Join-Path $DataRoot "DADA2K_my_split\test.txt"
if (-not (Test-Path $TestSplit)) { throw "Missing held-out split: $TestSplit" }

$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$Output = Join-Path $RepoRoot "outputs\videomae_s_windows_half\final_test_$stamp"
New-Item -ItemType Directory -Force -Path $Output | Out-Null

$accessRecord = [ordered]@{
    evaluated_at_utc = (Get-Date).ToUniversalTime().ToString("o")
    checkpoint = $Checkpoint
    checkpoint_sha256 = (Get-FileHash -Algorithm SHA256 $Checkpoint).Hash.ToLower()
    test_split = $TestSplit
    test_split_sha256 = (Get-FileHash -Algorithm SHA256 $TestSplit).Hash.ToLower()
    policy = "One-shot held-out test; do not use results for checkpoint or hyperparameter selection"
}
$accessRecord | ConvertTo-Json | Set-Content -Encoding UTF8 (Join-Path $Output "test_access_record.json")

$env:PYTHONPATH = "$RepoRoot\training_compat;$RepoRoot\third_party\simple_tad"
$env:DADA_HELDOUT_SPLIT = "DADA2K_my_split/test.txt"
$env:PYTHONHASHSEED = "42"
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"

& $Python $Runner `
    --eval `
    --dist_eval `
    --model vit_small_patch16_224 `
    --data_set DADA2K_half `
    --data_path $DataRoot `
    --finetune $Checkpoint `
    --output_dir $Output `
    --nb_classes 2 `
    --tubelet_size 2 `
    --batch_size 1 `
    --num_frames 16 `
    --view_fps 10 `
    --input_size 224 `
    --short_side_size 224 `
    --sampling_rate 1 `
    --sampling_rate_val 1 `
    --num_workers 2 `
    --test_num_segment 1 `
    --test_num_crop 1 `
    --no_flash_attn `
    --no_auto_resume `
    --pin_mem `
    --seed 42
if ($LASTEXITCODE -ne 0) { throw "Held-out evaluation failed." }
Write-Host "Final held-out test saved to $Output"
