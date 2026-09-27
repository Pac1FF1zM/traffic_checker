param(
    [Parameter(Mandatory = $true)]
    [string]$DataRoot,
    [switch]$Smoke,
    [switch]$FullArchiveCheck
)

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$DataRoot = (Resolve-Path $DataRoot).Path
$OutputRoot = Join-Path $RepoRoot "outputs\videomae_s_windows_half"
$Checkpoint = Join-Path $RepoRoot "weights\simpletad_ft-dota_dapt-vm1-s_auroc.pth"
$Python = Join-Path $RepoRoot ".venv\Scripts\python.exe"
$Runner = Join-Path $RepoRoot "scripts\windows\run_simple_tad_standalone.py"

if (-not (Test-Path $Python)) { throw "Missing virtual environment: $Python" }
if (-not (Test-Path $Checkpoint)) { throw "Missing checkpoint: $Checkpoint" }

$env:PYTHONPATH = "$RepoRoot\training_compat;$RepoRoot\third_party\simple_tad"
$env:PYTHONHASHSEED = "42"
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"
$env:CUBLAS_WORKSPACE_CONFIG = ":4096:8"

$preflightArgs = @(
    (Join-Path $RepoRoot "scripts\lab\preflight_training.py"),
    "--dataset-kind", "DADA2K",
    "--data-root", $DataRoot,
    "--checkpoint", $Checkpoint,
    "--output", (Join-Path $OutputRoot "preflight.json"),
    "--train-split", "half_training.txt",
    "--validation-split", "validation.txt",
    "--test-split", "test.txt",
    "--min-vram-gib", "7.0"
)
if ($FullArchiveCheck) { $preflightArgs += "--full-check" }
& $Python @preflightArgs
if ($LASTEXITCODE -ne 0) { throw "Preflight failed; training was not started." }

$samples = if ($Smoke) { 128 } else { 12000 }
$stage1Epochs = if ($Smoke) { 1 } else { 2 }
$stage2Epochs = if ($Smoke) { 1 } else { 12 }

function Invoke-Stage {
    param(
        [string]$Name,
        [string]$InitialCheckpoint,
        [int]$Epochs,
        [int]$WarmupEpochs,
        [string]$FreezeSpec
    )
    $stageOutput = Join-Path $OutputRoot $Name
    New-Item -ItemType Directory -Force -Path $stageOutput | Out-Null
    $arguments = @(
        $Runner,
        "--model", "vit_small_patch16_224",
        "--data_set", "DADA2K_half",
        "--data_path", $DataRoot,
        "--finetune", $InitialCheckpoint,
        "--output_dir", $stageOutput,
        "--log_dir", $stageOutput,
        "--loss", "crossentropy",
        "--nb_classes", "2",
        "--tubelet_size", "2",
        "--batch_size", "1",
        "--update_freq", "16",
        "--num_sample", "1",
        "--num_frames", "16",
        "--view_fps", "10",
        "--input_size", "224",
        "--short_side_size", "224",
        "--sampling_rate", "1",
        "--sampling_rate_val", "1",
        "--nb_samples_per_epoch", "$samples",
        "--num_workers", "2",
        "--epochs", "$Epochs",
        "--warmup_epochs", "$WarmupEpochs",
        "--opt", "adamw",
        "--opt_betas", "0.9", "0.999",
        "--lr", "5e-4",
        "--min_lr", "1e-6",
        "--warmup_lr", "1e-6",
        "--weight_decay", "0.05",
        "--layer_decay", "0.6",
        "--drop_path", "0.2",
        "--clip_grad", "1.0",
        "--aa", "rand-m6-n3-mstd0.5-inc1",
        "--reprob", "0.1",
        "--test_num_segment", "1",
        "--test_num_crop", "1",
        "--use_checkpoint",
        "--no_flash_attn",
        "--pin_mem",
        "--save_ckpt",
        "--no_auto_resume",
        "--seed", "42"
    )
    $resume = Join-Path $stageOutput "checkpoint-last.pth"
    if (Test-Path $resume) { $arguments += @("--resume", $resume) }
    if ($FreezeSpec) { $arguments += @("--freeze_layers", $FreezeSpec) }
    & $Python @arguments
    if ($LASTEXITCODE -ne 0) { throw "Training stage $Name failed." }
    return $stageOutput
}

$stage1 = Invoke-Stage "stage1_frozen" $Checkpoint $stage1Epochs 0 "first N blocks;8"
$bestStage1 = Join-Path $stage1 "checkpoint-bestauroc.pth"
$lastStage1 = Join-Path $stage1 "checkpoint-last.pth"
$stage2Init = if (Test-Path $bestStage1) { $bestStage1 } else { $lastStage1 }
if (-not (Test-Path $stage2Init)) { throw "Stage 1 produced no checkpoint." }

$stage2 = Invoke-Stage "stage2_full" $stage2Init $stage2Epochs $(if ($Smoke) { 0 } else { 2 }) ""
Write-Host "Training complete. Preferred checkpoint: $stage2\checkpoint-bestauroc.pth"
