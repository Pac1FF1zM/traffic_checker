param(
    [Parameter(Mandatory = $true)]
    [string]$DataRoot,
    [switch]$Smoke,
    [switch]$Deadline,
    [switch]$FullArchiveCheck,
    [ValidateSet(1, 2)]
    [int]$BatchSize = 1,
    [ValidateRange(0, 8)]
    [int]$NumWorkers = 4,
    [switch]$NoGradientCheckpointing,
    [switch]$ConservativeAllocator
)

$ErrorActionPreference = "Stop"
if ($Smoke -and $Deadline) { throw "Smoke and Deadline modes cannot be used together." }
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$DataRoot = (Resolve-Path $DataRoot).Path
$EffectiveBatchSize = 16
if (($EffectiveBatchSize % $BatchSize) -ne 0) {
    throw "BatchSize must divide effective batch size $EffectiveBatchSize."
}
$UpdateFreq = [int]($EffectiveBatchSize / $BatchSize)
$CheckpointTag = if ($NoGradientCheckpointing) { "gc0" } else { "gc1" }
$AllocatorTag = if ($ConservativeAllocator) { "alloc-safe" } else { "alloc-fast" }
$ProfileTag = "b$($BatchSize)_u$($UpdateFreq)_w$($NumWorkers)_$($CheckpointTag)_$($AllocatorTag)"
$RunTag = if ($Smoke) {
    "smoke_$($ProfileTag)_$((Get-Date).ToString('yyyyMMdd-HHmmss'))"
} elseif ($Deadline) {
    "deadline_$ProfileTag"
} else {
    "train_$ProfileTag"
}
$OutputRoot = Join-Path $RepoRoot "outputs\videomae_s_windows_half\$RunTag"
$Checkpoint = Join-Path $RepoRoot "weights\simpletad_ft-dota_dapt-vm1-s_auroc.pth"
$Python = Join-Path $RepoRoot ".venv\Scripts\python.exe"
$Runner = Join-Path $RepoRoot "scripts\windows\run_simple_tad_standalone.py"

if (-not (Test-Path $Python)) { throw "Missing virtual environment: $Python" }
if (-not (Test-Path $Checkpoint)) { throw "Missing checkpoint: $Checkpoint" }

$env:PYTHONPATH = "$RepoRoot\training_compat;$RepoRoot\third_party\simple_tad"
$env:PYTHONHASHSEED = "42"
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"
$env:CUBLAS_WORKSPACE_CONFIG = ":4096:8"
$env:TRAFFIC_DISABLE_PER_BATCH_CACHE_CLEAR = if ($ConservativeAllocator) { "0" } else { "1" }

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

$samples = if ($Smoke) { 128 } elseif ($Deadline) { 8000 } else { 12000 }
$stage1Epochs = if ($Smoke -or $Deadline) { 1 } else { 2 }
$stage2Epochs = if ($Smoke) { 1 } elseif ($Deadline) { 5 } else { 12 }

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
        "--batch_size", "$BatchSize",
        "--update_freq", "$UpdateFreq",
        "--num_sample", "1",
        "--num_frames", "16",
        "--view_fps", "10",
        "--input_size", "224",
        "--short_side_size", "224",
        "--sampling_rate", "1",
        "--sampling_rate_val", "1",
        "--nb_samples_per_epoch", "$samples",
        "--num_workers", "$NumWorkers",
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
        "--no_flash_attn",
        "--pin_mem",
        "--save_ckpt",
        "--no_auto_resume",
        "--seed", "42"
    )
    if (-not $NoGradientCheckpointing) { $arguments += "--use_checkpoint" }
    if ($Smoke) { $arguments += "--disable_eval_during_finetuning" }
    $resume = Join-Path $stageOutput "checkpoint-last.pth"
    if (Test-Path $resume) { $arguments += @("--resume", $resume) }
    if ($FreezeSpec) { $arguments += @("--freeze_layers", $FreezeSpec) }
    # Keep native stdout visible without returning it as part of this function's
    # value. Windows PowerShell otherwise mixes every training log line into
    # $stageOutput when the caller assigns Invoke-Stage's return value.
    & $Python @arguments | Out-Host
    $trainingExitCode = $LASTEXITCODE
    if ($trainingExitCode -ne 0) { throw "Training stage $Name failed." }
    return $stageOutput
}

$Stopwatch = [System.Diagnostics.Stopwatch]::StartNew()
Write-Host "Training profile: $ProfileTag"
Write-Host "Output directory: $OutputRoot"
$stage1 = Invoke-Stage "stage1_frozen" $Checkpoint $stage1Epochs 0 "first N blocks;8"
$bestStage1 = Join-Path $stage1 "checkpoint-bestauroc.pth"
$lastStage1 = Join-Path $stage1 "checkpoint-last.pth"
$stage2Init = if (Test-Path $bestStage1) { $bestStage1 } else { $lastStage1 }
if (-not (Test-Path $stage2Init)) { throw "Stage 1 produced no checkpoint." }

$stage2Warmup = if ($Smoke) { 0 } elseif ($Deadline) { 1 } else { 2 }
$stage2 = Invoke-Stage "stage2_full" $stage2Init $stage2Epochs $stage2Warmup ""
$Stopwatch.Stop()
$Benchmark = [ordered]@{
    profile = $ProfileTag
    smoke = [bool]$Smoke
    deadline = [bool]$Deadline
    batch_size = $BatchSize
    update_freq = $UpdateFreq
    effective_batch_size = $EffectiveBatchSize
    num_workers = $NumWorkers
    gradient_checkpointing = -not [bool]$NoGradientCheckpointing
    fast_allocator = -not [bool]$ConservativeAllocator
    samples_per_stage_epoch = $samples
    stage1_epochs = $stage1Epochs
    stage2_epochs = $stage2Epochs
    elapsed_seconds = [math]::Round($Stopwatch.Elapsed.TotalSeconds, 2)
    elapsed = $Stopwatch.Elapsed.ToString()
    output_root = $OutputRoot
}
$BenchmarkPath = Join-Path $OutputRoot "benchmark.json"
$Benchmark | ConvertTo-Json | Set-Content -Path $BenchmarkPath -Encoding UTF8
$bestStage2 = Join-Path $stage2 "checkpoint-bestauroc.pth"
$lastStage2 = Join-Path $stage2 "checkpoint-last.pth"
$preferredStage2 = if (Test-Path $bestStage2) { $bestStage2 } else { $lastStage2 }

if ($Deadline) {
    $stage1Log = Join-Path $stage1 "log.txt"
    $stage2Log = Join-Path $stage2 "log.txt"
    if ((Test-Path $stage1Log) -and (Test-Path $stage2Log)) {
        $stage1Records = @(Get-Content $stage1Log | Where-Object { $_.Trim() } | ForEach-Object { $_ | ConvertFrom-Json })
        $stage2Records = @(Get-Content $stage2Log | Where-Object { $_.Trim() } | ForEach-Object { $_ | ConvertFrom-Json })
        $reference = $stage1Records | Select-Object -Last 1
        $best = $stage2Records | Sort-Object -Property @{ Expression = { [double]$_.val_auroc }; Descending = $true } | Select-Object -First 1
        $last = $stage2Records | Select-Object -Last 1
        if (($null -ne $reference) -and ($null -ne $best) -and ($null -ne $last)) {
            function Select-ValidationMetrics($record) {
                return [ordered]@{
                    epoch = [int]$record.epoch
                    auroc = [math]::Round([double]$record.val_auroc, 6)
                    average_precision = [math]::Round([double]$record.val_ap, 6)
                    accuracy = [math]::Round([double]$record.val_metr_acc, 6)
                    precision = [math]::Round([double]$record.val_precision, 6)
                    recall = [math]::Round([double]$record.val_recall, 6)
                    f1 = [math]::Round([double]$record.val_f1, 6)
                }
            }
            $summary = [ordered]@{
                selection_policy = "Best stage-2 checkpoint selected by validation AUROC only"
                test_split_accessed = $false
                stage1_reference = Select-ValidationMetrics $reference
                best_stage2 = Select-ValidationMetrics $best
                last_stage2 = Select-ValidationMetrics $last
                best_vs_stage1_delta = [ordered]@{
                    auroc = [math]::Round(([double]$best.val_auroc - [double]$reference.val_auroc), 6)
                    average_precision = [math]::Round(([double]$best.val_ap - [double]$reference.val_ap), 6)
                    accuracy = [math]::Round(([double]$best.val_metr_acc - [double]$reference.val_metr_acc), 6)
                    f1 = [math]::Round(([double]$best.val_f1 - [double]$reference.val_f1), 6)
                }
                selected_checkpoint = $preferredStage2
            }
            $summaryPath = Join-Path $OutputRoot "metrics_summary.json"
            $summary | ConvertTo-Json -Depth 6 | Set-Content -Path $summaryPath -Encoding UTF8
            Write-Host "Metrics summary: $summaryPath"
        }
    }
}
Write-Host "Elapsed: $($Stopwatch.Elapsed)"
Write-Host "Benchmark: $BenchmarkPath"
Write-Host "Training complete. Preferred checkpoint: $preferredStage2"
