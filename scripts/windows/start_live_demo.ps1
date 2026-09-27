param(
    [Parameter(Mandatory = $true)]
    [string]$Checkpoint,
    [int]$Port = 8000
)

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$Python = Join-Path $RepoRoot ".venv\Scripts\python.exe"
$Checkpoint = (Resolve-Path $Checkpoint).Path
$WebsiteDist = Join-Path $RepoRoot "website\dist"

if (-not (Test-Path $Python)) { throw "Missing virtual environment: $Python" }
if (-not (Test-Path $Checkpoint)) { throw "Missing checkpoint: $Checkpoint" }
if (-not (Test-Path $WebsiteDist)) {
    throw "Website build is missing. Run npm install and npm run build inside the website directory."
}

$env:WIUT_DEMO_CHECKPOINT = $Checkpoint
$env:PYTHONPATH = $RepoRoot
Write-Host "Live demo: http://127.0.0.1:$Port"
Write-Host "Checkpoint: $Checkpoint"
& $Python -m uvicorn demo_api:app --host 127.0.0.1 --port $Port
