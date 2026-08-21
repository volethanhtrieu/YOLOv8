[CmdletBinding()]
param(
    [string]$Video = ".\videos\test.mp4",
    [int]$MaxFrames = 50,
    [string]$ExperimentName = ("ablation_" + (Get-Date -Format "yyyyMMdd_HHmmss"))
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$projectDir = $PSScriptRoot
$python = Join-Path $projectDir ".venv\Scripts\python.exe"
$runner = Join-Path $projectDir "run_pipeline_safe.py"
$evaluator = Join-Path $projectDir "evaluate_ablation.py"
$runsRoot = Join-Path $projectDir "outputs\runs"
$reportRoot = Join-Path $projectDir ("outputs\ablation\" + $ExperimentName)

if (-not (Test-Path -LiteralPath $python)) {
    throw "Project .venv is missing. Run .\bootstrap_windows.ps1 first."
}

Push-Location $projectDir
try {
    $videoPath = (Resolve-Path -LiteralPath $Video).Path

    $sourceName = $ExperimentName + "_source"
    $byteTrackName = $ExperimentName + "_bytetrack"
    $offName = $ExperimentName + "_tracking_off"

    $sourceRoot = Join-Path $runsRoot $sourceName
    $byteTrackRoot = Join-Path $runsRoot $byteTrackName
    $offRoot = Join-Path $runsRoot $offName

    foreach ($target in @($sourceRoot, $byteTrackRoot, $offRoot, $reportRoot)) {
        if (Test-Path -LiteralPath $target) {
            throw "Ablation target already exists: $target"
        }
    }

    Write-Host "[1/4] Run YOLO once and create detections.csv..."
    & $python $runner `
        --video $videoPath `
        --max-frames $MaxFrames `
        --run-name $sourceName `
        --tracking-mode bytetrack
    if ($LASTEXITCODE -ne 0) {
        throw "Source detection run failed with exit code $LASTEXITCODE."
    }

    $cachePath = Join-Path $sourceRoot "tracking\detections.csv"
    if (-not (Test-Path -LiteralPath $cachePath)) {
        throw "Detection cache was not created: $cachePath"
    }

    Write-Host "[2/4] Replay identical detections with ByteTrack..."
    & $python $runner `
        --video $videoPath `
        --max-frames $MaxFrames `
        --run-name $byteTrackName `
        --tracking-mode bytetrack `
        --detections-cache $cachePath
    if ($LASTEXITCODE -ne 0) {
        throw "ByteTrack replay failed with exit code $LASTEXITCODE."
    }

    Write-Host "[3/4] Replay identical detections with tracking off..."
    & $python $runner `
        --video $videoPath `
        --max-frames $MaxFrames `
        --run-name $offName `
        --tracking-mode off `
        --detections-cache $cachePath
    if ($LASTEXITCODE -ne 0) {
        throw "Tracking-off replay failed with exit code $LASTEXITCODE."
    }

    Write-Host "[4/4] Build proxy-only ablation report..."
    & $python $evaluator `
        --detection-only-run $offRoot `
        --bytetrack-run $byteTrackRoot `
        --output-dir $reportRoot
    if ($LASTEXITCODE -ne 0) {
        throw "Ablation evaluation failed with exit code $LASTEXITCODE."
    }

    Write-Host ""
    Write-Host "ABLATION COMPLETE"
    Write-Host "Source cache : $sourceRoot"
    Write-Host "ByteTrack    : $byteTrackRoot"
    Write-Host "Tracking off : $offRoot"
    Write-Host "Report       : $reportRoot\report.md"
    Write-Host ""
    Write-Host "No ablation run was published."
}
finally {
    Pop-Location
}
