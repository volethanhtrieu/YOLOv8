$ErrorActionPreference = "Stop"

$python = ".\.venv\Scripts\python.exe"
$baseline = "data\processed\chvg5"
$experiment = "data\processed\chvg5_noise_experiment"
$reports = "reports\generated\chvg_noise"

if (-not (Test-Path $python)) {
    throw "Missing $python. Run run_prepare_chvg.ps1 first."
}

if (-not (Test-Path $baseline)) {
    throw "Missing $baseline. Prepare the baseline CHVG5 dataset first."
}

if (Test-Path $experiment) {
    throw "$experiment already exists. Keep it or rename it before creating another experiment."
}

Copy-Item -Path $baseline -Destination $experiment -Recurse

& $python scripts\data\augment_train.py `
    --dataset $experiment `
    --fraction 0.25 `
    --seed 42 `
    --report-dir $reports

& $python scripts\data\validate_noise_experiment.py `
    --baseline $baseline `
    --experiment $experiment `
    --manifest "$reports\augmentation_manifest.csv" `
    --report-dir $reports

if ($LASTEXITCODE -ne 0) {
    throw "Noise experiment validation failed. Open $reports\augmentation_validation.csv."
}

& $python scripts\data\preview_labels.py `
    --yaml "$experiment\chvg5.yaml" `
    --split train `
    --stem-contains "__aug_" `
    --count 40 `
    --seed 42 `
    --output "$reports\previews"

Write-Host "CHVG noise experiment completed. Review $reports before handoff."
