$ErrorActionPreference = "Stop"

if (-not (Test-Path ".venv")) {
    py -3.11 -m venv .venv
}

& .\.venv\Scripts\python.exe -m pip install --upgrade pip
& .\.venv\Scripts\python.exe -m pip install -r requirements-data.txt

$chvg = "data\raw\chvg.zip"
$shel5k = "data\raw\shel5k.zip"

if (-not (Test-Path $chvg)) {
    throw "Missing $chvg. Download CHVG v1 as YOLOv8 ZIP and rename it to chvg.zip."
}

if (-not (Test-Path $shel5k)) {
    throw "Missing $shel5k. Download SHEL5K Version 4 and rename it to shel5k.zip."
}

& .\.venv\Scripts\python.exe scripts\data\prepare_dataset.py `
    --chvg-zip $chvg `
    --shel5k-zip $shel5k `
    --workspace data `
    --class-map configs\class_map.yaml `
    --seed 42

if ($LASTEXITCODE -ne 0) {
    throw "Dataset validation failed. Open reports\generated\validation_report.csv."
}

& .\.venv\Scripts\python.exe scripts\data\preview_labels.py `
    --yaml configs\common3.yaml `
    --count 50 `
    --output reports\generated\common3_preview

& .\.venv\Scripts\python.exe scripts\data\preview_labels.py `
    --yaml configs\ppe5.yaml `
    --count 50 `
    --output reports\generated\ppe5_preview

Write-Host "Dataset preparation completed. Review reports\generated before handoff."

