$ErrorActionPreference = "Stop"

if (-not (Test-Path ".venv")) {
    py -3.11 -m venv .venv
}

& .\.venv\Scripts\python.exe -m pip install --upgrade pip
& .\.venv\Scripts\python.exe -m pip install -r requirements-data.txt

$chvg = "data\raw\chvg.zip"

if (-not (Test-Path $chvg)) {
    throw "Missing $chvg. Place the CHVG v1 YOLOv8 ZIP at this path."
}

& .\.venv\Scripts\python.exe scripts\data\prepare_chvg.py `
    --chvg-zip $chvg `
    --workspace data `
    --reports reports\generated\chvg `
    --seed 42

if ($LASTEXITCODE -ne 0) {
    throw "CHVG validation failed. Open reports\generated\chvg\validation_report.csv."
}

foreach ($split in @("train", "val", "test")) {
    & .\.venv\Scripts\python.exe scripts\data\preview_labels.py `
        --yaml configs\chvg5.yaml `
        --split $split `
        --count 20 `
        --output "reports\generated\chvg\previews\$split"
}

Write-Host "CHVG preparation completed. Review reports\generated\chvg before handoff."
