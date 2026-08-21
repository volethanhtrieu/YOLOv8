[CmdletBinding()]
param(
    [int]$Port = 8501
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$projectDir = $PSScriptRoot
$streamlit = Join-Path $projectDir ".venv\Scripts\streamlit.exe"

if (-not (Test-Path -LiteralPath $streamlit)) {
    throw "Project .venv is missing or Streamlit is not installed. Run .\bootstrap_windows.ps1 first."
}

Push-Location $projectDir
try {
    Write-Host "Starting dashboard at http://127.0.0.1:$Port"
    & $streamlit run (Join-Path $projectDir "dashboard.py") --server.address 127.0.0.1 --server.port $Port
}
finally {
    Pop-Location
}
