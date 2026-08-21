[CmdletBinding()]
param(
    [ValidateSet("waitress", "flask")]
    [string]$Mode = "waitress",
    [int]$Port = 5000
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$projectDir = $PSScriptRoot
$venvPython = Join-Path $projectDir ".venv\Scripts\python.exe"
$waitress = Join-Path $projectDir ".venv\Scripts\waitress-serve.exe"

if (-not (Test-Path -LiteralPath $venvPython)) {
    throw "Project .venv is missing. Run .\bootstrap_windows.ps1 first."
}

Push-Location $projectDir
try {
    if ($Mode -eq "waitress") {
        if (-not (Test-Path -LiteralPath $waitress)) {
            throw "Waitress is missing. Run .\bootstrap_windows.ps1 again."
        }

        Write-Host "Starting backend with Waitress at http://127.0.0.1:$Port"
        & $waitress --host=127.0.0.1 --port=$Port --call app:create_app
    }
    else {
        if ($Port -ne 5000) {
            Write-Warning "app.py development mode uses fixed port 5000; -Port is ignored."
        }

        Write-Warning "Flask development mode is for local debugging only."
        & $venvPython (Join-Path $projectDir "app.py")
    }
}
finally {
    Pop-Location
}
