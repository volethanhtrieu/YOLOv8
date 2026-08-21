[CmdletBinding()]
param(
    [string]$PythonVersion = "3.14",
    [switch]$SkipVerify
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$projectDir = $PSScriptRoot
$venvDir = Join-Path $projectDir ".venv"
$venvPython = Join-Path $venvDir "Scripts\python.exe"
$requirements = Join-Path $projectDir "requirements-release.txt"

if (-not (Test-Path -LiteralPath $requirements)) {
    throw "Missing requirements file: $requirements"
}

Push-Location $projectDir
try {
    if (-not (Test-Path -LiteralPath $venvPython)) {
        $pyLauncher = Get-Command "py.exe" -ErrorAction SilentlyContinue
        if ($null -ne $pyLauncher) {
            Write-Host "Creating .venv with Python $PythonVersion..."
            & $pyLauncher.Source "-$PythonVersion" -m venv $venvDir
        }
        else {
            $pythonCommand = Get-Command "python.exe" -ErrorAction SilentlyContinue
            if ($null -eq $pythonCommand) {
                throw "Python was not found. Install 64-bit Python $PythonVersion and retry."
            }

            Write-Host "py.exe was not found; creating .venv with $($pythonCommand.Source)."
            & $pythonCommand.Source -m venv $venvDir
        }
    }
    else {
        Write-Host "Reusing existing project environment: $venvDir"
    }

    & $venvPython -m pip install --upgrade pip
    & $venvPython -m pip install -r $requirements

    if (-not $SkipVerify) {
        & $venvPython (Join-Path $projectDir "verify_install.py")
    }

    Write-Host ""
    Write-Host "Bootstrap complete."
    Write-Host "Backend : .\run_backend.ps1"
    Write-Host "Dashboard: .\run_dashboard.ps1"
}
finally {
    Pop-Location
}
