$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$Artifact = Join-Path $Root "models\rentednoodle\qwen3_8_27b_orcarouter_rentednoodle_gsqrco_iq3xxs.ninfer"

if (-not (Test-Path $Artifact)) {
    Write-Host "RentedNoodle NInfer artifact is missing; starting conversion."
    & (Join-Path $Root "scripts\convert-rentednoodle.ps1")
}

$env:NINFER_ARTIFACT = $Artifact
if (-not $env:NINFER_CONTEXT) { $env:NINFER_CONTEXT = "102400" }
if (-not $env:NINFER_VISION) { $env:NINFER_VISION = "1" }
& (Join-Path $Root "scripts\run-mtp.ps1")
