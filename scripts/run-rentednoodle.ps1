$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$Runtime = if ($env:NINFER_RUNTIME) { $env:NINFER_RUNTIME.ToLowerInvariant() } else { "native" }

if ($Runtime -eq "native") {
    & (Join-Path $Root "scripts\run-rentednoodle-native.ps1")
    exit $LASTEXITCODE
}

if ($Runtime -ne "docker") {
    throw "Unsupported NINFER_RUNTIME=$Runtime. Use native or docker."
}

# Explicit Docker fallback.
$Artifact = Join-Path $Root "models\rentednoodle\qwen3_8_27b_orcarouter_rentednoodle_gsqrco_iq3xxs.ninfer"
if (-not (Test-Path $Artifact)) {
    & (Join-Path $Root "scripts\convert-rentednoodle.ps1")
}
$env:NINFER_ARTIFACT = $Artifact
if (-not $env:NINFER_CONTEXT) { $env:NINFER_CONTEXT = "65536" }
if (-not $env:NINFER_VISION) { $env:NINFER_VISION = "1" }
& (Join-Path $Root "scripts\run-mtp.ps1")
