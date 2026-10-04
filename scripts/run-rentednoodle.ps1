$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$Pipeline = if ($env:NINFER_PIPELINE) { $env:NINFER_PIPELINE.ToLowerInvariant() } else { "v3" }

switch ($Pipeline) {
    "v3" {
        & (Join-Path $Root "scripts\run-rentednoodle-v3.ps1")
        exit $LASTEXITCODE
    }
    "legacy" {
        Write-Warning "Using legacy requantized RentedNoodle path."
        & (Join-Path $Root "scripts\run-rentednoodle-native.ps1")
        exit $LASTEXITCODE
    }
    "legacy-docker" {
        Write-Warning "Using legacy Docker/WSL path."
        $Artifact = Join-Path $Root "models\rentednoodle\qwen3_8_27b_orcarouter_rentednoodle_gsqrco_iq3xxs.ninfer"
        if (-not (Test-Path $Artifact)) { & (Join-Path $Root "scripts\convert-rentednoodle.ps1") }
        $env:NINFER_ARTIFACT = $Artifact
        & (Join-Path $Root "scripts\run-mtp.ps1")
        exit $LASTEXITCODE
    }
    default { throw "Unknown NINFER_PIPELINE=$Pipeline. Use v3, legacy, or legacy-docker." }
}
