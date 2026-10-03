$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$Dir = Join-Path $Root "models\rentednoodle"
$File = "Qwen3.8-27B-OrcaRouter-GSQ-RCO-IQ3_XXS-v2.1.gguf"
$Model = Join-Path $Dir $File
$Revision = "59a3d12af8e41ddd518994ab8dd7cce8efca2252"
$Expected = "ab955b5083d9cdf0bf55c37acdcae359b78756c4544d97960c23d8fca98feb9b"
$Url = "https://huggingface.co/RentedNoodle/Qwen3.8-27B-OrcaRouter-GSQ-RCO-IQ3_XXS-Uncensored/resolve/$Revision/$File"

New-Item -ItemType Directory -Force $Dir | Out-Null

if (-not (Test-Path $Model)) {
    Write-Host "Downloading pinned RentedNoodle OrcaRouter GSQ-RCO IQ3_XXS v2.1..."
    & curl.exe -L -C - --fail --output $Model $Url
    if ($LASTEXITCODE -ne 0) { throw "RentedNoodle model download failed." }
} else {
    Write-Host "RentedNoodle GGUF already present; verifying SHA-256."
}

$Actual = (Get-FileHash -Algorithm SHA256 $Model).Hash.ToLowerInvariant()
if ($Actual -ne $Expected) {
    throw "SHA-256 mismatch. Expected $Expected, got $Actual"
}

Write-Host "RentedNoodle source verified:"
Write-Host "  $Model"
