$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$Dir = Join-Path $Root "models\rentednoodle-v3"
$Meta = Join-Path $Dir "metadata"
$MmprojDir = Join-Path $Dir "mmproj"
$HfRepo = "RentedNoodle/Qwen3.8-27B-OrcaRouter-GSQ-RCO-IQ3_XXS-Uncensored"
$ModelFile = "Qwen3.8-27B-OrcaRouter-GSQ-RCO-IQ3_XXS-v2.1.gguf"
$ModelSha = "ab955b5083d9cdf0bf55c37acdcae359b78756c4544d97960c23d8fca98feb9b"
$MmprojFile = "mmproj-Qwen3.8-27B-BF16.gguf"

New-Item -ItemType Directory -Force $Dir,$Meta,$MmprojDir | Out-Null

function Download-Hf([string]$Repo,[string]$Remote,[string]$Local) {
    if (Test-Path $Local) { Write-Host "Already present: $Local"; return }
    $Url = "https://huggingface.co/$Repo/resolve/main/$Remote"
    Write-Host "Downloading $Repo/$Remote..."
    & curl.exe -L -C - --fail --output $Local $Url
    if ($LASTEXITCODE -ne 0) { throw "Download failed: $Repo/$Remote" }
}

$Model = Join-Path $Dir $ModelFile
$Mmproj = Join-Path $MmprojDir $MmprojFile
Download-Hf $HfRepo $ModelFile $Model
Download-Hf $HfRepo "mmproj/$MmprojFile" $Mmproj

# RentedNoodle metadata and Froggeric template.
foreach ($Name in @("config.json","tokenizer.json","tokenizer_config.json","preprocessor_config.json")) {
    Download-Hf $HfRepo $Name (Join-Path $Meta $Name)
}
$Froggeric = Join-Path $Meta "chat_template.jinja"
Download-Hf $HfRepo "froggeric-qwen3.8-tool-use.jinja" $Froggeric

# The converter requires these two frontend resources; RentedNoodle does not ship them.
# They are architecture/frontend metadata, so take them from the official Qwen3.8-27B repo.
Download-Hf "Qwen/Qwen3.8-27B" "generation_config.json" (Join-Path $Meta "generation_config.json")
Download-Hf "Qwen/Qwen3.8-27B" "video_preprocessor_config.json" (Join-Path $Meta "video_preprocessor_config.json")

$Actual = (Get-FileHash -Algorithm SHA256 $Model).Hash.ToLowerInvariant()
if ($Actual -ne $ModelSha) { throw "RentedNoodle v2.1 SHA-256 mismatch. Expected $ModelSha, got $Actual" }

Write-Host "RentedNoodle v3 conversion inputs ready:"
Write-Host "  model:  $Model"
Write-Host "  vision: $Mmproj"
Write-Host "  meta:   $Meta"
