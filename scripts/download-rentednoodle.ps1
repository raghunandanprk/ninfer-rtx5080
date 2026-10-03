$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$Dir = Join-Path $Root "models\rentednoodle"
$MmprojDir = Join-Path $Dir "mmproj"
$Repo = "RentedNoodle/Qwen3.8-27B-OrcaRouter-GSQ-RCO-IQ3_XXS-Uncensored"
$Revision = "main"
$File = "Qwen3.8-27B-OrcaRouter-GSQ-RCO-IQ3_XXS-v2.0.gguf"
$Model = Join-Path $Dir $File
$ModelSha = "41ad7dfb3f4397d626408a96e88a46c5964e88bd6c4240c191e1131af92ea8cd"
$MmprojFile = "mmproj-Qwen3.8-27B-BF16.gguf"
$Mmproj = Join-Path $MmprojDir $MmprojFile

New-Item -ItemType Directory -Force $Dir | Out-Null
New-Item -ItemType Directory -Force $MmprojDir | Out-Null

function Download-RepoFile([string]$Remote, [string]$Local) {
    if (Test-Path $Local) {
        Write-Host "Already present: $Local"
        return
    }
    $Url = "https://huggingface.co/$Repo/resolve/$Revision/$Remote"
    Write-Host "Downloading $Remote..."
    & curl.exe -L -C - --fail --output $Local $Url
    if ($LASTEXITCODE -ne 0) { throw "Download failed: $Remote" }
}

Download-RepoFile $File $Model
Download-RepoFile "mmproj/$MmprojFile" $Mmproj
Download-RepoFile "froggeric-qwen3.8-tool-use.jinja" (Join-Path $Dir "froggeric-qwen3.8-tool-use.jinja")
Download-RepoFile "config.json" (Join-Path $Dir "config.json")
Download-RepoFile "tokenizer.json" (Join-Path $Dir "tokenizer.json")
Download-RepoFile "tokenizer_config.json" (Join-Path $Dir "tokenizer_config.json")
Download-RepoFile "preprocessor_config.json" (Join-Path $Dir "preprocessor_config.json")
Download-RepoFile "SHA256SUMS.txt" (Join-Path $Dir "SHA256SUMS.txt")

$Actual = (Get-FileHash -Algorithm SHA256 $Model).Hash.ToLowerInvariant()
if ($Actual -ne $ModelSha) {
    throw "Main GGUF SHA-256 mismatch. Expected $ModelSha, got $Actual"
}

# Verify the projector against the repository-provided SHA256SUMS when present.
$Sums = Join-Path $Dir "SHA256SUMS.txt"
if (Test-Path $Sums) {
    $Line = Get-Content $Sums | Where-Object { $_ -match [regex]::Escape("mmproj/$MmprojFile") -or $_ -match [regex]::Escape($MmprojFile) } | Select-Object -First 1
    if ($Line -and $Line -match "^([0-9a-fA-F]{64})\s+\*?.*$([regex]::Escape($MmprojFile))") {
        $ExpectedMmproj = $Matches[1].ToLowerInvariant()
        $ActualMmproj = (Get-FileHash -Algorithm SHA256 $Mmproj).Hash.ToLowerInvariant()
        if ($ActualMmproj -ne $ExpectedMmproj) {
            throw "BF16 mmproj SHA-256 mismatch. Expected $ExpectedMmproj, got $ActualMmproj"
        }
        Write-Host "BF16 mmproj verified: $ActualMmproj"
    } else {
        Write-Warning "Could not parse BF16 mmproj hash from SHA256SUMS.txt; file was downloaded but not hash-verified."
    }
}

Write-Host "RentedNoodle source bundle ready:"
Write-Host "  model:  $Model"
Write-Host "  vision: $Mmproj"
