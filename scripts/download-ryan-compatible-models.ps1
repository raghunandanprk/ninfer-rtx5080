param(
    [string]$Destination = "",
    [string]$Repo = "raghualgt/Qwen3.8-27B-OrcaRouter-GSQ-RCO-IQ3_XXS-NInfer",
    [string]$Revision = "main",
    [ValidateSet("all", "text", "mtp", "dflash2", "vision")][string]$Variant = "all"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
if (-not $Destination) { $Destination = Join-Path $Root "models\rentednoodle-native-v3" }
$Destination = [IO.Path]::GetFullPath($Destination)

$Python = if (Test-Path (Join-Path $Root ".deps\build-venv\Scripts\python.exe")) {
    Join-Path $Root ".deps\build-venv\Scripts\python.exe"
} elseif (Get-Command python.exe -ErrorAction SilentlyContinue) {
    (Get-Command python.exe).Source
} elseif (Get-Command py.exe -ErrorAction SilentlyContinue) {
    (Get-Command py.exe).Source
} else {
    throw "Python was not found. Please install Python or ensure it is available in your PATH."
}

$HfCmd = Get-Command hf.exe -ErrorAction SilentlyContinue
$Hf = if ($HfCmd) { $HfCmd.Source } else { "" }

if ($Repo -eq "2beng2/Qwen3.8-27B-OrcaRouter-GSQ-RCO-IQ3_XXS-NInfer" -and $Revision -eq "main") {
    $Revision = "d19429861f973f87c6331af120470d22942430ca"
}

$AllAvailable = [ordered]@{
    "qwen3.8-27b-orcarouter-iq3-xxs-mtp-only.ninfer" = "947d2c5197d72518eb1a749e7d24b22582249880f6437c76b7c1289ef8a72b08"
    "qwen3.8-27b-orcarouter-iq3-xxs-mtp-dflash2.ninfer" = "a2cf5282288289d62dfd19f2a736909c06c809ed5c476e7435e82ff7b80344ed"
    "qwen3.8-27b-orcarouter-iq3-xxs-vision-mtp.ninfer" = "63aa158c9f749088a4cab31b953ef6179d58cabce73e1ff389c15fd262086b07"
}

$Files = [ordered]@{}
if ($Variant -eq "all") {
    if ($Repo -eq "2beng2/Qwen3.8-27B-OrcaRouter-GSQ-RCO-IQ3_XXS-NInfer") {
        # 2beng2 repo does not host the vision artifact
        $Files["qwen3.8-27b-orcarouter-iq3-xxs-mtp-only.ninfer"] = $AllAvailable["qwen3.8-27b-orcarouter-iq3-xxs-mtp-only.ninfer"]
        $Files["qwen3.8-27b-orcarouter-iq3-xxs-mtp-dflash2.ninfer"] = $AllAvailable["qwen3.8-27b-orcarouter-iq3-xxs-mtp-dflash2.ninfer"]
    } else {
        foreach ($k in $AllAvailable.Keys) { $Files[$k] = $AllAvailable[$k] }
    }
} elseif ($Variant -eq "text") {
    $Files["qwen3.8-27b-orcarouter-iq3-xxs-mtp-only.ninfer"] = $AllAvailable["qwen3.8-27b-orcarouter-iq3-xxs-mtp-only.ninfer"]
    $Files["qwen3.8-27b-orcarouter-iq3-xxs-mtp-dflash2.ninfer"] = $AllAvailable["qwen3.8-27b-orcarouter-iq3-xxs-mtp-dflash2.ninfer"]
} elseif ($Variant -eq "mtp") {
    $Files["qwen3.8-27b-orcarouter-iq3-xxs-mtp-only.ninfer"] = $AllAvailable["qwen3.8-27b-orcarouter-iq3-xxs-mtp-only.ninfer"]
} elseif ($Variant -eq "dflash2") {
    $Files["qwen3.8-27b-orcarouter-iq3-xxs-mtp-dflash2.ninfer"] = $AllAvailable["qwen3.8-27b-orcarouter-iq3-xxs-mtp-dflash2.ninfer"]
} elseif ($Variant -eq "vision") {
    if ($Repo -eq "2beng2/Qwen3.8-27B-OrcaRouter-GSQ-RCO-IQ3_XXS-NInfer") {
        throw "The vision artifact is only available in 'raghualgt/Qwen3.8-27B-OrcaRouter-GSQ-RCO-IQ3_XXS-NInfer'."
    }
    $Files["qwen3.8-27b-orcarouter-iq3-xxs-vision-mtp.ninfer"] = $AllAvailable["qwen3.8-27b-orcarouter-iq3-xxs-vision-mtp.ninfer"]
}
$env:HF_HUB_CACHE = Join-Path $Root ".deps\hf-cache"
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
New-Item -ItemType Directory -Force $Destination | Out-Null
function Invoke-Download {
    param([string[]]$Arguments)
    $savedPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = "Continue"
        if ($Hf) {
            & $Hf @Arguments 2>&1 | ForEach-Object { Write-Host $_ }
        } else {
            & $Python -m huggingface_hub.cli.core @Arguments 2>&1 | ForEach-Object { Write-Host $_ }
        }
        $downloadExitCode = $LASTEXITCODE
    } finally { $ErrorActionPreference = $savedPreference }
    if ($downloadExitCode -ne 0) { throw "Compatible-model download failed; hf can resume it on rerun." }
}
foreach ($Name in $Files.Keys) {
    $Url = "https://huggingface.co/$Repo/resolve/$Revision/$Name"
    & $Python (Join-Path $Root "scripts\inspect-ryan-model.py") $Url
    if ($LASTEXITCODE -ne 0) { throw "Pinned source rejected the remote artifact: $Name" }
}
Invoke-Download -Arguments (@("download", $Repo) + @($Files.Keys) + @("README.md", "NOTICE", "LICENSE", "--revision", $Revision, "--local-dir", $Destination, "--max-workers", "1"))
$Checks = @()
foreach ($Name in $Files.Keys) {
    $Artifact = Join-Path $Destination $Name
    $Report = "$Artifact.validation.json"
    & $Python (Join-Path $Root "scripts\inspect-ryan-model.py") $Artifact --sha256 $Files[$Name] --output $Report
    if ($LASTEXITCODE -ne 0) { throw "Downloaded model validation failed: $Name" }
    $Checks += Get-Content -LiteralPath $Report -Raw | ConvertFrom-Json
}
[ordered]@{
    repository = $Repo
    revision = $Revision
    source_gguf_sha256 = "41ad7dfb3f4397d626408a96e88a46c5964e88bd6c4240c191e1131af92ea8cd"
    source_gguf_release = "RentedNoodle OrcaRouter v2.0"
    local_weight_changes = $false
    downloaded_utc = [DateTime]::UtcNow.ToString("o")
    artifacts = $Checks
} | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $Destination "download-manifest.json") -Encoding UTF8
Write-Host "Compatible native v3 artifacts ($($Files.Count) file(s)) are verified in $Destination"
