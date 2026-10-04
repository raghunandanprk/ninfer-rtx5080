$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$V3 = Join-Path $Root ".deps\ninfer-v3"
$Venv = Join-Path $Root ".deps\convert-v3"
$Input = Join-Path $Root "models\rentednoodle-v3"
$Meta = Join-Path $Input "metadata"
$Gguf = Join-Path $Input "Qwen3.8-27B-OrcaRouter-GSQ-RCO-IQ3_XXS-v2.1.gguf"
$Vision = Join-Path $Input "mmproj\mmproj-Qwen3.8-27B-BF16.gguf"
$OutDir = Join-Path $Root "models\rentednoodle-v3\converted"
$Out = Join-Path $OutDir "Qwen3.8-27B-OrcaRouter-GSQ-RCO-IQ3_XXS-v2.1-vision-mtp.ninfer"

if (-not (Test-Path (Join-Path $V3 "tools\convert\gguf_blocks.py"))) {
    & (Join-Path $Root "scripts\bootstrap-v3.ps1")
}
if (-not (Test-Path $Gguf) -or -not (Test-Path $Vision)) {
    & (Join-Path $Root "scripts\download-rentednoodle-v3.ps1")
}

# The v3 block-preserving text/MTP route is CPU-only. gguf is additionally
# installed because the Vision importer reads the separate mmproj GGUF.
$Python = Join-Path $Venv "Scripts\python.exe"
if (-not (Test-Path $Python)) {
    $Launcher = Get-Command py.exe -ErrorAction SilentlyContinue
    if ($Launcher) {
        & py.exe -3.11 -m venv $Venv
    } else {
        & python.exe -m venv $Venv
    }
    if ($LASTEXITCODE -ne 0) { throw "Failed to create converter Python environment." }
    & $Python -m pip install --upgrade pip
    & $Python -m pip install numpy huggingface_hub gguf
    & $Python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
    if ($LASTEXITCODE -ne 0) { throw "Failed to install v3 converter dependencies." }
}

foreach ($Name in @("config.json","tokenizer.json","tokenizer_config.json","chat_template.jinja","generation_config.json","preprocessor_config.json","video_preprocessor_config.json")) {
    if (-not (Test-Path (Join-Path $Meta $Name))) { throw "Missing metadata resource: $Name" }
}

New-Item -ItemType Directory -Force $OutDir | Out-Null
if (Test-Path $Out) {
    Write-Host "v3 artifact already exists:"
    Write-Host "  $Out"
    exit 0
}

Push-Location $V3
try {
    & $Python -u -m tools.convert `
        --model $Meta `
        --recipe qwen3_8_27b_gguf `
        --source "gguf=$Gguf" `
        --source "vision=$Vision" `
        --components text,vision,mtp `
        --proposal `
        --device cpu `
        --rows-per-chunk 512 `
        --name "qwen3.8-27b-rentednoodle-orcarouter-v2.1" `
        --out $Out
    if ($LASTEXITCODE -ne 0) { throw "NInfer v3 preserved-block conversion failed." }
} finally { Pop-Location }

Write-Host ""
Write-Host "NInfer v3 artifact created without requantizing the GSQ/RCO trunk:"
Write-Host "  $Out"
Write-Host "  report: $Out.conversion.json"
