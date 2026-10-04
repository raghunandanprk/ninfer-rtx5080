$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$Engine = Join-Path $Root "engine"
$Overlay = Join-Path $Root "overlays\convert_rentednoodle.py"
$OverlayDest = Join-Path $Engine "tools\convert\qwen3_8_27b\convert_rentednoodle.py"
$Source = Join-Path $Root "models\rentednoodle\Qwen3.8-27B-OrcaRouter-GSQ-RCO-IQ3_XXS-v2.0.gguf"
$Donor = Join-Path $Root "models\qwen3_8_27b_gsq3.ninfer"
$Mmproj = Join-Path $Root "models\rentednoodle\mmproj\mmproj-Qwen3.8-27B-BF16.gguf"
$Frontend = Join-Path $Root "models\rentednoodle"
$OutDir = Join-Path $Root "models\rentednoodle"
$Output = Join-Path $OutDir "qwen3_8_27b_orcarouter_rentednoodle_gsqrco_iq3xxs.ninfer"

if (-not (Test-Path (Join-Path $Engine "CMakeLists.txt"))) {
    & (Join-Path $Root "scripts\bootstrap-source.ps1")
}

# Always refresh the project overlay after git pull, even when engine/ already exists.
if (-not (Test-Path $Overlay)) { throw "Missing converter overlay: $Overlay" }
Copy-Item $Overlay $OverlayDest -Force
Write-Host "Refreshed converter overlay: $OverlayDest"

if (-not (Test-Path $Source) -or -not (Test-Path $Mmproj)) {
    & (Join-Path $Root "scripts\download-rentednoodle.ps1")
}
if (-not (Test-Path $Donor)) {
    & (Join-Path $Root "scripts\download-model.ps1")
}

$Python = if ($env:NINFER_PYTHON) { $env:NINFER_PYTHON } else { "python" }

& $Python -c "import torch, numpy, safetensors, gguf; print('converter deps OK; torch=', torch.__version__, 'cuda=', torch.cuda.is_available())"
if ($LASTEXITCODE -ne 0) {
    throw "Missing converter Python dependencies. Install torch, numpy, safetensors and gguf, or set NINFER_PYTHON to a prepared Python executable."
}

New-Item -ItemType Directory -Force $OutDir | Out-Null
Push-Location $Engine
try {
    & $Python -m tools.convert.qwen3_8_27b.convert_rentednoodle `
        --gguf $Source `
        --donor-artifact $Donor `
        --mmproj $Mmproj `
        --frontend-dir $Frontend `
        --out $Output `
        --device cuda
    if ($LASTEXITCODE -ne 0) { throw "RentedNoodle NInfer conversion failed." }
} finally {
    Pop-Location
}

Write-Host "Converted artifact:"
Write-Host "  $Output"
Write-Host "Conversion report:"
Write-Host "  $Output.conversion.json"
