$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$Model = Join-Path $Root "models\qwen3_8_27b_gsq3.ninfer"
$Image = "ninfer-rtx5080:gsq3"
$Port = if ($env:NINFER_PORT) { $env:NINFER_PORT } else { "8080" }
$VisionEnabled = $env:NINFER_VISION -eq "1"
$DefaultContext = if ($VisionEnabled) { "65536" } else { "100000" }
$Context = if ($env:NINFER_CONTEXT) { $env:NINFER_CONTEXT } else { $DefaultContext }
$Vision = if ($VisionEnabled) { @("--vision") } else { @() }
if (-not (Test-Path $Model)) { & (Join-Path $Root "scripts\download-model.ps1") }
docker image inspect $Image *> $null
if ($LASTEXITCODE -ne 0) { & (Join-Path $Root "scripts\build-docker.ps1") }
$ModelDir = Split-Path -Parent $Model
Write-Host "Serving GSQ3 DFlash2 K=7 at context=$Context on http://127.0.0.1:$Port/v1"
$Args = @(
  "run","--rm","--gpus","all",
  "-p","127.0.0.1:$Port:8080",
  "-v","$ModelDir:/models:ro",
  $Image,
  "ninfer-serve","/models/qwen3_8_27b_gsq3.ninfer",
  "--host","0.0.0.0","--port","8080",
  "--max-context",$Context,"--kv-capacity",$Context,"--kv-dtype","rk4v4-e8",
  "--max-concurrency","1","--max-pending-requests","16",
  "--prefill-chunk","1024","--host-kv-mib","4096",
  "--spec","dflash2","--draft-tokens","7","--lm-head-draft"
) + $Vision
& docker @Args
