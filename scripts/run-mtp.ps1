$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$DefaultModel = Join-Path $Root "models\qwen3_8_27b_gsq3.ninfer"
$Model = if ($env:NINFER_ARTIFACT) { $env:NINFER_ARTIFACT } else { $DefaultModel }
$Image = "ninfer-rtx5080:gsq3"
$Port = if ($env:NINFER_PORT) { $env:NINFER_PORT } else { "8080" }
$Context = if ($env:NINFER_CONTEXT) { $env:NINFER_CONTEXT } else { "102400" }
$Vision = if ($env:NINFER_VISION -eq "0") { @() } else { @("--vision") }

if (-not (Test-Path $Model)) {
    if ($env:NINFER_ARTIFACT) { throw "NINFER_ARTIFACT does not exist: $Model" }
    & (Join-Path $Root "scripts\download-model.ps1")
}

docker image inspect $Image *> $null
if ($LASTEXITCODE -ne 0) { & (Join-Path $Root "scripts\build-docker.ps1") }

$Model = (Resolve-Path $Model).Path
$ModelDir = Split-Path -Parent $Model
$ModelName = Split-Path -Leaf $Model
Write-Host "Serving $ModelName with MTP3 at context=$Context on http://127.0.0.1:$Port/v1"

$Args = @(
  "run","--rm","--gpus","all",
  "-p","127.0.0.1:${Port}:8080",
  "-v","${ModelDir}:/models:ro",
  $Image,
  "ninfer-serve","/models/$ModelName",
  "--host","0.0.0.0","--port","8080",
  "--max-context",$Context,"--kv-capacity",$Context,"--kv-dtype","rk4v4-e8",
  "--max-concurrency","1","--max-pending-requests","16",
  "--prefill-chunk","2688","--host-kv-mib","4096",
  "--spec","mtp","--draft-tokens","3","--lm-head-draft","--preserve-thinking"
) + $Vision

& docker @Args
