$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$Artifact = Join-Path $Root "models\rentednoodle-v3\converted\Qwen3.8-27B-OrcaRouter-GSQ-RCO-IQ3_XXS-v2.1-vision-mtp.ninfer"
$Exe = Join-Path $Root "runtime-v3\engine\ninfer-serve.exe"

if (-not (Test-Path $Artifact)) {
    & (Join-Path $Root "scripts\convert-rentednoodle-v3.ps1")
}
if (-not (Test-Path $Exe)) {
    Write-Host "Native Ryan-gsq SM120a runtime is missing; building it now."
    & (Join-Path $Root "scripts\build-ryan-engine.ps1")
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path $Exe)) {
        throw "Native Ryan-gsq engine build did not produce $Exe"
    }
}

$Artifact = (Resolve-Path $Artifact).Path
$Exe = (Resolve-Path $Exe).Path
$Port = if ($env:NINFER_PORT) { [int]$env:NINFER_PORT } else { 8080 }
$Vision = if ($env:NINFER_VISION) { $env:NINFER_VISION -ne "0" } else { $true }
$Context = if ($env:NINFER_CONTEXT) { [int]$env:NINFER_CONTEXT } else { if ($Vision) { 98304 } else { 131072 } }
$HostCache = if ($env:NINFER_HOST_CACHE_MIB) { [int]$env:NINFER_HOST_CACHE_MIB } else { 1024 }
$Chunk = if ($env:NINFER_PREFILL_CHUNK) { [int]$env:NINFER_PREFILL_CHUNK } else { 1024 }
$Kv = if ($env:NINFER_KV_DTYPE) { $env:NINFER_KV_DTYPE } else { "rk8v4" }
$MemoryPolicy = if ($env:NINFER_MEMORY_POLICY) { $env:NINFER_MEMORY_POLICY } else { if ($Vision) { "default" } else { "strict" } }

$env:NINFER_PREFILL_ALIGN = "0"
Remove-Item Env:NINFER_PROMPT_FAST -ErrorAction SilentlyContinue
Remove-Item Env:CUDA_LAUNCH_BLOCKING -ErrorAction SilentlyContinue

$Args = @(
    $Artifact,
    "--host","127.0.0.1",
    "--port","$Port",
    "--model-id","qwen3.8-27b-rentednoodle-orcarouter",
    "--max-context","$Context",
    "--kv-capacity","$Context",
    "--max-concurrency","1",
    "--default-max-tokens","0",
    "--prefill-chunk","$Chunk",
    "--kv-dtype",$Kv,
    "--host-cache-mib","$HostCache",
    "--device-snapshot-slots","1",
    "--cuda-graph-allowance-mib","72",
    "--spec","mtp",
    "--draft-tokens","4",
    "--adaptive-mtp",
    "--ngram-draft-tokens","31",
    "--gdn-state-fp16",
    "--cuda-memory-policy",$MemoryPolicy,
    "--default-reasoning-effort","xhigh",
    "--preserve-thinking",
    "--temperature","1.0",
    "--top-p","0.95",
    "--top-k","20",
    "--min-p","0.0"
)
if ($Vision) { $Args += "--vision" }

try {
    $Gpu = & nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free --format=csv,noheader,nounits 2>$null
    if ($LASTEXITCODE -eq 0) { Write-Host "GPU before launch: $Gpu" }
} catch {}

Write-Host ""
Write-Host "Starting preserved-block NInfer v3 runtime"
Write-Host "  model:   $Artifact"
Write-Host "  engine:  $Exe"
Write-Host "  API:     http://127.0.0.1:$Port/v1"
Write-Host "  context: $Context"
Write-Host "  KV:      $Kv"
Write-Host "  vision:  $Vision"
Write-Host "  RAM cache: $HostCache MiB"
Write-Host ""

Push-Location (Split-Path -Parent $Exe)
try { & $Exe @Args } finally { Pop-Location }
