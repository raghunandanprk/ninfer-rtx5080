$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$Artifact = if ($env:NINFER_ARTIFACT) { $env:NINFER_ARTIFACT } else { Join-Path $Root "models\rentednoodle\qwen3_8_27b_orcarouter_rentednoodle_gsqrco_iq3xxs.ninfer" }
$Exe = Join-Path $Root "engine\build-windows\apps\ninfer-serve.exe"

if (-not (Test-Path $Artifact)) {
    Write-Host "RentedNoodle NInfer artifact is missing; starting conversion."
    & (Join-Path $Root "scripts\convert-rentednoodle.ps1")
}
if (-not (Test-Path $Exe)) {
    Write-Host "Native ninfer-serve.exe is missing; building it now."
    & (Join-Path $Root "scripts\build-native-windows.ps1")
}

$Artifact = (Resolve-Path $Artifact).Path
$Exe = (Resolve-Path $Exe).Path
$Port = if ($env:NINFER_PORT) { [int]$env:NINFER_PORT } else { 8080 }
$VisionEnabled = if ($env:NINFER_VISION) { $env:NINFER_VISION -ne "0" } else { $true }
$DefaultContext = if ($VisionEnabled) { 65536 } else { 100000 }
$Context = if ($env:NINFER_CONTEXT) { [int]$env:NINFER_CONTEXT } else { $DefaultContext }
$HostKvMiB = if ($env:NINFER_HOST_KV_MIB) { [int]$env:NINFER_HOST_KV_MIB } else { 512 }
$VisionMax = if ($env:NINFER_VISION_MAX_TOKENS) { [int]$env:NINFER_VISION_MAX_TOKENS } else { 1792 }
$Prefill = if ($env:NINFER_PREFILL_CHUNK) { [int]$env:NINFER_PREFILL_CHUNK } else { 2688 }
$ApiKey = $env:NINFER_API_KEY

# This is intentionally much smaller than NInfer's 8192 MiB Host-KV default.
# Raise NINFER_HOST_KV_MIB only if you need deep host-side prefix/session retention.
$Args = @(
    $Artifact,
    "--host","127.0.0.1",
    "--port","$Port",
    "--model-id","qwen3.8-27b-rentednoodle",
    "--max-context","$Context",
    "--kv-capacity","$Context",
    "--kv-dtype","rk4v4-e8",
    "--max-concurrency","1",
    "--max-pending-requests","16",
    "--prefill-chunk","$Prefill",
    "--host-kv-mib","$HostKvMiB",
    "--host-state-slots","2",
    "--max-private-continuations","2",
    "--max-shared-prefixes","2",
    "--max-long-anchors-per-continuation","1",
    "--auto-long-anchors","1",
    "--media-cache-mib","256",
    "--media-live-mib","768",
    "--spec","mtp",
    "--draft-tokens","3",
    "--lm-head-draft",
    "--embedding-host",
    "--preserve-thinking"
)

if ($VisionEnabled) {
    $Args += @("--vision","--vision-max-tokens","$VisionMax")
}
if ($env:NINFER_CUDA_GRAPH -eq "0") {
    $Args += "--no-cuda-graph"
}
if ($ApiKey) {
    $Args += @("--api-key",$ApiKey)
}

# Show current VRAM before allocating a near-capacity 16 GB model.
try {
    $Gpu = & nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free --format=csv,noheader,nounits 2>$null
    if ($LASTEXITCODE -eq 0) { Write-Host "GPU before launch: $Gpu" }
} catch {}

Write-Host ""
Write-Host "Starting native Windows NInfer (no Docker / no WSL)"
Write-Host "  artifact: $Artifact"
Write-Host "  API:      http://127.0.0.1:$Port/v1"
Write-Host "  context:  $Context"
Write-Host "  vision:   $VisionEnabled"
Write-Host "  host KV:  $HostKvMiB MiB"
Write-Host ""

& $Exe @Args
