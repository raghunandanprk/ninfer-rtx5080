param(
    [string]$Model = "",
    [ValidateSet("none", "mtp", "dflash2")][string]$Spec = "mtp"
)

$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$FileName = if ($Spec -eq "dflash2") { "qwen3.8-27b-orcarouter-iq3-xxs-mtp-dflash2.ninfer" } else { "qwen3.8-27b-orcarouter-iq3-xxs-mtp-only.ninfer" }
$Artifact = if ($Model) { $Model } elseif ($env:NINFER_ARTIFACT) { $env:NINFER_ARTIFACT } else { "" }
if (-not $Artifact) {
    foreach ($Candidate in @((Join-Path $Root "models\rentednoodle-native-v3\$FileName"),
                              (Join-Path $Root "models\rentednoodle-v3\converted\Qwen3.8-27B-OrcaRouter-GSQ-RCO-IQ3_XXS-v2.1-vision-mtp.ninfer"),
                              (Join-Path "E:\llm\RentedNoodle-NInfer-v3" $FileName))) {
        if (Test-Path -LiteralPath $Candidate) { $Artifact = $Candidate; break }
    }
}
$Exe = Join-Path $Root "runtime-v3\engine\ninfer-serve.exe"

if (-not $Artifact -or -not (Test-Path -LiteralPath $Artifact)) {
    throw "A native v3 artifact is required. Pass -Model PATH or download with scripts\download-ryan-compatible-models.ps1."
}
$Artifact = (Resolve-Path -LiteralPath $Artifact).Path
$Stream = [IO.File]::OpenRead($Artifact)
try {
    $Reader = [IO.BinaryReader]::new($Stream)
    $Magic = $Reader.ReadBytes(8)
    if (($Magic -join ',') -ne '78,73,78,70,69,82,0,3') { throw "Expected NInfer v3; refusing to convert or launch this artifact." }
    $DirectoryBytes = $Reader.ReadUInt64()
    if ($DirectoryBytes -eq 0 -or $DirectoryBytes -gt 8MB) { throw "Invalid v3 directory size." }
    $Reader.ReadBytes(16) | Out-Null
    $Directory = [Text.Encoding]::UTF8.GetString($Reader.ReadBytes([int]$DirectoryBytes)) | ConvertFrom-Json
} finally { $Stream.Dispose() }
if ($Spec -ne "none" -and $Directory.components.PSObject.Properties.Name -notcontains $Spec) {
    throw "The selected artifact does not contain the $Spec component."
}
if (-not (Test-Path $Exe)) {
    throw "Native engine is missing. Build with scripts\build-ryan-engine.ps1 -ModelArtifact '$Artifact'."
}

$Exe = (Resolve-Path $Exe).Path
$Port = if ($env:NINFER_PORT) { [int]$env:NINFER_PORT } else { 8080 }
$HasVision = $Directory.components.PSObject.Properties.Name -contains "vision"
$Vision = if ($env:NINFER_VISION) { $env:NINFER_VISION -ne "0" } else { $HasVision }
if ($Vision -and -not $HasVision) { throw "The selected artifact has no vision projector." }
$Context = if ($env:NINFER_CONTEXT) { [int]$env:NINFER_CONTEXT } else { 2048 }
$HostCache = if ($env:NINFER_HOST_CACHE_MIB) { [int]$env:NINFER_HOST_CACHE_MIB } else { if ($Spec -eq "dflash2") { 6144 } else { 5120 } }
$Chunk = if ($env:NINFER_PREFILL_CHUNK) { [int]$env:NINFER_PREFILL_CHUNK } else { 256 }
$Kv = if ($env:NINFER_KV_DTYPE) { $env:NINFER_KV_DTYPE } else { "bf16" }
$MemoryPolicy = if ($env:NINFER_MEMORY_POLICY) { $env:NINFER_MEMORY_POLICY } else { "default" }

$env:NINFER_PREFILL_ALIGN = "0"
Remove-Item Env:NINFER_PROMPT_FAST -ErrorAction SilentlyContinue
Remove-Item Env:CUDA_LAUNCH_BLOCKING -ErrorAction SilentlyContinue

$ServeArgs = @(
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
    "--cuda-memory-policy",$MemoryPolicy,
    "--default-reasoning-effort","xhigh",
    "--preserve-thinking",
    "--temperature","1.0",
    "--top-p","0.95",
    "--top-k","20",
    "--min-p","0.0"
)
if ($Spec -ne "none") {
    $DraftTokens = if ($Spec -eq "mtp") { "3" } else { "4" }
    $ServeArgs += @("--spec", $Spec, "--draft-tokens", $DraftTokens, "--lm-head-draft")
}
if ($Vision) { $ServeArgs += "--vision" }

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
Write-Host "  speculation: $Spec"
Write-Host "  RAM cache: $HostCache MiB"
Write-Host ""

Push-Location (Split-Path -Parent $Exe)
try { & $Exe @ServeArgs } finally { Pop-Location }
