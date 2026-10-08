# Research engine: MTP3, rk4v4 KV, 231,424-token window.
# Validated with all three needles retrieved from a 229,375-token prompt. Prefill runs about
# 1,000 tok/s at 100K and 710 tok/s at 229K; follow-ups on the same documents reuse the cache.
# Order prompts as instructions, then documents, then the question: the hybrid prefix cache
# (selected by the strict policy) shares any common prefix by content. --prefix-cache-file
# keeps cached documents across restarts (stop with Ctrl+C so it saves).
# Add --structured-output for JSON-schema responses.
# Measurements: benchmark-results/2026-10-05-rtx5080-tuning/summary.md
# Extra arguments are appended to the server command line.
[CmdletBinding(PositionalBinding = $false)]
param(
    [int]$Port = 8093,
    [string]$Model = "",
    [string]$CacheDir = "",
    [Parameter(ValueFromRemainingArguments = $true)][string[]]$Extra = @()
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$FileName = "qwen3.8-27b-orcarouter-iq3-xxs-mtp-only.ninfer"

if (-not $Model) {
    foreach ($Candidate in @(
        $env:NINFER_ARTIFACT,
        (Join-Path $Root "models\rentednoodle-native-v3\$FileName"),
        (Join-Path "E:\llm\RentedNoodle-NInfer-v3" $FileName),
        (Join-Path $Root "models\$FileName")
    )) {
        if ($Candidate -and (Test-Path -LiteralPath $Candidate)) { $Model = $Candidate; break }
    }
}
if (-not $Model -or -not (Test-Path -LiteralPath $Model)) {
    throw "Model '$FileName' not found. Run '.\scripts\download-ryan-compatible-models.ps1' or pass -Model <path>."
}

if (-not $CacheDir) {
    if (Test-Path "E:\llm\ninfer-kv") {
        $CacheDir = "E:\llm\ninfer-kv"
    } else {
        $CacheDir = Join-Path $Root ".cache\ninfer-kv"
    }
}

$Exe = Join-Path $Root "runtime-v3\engine\ninfer-serve.exe"
if (-not (Test-Path -LiteralPath $Exe)) {
    throw "Native engine not found at $Exe. Run '.\scripts\install-prebuilt-v3-runtime.ps1' or '.\scripts\build-ryan-engine.ps1' first."
}
if (Get-Process ninfer-serve -ErrorAction SilentlyContinue) {
    throw "Another ninfer-serve is running; 16 GB holds one server at a time. Stop it first."
}
New-Item -ItemType Directory -Force -Path $CacheDir | Out-Null

$env:NINFER_PREFILL_ALIGN = "0"
Remove-Item Env:NINFER_PROMPT_FAST, Env:CUDA_LAUNCH_BLOCKING -ErrorAction SilentlyContinue

$ServeArgs = @(
    $Model,
    "--host", "127.0.0.1", "--port", "$Port", "--model-id", "qwen3.8-27b-research",
    "--max-context", "231424", "--kv-capacity", "231424", "--max-concurrency", "1",
    "--default-max-tokens", "0", "--prefill-chunk", "1024",
    "--kv-dtype", "rk4v4", "--gdn-state-fp16",
    "--cuda-memory-policy", "strict", "--host-cache-mib", "5120",
    "--spec", "mtp", "--draft-tokens", "3",
    "--prefix-cache-file", (Join-Path $CacheDir "research.cache"),
    "--default-reasoning-effort", "high",
    "--temperature", "1.0", "--top-p", "0.95", "--top-k", "20", "--min-p", "0"
) + $Extra

Write-Host "Research server: http://127.0.0.1:$Port/v1 (model qwen3.8-27b-research, 231,424 tokens)"
Push-Location (Split-Path -Parent $Exe)
try { & $Exe @ServeArgs } finally { Pop-Location }
