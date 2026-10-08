# Image analysis (image to prompt): MTP3 + proposal head, rk4v4 KV, Vision tower resident on the
# GPU, 147,456-token window. A 1.6 MP image encodes in ~0.2 s and reaches its first token in ~1.5 s.
# The strict memory policy accepts only text, so this runs under the default policy; no spill into
# system memory was measured. Needs the vision artifact built in the 2026-10-05 tuning run.
# Thinking is off by default (Qwen's non-thinking preset); a request can turn it on.
# Measurements: benchmark-results/2026-10-05-rtx5080-tuning/summary.md
# Extra arguments are appended to the server command line.
[CmdletBinding(PositionalBinding = $false)]
param(
    [int]$Port = 8094,
    [string]$Model = "",
    [Parameter(ValueFromRemainingArguments = $true)][string[]]$Extra = @()
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$FileName = "qwen3.8-27b-orcarouter-iq3-xxs-vision-mtp.ninfer"

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

$Exe = Join-Path $Root "runtime-v3\engine\ninfer-serve.exe"
if (-not (Test-Path -LiteralPath $Exe)) {
    throw "Native engine not found at $Exe. Run '.\scripts\install-prebuilt-v3-runtime.ps1' or '.\scripts\build-ryan-engine.ps1' first."
}
if (Get-Process ninfer-serve -ErrorAction SilentlyContinue) {
    throw "Another ninfer-serve is running; 16 GB holds one server at a time. Stop it first."
}

$env:NINFER_PREFILL_ALIGN = "0"
Remove-Item Env:NINFER_PROMPT_FAST, Env:CUDA_LAUNCH_BLOCKING -ErrorAction SilentlyContinue

$ServeArgs = @(
    $Model,
    "--host", "127.0.0.1", "--port", "$Port", "--model-id", "qwen3.8-27b-image",
    "--max-context", "147456", "--kv-capacity", "147456", "--max-concurrency", "1",
    "--default-max-tokens", "0", "--prefill-chunk", "1024",
    "--kv-dtype", "rk4v4", "--gdn-state-fp16",
    "--cuda-memory-policy", "default", "--host-cache-mib", "5120",
    "--spec", "mtp", "--draft-tokens", "3", "--lm-head-draft",
    "--vision", "--vision-residency", "resident",
    "--no-thinking",
    "--temperature", "0.7", "--top-p", "0.8", "--top-k", "20", "--min-p", "0"
) + $Extra

Write-Host "Image server: http://127.0.0.1:$Port/v1 (model qwen3.8-27b-image, 147,456 tokens)"
Push-Location (Split-Path -Parent $Exe)
try { & $Exe @ServeArgs } finally { Pop-Location }
