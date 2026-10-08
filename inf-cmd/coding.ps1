# Coding harnesses (Claude Code, opencode, Pi): MTP3, rk4v4 KV, 229,376-token window.
# Ceiling is 2,048 tokens below research.ps1 because --ngram-draft-tokens 31 widens verification.
# Larger than the 200K window Claude Code compacts against; set opencode/Pi to ~223K.
# N-gram drafting copies from files and tool output, with a per-session archive of earlier
# requests. The strict policy selects the hybrid prefix cache, which resumes a rewritten
# history from its nearest snapshot; --prefix-cache-file keeps the Host tier across restarts
# (stop with Ctrl+C so it saves). Measurements: benchmark-results/2026-10-05-rtx5080-tuning/summary.md
# Extra arguments are appended to the server command line.
[CmdletBinding(PositionalBinding = $false)]
param(
    [int]$Port = 8092,
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
    "--host", "127.0.0.1", "--port", "$Port", "--model-id", "qwen3.8-27b-coding",
    "--max-context", "229376", "--kv-capacity", "229376", "--max-concurrency", "1",
    "--default-max-tokens", "0", "--prefill-chunk", "1024",
    "--kv-dtype", "rk4v4", "--gdn-state-fp16",
    "--cuda-memory-policy", "strict", "--host-cache-mib", "5120",
    "--spec", "mtp", "--draft-tokens", "3",
    "--ngram-draft-tokens", "31", "--ngram-native-sessions", "--ngram-archive-mib", "512",
    "--prefix-cache-file", (Join-Path $CacheDir "coding.cache"),
    "--preserve-thinking", "--default-reasoning-effort", "high",
    "--temperature", "0.6", "--top-p", "0.95", "--top-k", "20", "--min-p", "0"
) + $Extra

Write-Host "Coding server: http://127.0.0.1:$Port (model qwen3.8-27b-coding, 229,376 tokens)"
Write-Host "  OpenAI base: http://127.0.0.1:$Port/v1   Anthropic base: http://127.0.0.1:$Port"
Push-Location (Split-Path -Parent $Exe)
try { & $Exe @ServeArgs } finally { Pop-Location }
