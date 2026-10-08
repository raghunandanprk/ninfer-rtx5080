# Video analysis (video to prompt, YouTube summarization): MTP3 + proposal head, rk4v4 KV, Vision
# tower in host memory and encoded on the GPU per request (overlay), 169,984-token window.
# Overlay costs ~0.5-1.5 s of first-token time against resident and leaves 22K more tokens for
# transcripts and follow-ups. The strict memory policy accepts only text, so this runs under the
# default policy; no spill into system memory was measured.
#
# Qwen3.8 reads frames only, sampled at 2 fps (at most 768), and one video uses at most ~12,300
# Vision tokens however long it is. The engine rejects a video over 600 s, or whose sampled frames
# exceed 128 Mi decoded pixels (a 720p video over ~72 s, 4K over ~8 s): run prepare-video.ps1
# first, which also turns a YouTube URL's subtitles into a transcript to paste into the prompt for
# anything said aloud.
# Send videos as {"type":"video_url","video_url":{"url":"data:video/mp4;base64,..."}}.
# Measurements: benchmark-results/2026-10-05-rtx5080-tuning/summary.md
# Extra arguments are appended to the server command line.
[CmdletBinding(PositionalBinding = $false)]
param(
    [int]$Port = 8095,
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
    "--host", "127.0.0.1", "--port", "$Port", "--model-id", "qwen3.8-27b-video",
    "--max-context", "169984", "--kv-capacity", "169984", "--max-concurrency", "1",
    "--default-max-tokens", "0", "--prefill-chunk", "1024",
    "--kv-dtype", "rk4v4", "--gdn-state-fp16",
    "--cuda-memory-policy", "default", "--host-cache-mib", "5120",
    "--spec", "mtp", "--draft-tokens", "3", "--lm-head-draft",
    "--vision", "--vision-residency", "overlay",
    "--default-reasoning-effort", "medium",
    "--temperature", "1.0", "--top-p", "0.95", "--top-k", "20", "--min-p", "0"
) + $Extra

Write-Host "Video server: http://127.0.0.1:$Port/v1 (model qwen3.8-27b-video, 169,984 tokens)"
Push-Location (Split-Path -Parent $Exe)
try { & $Exe @ServeArgs } finally { Pop-Location }
