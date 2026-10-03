$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$ModelDir = Join-Path $Root "models"
$Model = Join-Path $ModelDir "qwen3_8_27b_gsq3.ninfer"
$Revision = "2359d374b0f6ed3cd400815e5114d32ce03b5899"
$Expected = "c6f27073393e5bcc629489420470d71f52a27553bfc5c360fef07a25b3b550d7"
$Url = "https://huggingface.co/roofkid/Qwen3.8-27B-GSQ3-NInfer/resolve/$Revision/qwen3_8_27b_gsq3.ninfer"
New-Item -ItemType Directory -Force $ModelDir | Out-Null
if (-not (Test-Path $Model)) {
  & curl.exe -L -C - --fail --output $Model $Url
  if ($LASTEXITCODE -ne 0) { throw "Model download failed." }
}
$Actual = (Get-FileHash -Algorithm SHA256 $Model).Hash.ToLowerInvariant()
if ($Actual -ne $Expected) { throw "SHA-256 mismatch. Expected $Expected, got $Actual" }
Write-Host "Model verified: $Model"
