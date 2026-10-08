param(
    [string]$Model = "",
    [ValidateSet("none", "mtp", "dflash2")][string]$Spec = "none"
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$Runtime = Join-Path $Root "runtime-v3\engine"
$Source = Join-Path $Root ".deps\ninfer-v3"
$Build = Join-Path $Root ".deps\ninfer-v3-build-sm120a"
$Validation = Join-Path $Root ".deps\gpu-validation"
$Python = if (Test-Path (Join-Path $Root ".deps\build-venv\Scripts\python.exe")) {
    Join-Path $Root ".deps\build-venv\Scripts\python.exe"
} elseif (Get-Command python.exe -ErrorAction SilentlyContinue) {
    (Get-Command python.exe).Source
} elseif (Get-Command py.exe -ErrorAction SilentlyContinue) {
    (Get-Command py.exe).Source
} else {
    throw "Python was not found. Please install Python or ensure it is available in your PATH."
}
$ManifestPath = Join-Path $Runtime "build-manifest.json"
$manifest = Get-Content -LiteralPath $ManifestPath -Raw | ConvertFrom-Json
New-Item -ItemType Directory -Force $Validation | Out-Null

& $Python (Join-Path $Root "scripts\inspect-windows-runtime.py") $Runtime --output (Join-Path $Validation "dll-dependencies.json")
if ($LASTEXITCODE -ne 0) { throw "Packaged DLL dependency closure failed." }

$savedPath = $env:PATH
try {
    $env:PATH = "$env:SystemRoot\System32;$env:SystemRoot"
    $help = Start-Process -FilePath (Join-Path $Runtime "ninfer-serve.exe") -ArgumentList "--help" -WindowStyle Hidden -PassThru -Wait `
        -RedirectStandardOutput (Join-Path $Validation "standalone-help.txt") -RedirectStandardError (Join-Path $Validation "standalone-help.err.txt")
    if ($help.ExitCode -ne 0) { throw "Standalone --help failed: $($help.ExitCode)" }
} finally { $env:PATH = $savedPath }

$gpu = (& nvidia-smi.exe --query-gpu=name,driver_version,memory.total --format=csv,noheader | Out-String).Trim()
$verification = [ordered]@{
    dll_dependency_closure = $true
    standalone_help_exit_code = $help.ExitCode
    gpu = $gpu
    logs = $Validation
    tested_utc = [DateTime]::UtcNow.ToString("o")
}
if ($Model) {
    $modelPath = (Resolve-Path -LiteralPath $Model).Path
    $before = Get-Item -LiteralPath $modelPath
    $sizeBefore = $before.Length
    $timeBefore = $before.LastWriteTimeUtc
    & $Python (Join-Path $Root "scripts\inspect-ryan-model.py") $modelPath
    if ($LASTEXITCODE -ne 0) { throw "Model compatibility check failed before GPU loading." }
    $listener = [Net.Sockets.TcpListener]::new([Net.IPAddress]::Loopback, 0)
    $listener.Start()
    $port = $listener.LocalEndpoint.Port
    $listener.Stop()
    $label = [IO.Path]::GetFileNameWithoutExtension($modelPath) + "-$Spec"
    $hostCacheMib = if ($Spec -eq "dflash2") { "6144" } else { "5120" }
    $modelArgs = @(('"' + $modelPath + '"'), "--host", "127.0.0.1", "--port", "$port", "--device", "0",
                   "--max-context", "2048", "--kv-capacity", "2048", "--host-cache-mib", $hostCacheMib,
                   "--max-concurrency", "1", "--prefill-chunk", "256", "--no-thinking", "--greedy")
    if ($Spec -ne "none") {
        $draftTokens = if ($Spec -eq "mtp") { "3" } else { "4" }
        $modelArgs += @("--spec", $Spec, "--draft-tokens", $draftTokens, "--lm-head-draft")
    }
    $savedPath = $env:PATH
    $server = $null
    try {
        $env:PATH = "$env:SystemRoot\System32;$env:SystemRoot"
        $server = Start-Process -FilePath (Join-Path $Runtime "ninfer-serve.exe") -ArgumentList $modelArgs -WindowStyle Hidden -PassThru `
            -RedirectStandardOutput (Join-Path $Validation "$label.txt") -RedirectStandardError (Join-Path $Validation "$label.err.txt")
        $responsePath = Join-Path $Validation "$label.responses.json"
        & $Python (Join-Path $Root "scripts\probe-ryan-server.py") --port $port --pid $server.Id --output $responsePath
        if ($LASTEXITCODE -ne 0) { throw "Real-model GPU inference failed; inspect $label startup logs." }
        $smi = Join-Path $env:SystemRoot "System32\nvidia-smi.exe"
        if (-not (Test-Path -LiteralPath $smi)) { $smi = (Get-Command nvidia-smi.exe).Source }
        $gpuProcesses = (& $smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv,noheader | Out-String).Trim()
        $after = Get-Item -LiteralPath $modelPath
        if ($after.Length -ne $sizeBefore -or $after.LastWriteTimeUtc -ne $timeBefore) { throw "Model size or timestamp changed during inference." }
        $verification.model_test = [ordered]@{
            path = $modelPath
            backend = $Spec
            pid = $server.Id
            unchanged_size_and_mtime = $true
            passed = $true
            gpu_processes = $gpuProcesses
            responses = $responsePath
            arguments = $modelArgs
        }
    } finally {
        if ($server -and -not $server.HasExited) { $server.Kill(); $server.WaitForExit() }
        $env:PATH = $savedPath
    }
}
if ($manifest.PSObject.Properties.Name -contains "verification") {
    if ($manifest.verification.PSObject.Properties.Name -contains "model_tests") {
        $verification.model_tests = @($manifest.verification.model_tests)
    } else { $verification.model_tests = @() }
} else { $verification.model_tests = @() }
if ($verification.Contains("model_test")) {
    $verification.model_tests += $verification.model_test
    $verification.Remove("model_test")
}
$manifest | Add-Member -NotePropertyName verification -NotePropertyValue $verification -Force
$manifest | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $ManifestPath -Encoding UTF8
Write-Host "Standalone package and selected GPU validation passed."
