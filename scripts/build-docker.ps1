$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$Engine = Join-Path $Root "engine"
$Bootstrap = Join-Path $Root "scripts\bootstrap-source.ps1"
if (-not (Test-Path (Join-Path $Engine "CMakeLists.txt"))) { & $Bootstrap }
docker version | Out-Null
if ($LASTEXITCODE -ne 0) { throw "Docker Desktop is not available." }
$Image = "ninfer-rtx5080:gsq3"
Write-Host "Building $Image for Blackwell sm_120a..."
docker build -t $Image $Engine
if ($LASTEXITCODE -ne 0) { throw "Docker build failed." }
Write-Host "Built: $Image"
