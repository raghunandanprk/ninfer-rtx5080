$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$Engine = Join-Path $Root "engine"
$Upstream = "https://github.com/roofkid/ninfer-4080.git"
$Branch = "rtx4080-port"
$Commit = "1794c8692d65fcd7152071a58e08886eee68134e"

if (Test-Path $Engine) {
    Write-Host "Engine already exists: $Engine"
    Write-Host "Delete it and rerun this script to refresh from the pinned upstream."
    exit 0
}

git clone --branch $Branch $Upstream $Engine
git -C $Engine checkout $Commit

$CMake = Join-Path $Engine "CMakeLists.txt"
$Text = Get-Content $CMake -Raw
$Text = $Text.Replace("compiled only for sm_89", "compiled only for sm_120a")
$Text = $Text.Replace("CMAKE_CUDA_ARCHITECTURES 89 CACHE", "CMAKE_CUDA_ARCHITECTURES 120a CACHE")
$Text = $Text.Replace('CMAKE_CUDA_ARCHITECTURES STREQUAL "89"', 'CMAKE_CUDA_ARCHITECTURES STREQUAL "120a"')
$Text = $Text.Replace("NInfer supports only CMAKE_CUDA_ARCHITECTURES=89", "This RTX 5080 GSQ3 port requires CMAKE_CUDA_ARCHITECTURES=120a")

if (-not $Text.Contains("CMAKE_CUDA_ARCHITECTURES 120a")) { throw "Failed to patch CMake architecture gate." }
Set-Content -Path $CMake -Value $Text -NoNewline

Write-Host "GSQ3 source prepared for RTX 5080 / sm_120a at $Engine"
