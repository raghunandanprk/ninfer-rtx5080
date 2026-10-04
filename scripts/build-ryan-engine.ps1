param(
    [switch]$Fresh,
    [int]$Jobs = 4
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$Source = Join-Path $Root ".deps\ninfer-v3"
$Build = Join-Path $Root ".deps\ninfer-v3-build-sm120a"
$Vcpkg = Join-Path $Root ".deps\vcpkg"
$Runtime = Join-Path $Root "runtime-v3\engine"

function Invoke-Checked {
    param(
        [Parameter(Mandatory=$true)][string]$FilePath,
        [Parameter(ValueFromRemainingArguments=$true)][string[]]$Arguments
    )
    & $FilePath @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$FilePath failed with exit code $LASTEXITCODE"
    }
}

function Import-VsEnvironment {
    $programFilesX86 = [Environment]::GetFolderPath([Environment+SpecialFolder]::ProgramFilesX86)
    $vswhere = Join-Path $programFilesX86 "Microsoft Visual Studio\Installer\vswhere.exe"
    if (-not (Test-Path $vswhere)) {
        throw "Visual Studio Installer/vswhere.exe not found. Install Visual Studio 2022 Build Tools with Desktop development with C++."
    }

    $vsRoot = (& $vswhere -latest -products * -version "[17.0,18.0)" -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath).Trim()
    if (-not $vsRoot) {
        throw "Visual Studio 2022 C++ toolchain not found. Install Desktop development with C++."
    }

    $vcvars = Join-Path $vsRoot "VC\Auxiliary\Build\vcvars64.bat"
    if (-not (Test-Path $vcvars)) {
        throw "vcvars64.bat not found under $vsRoot"
    }

    $cmdline = 'call "' + $vcvars + '" -vcvars_ver=14.44 >nul 2>nul && set'
    $dump = & cmd.exe /d /s /c $cmdline
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "MSVC 14.44 is not installed; using the current VS2022 x64 toolset instead."
        $cmdline = 'call "' + $vcvars + '" >nul && set'
        $dump = & cmd.exe /d /s /c $cmdline
        if ($LASTEXITCODE -ne 0) {
            throw "Visual Studio x64 developer environment setup failed."
        }
    }

    foreach ($line in $dump) {
        if ($line -match '^([^=]+)=(.*)$') {
            Set-Item -Path "Env:$($matches[1])" -Value $matches[2]
        }
    }

    $cl = (Get-Command cl.exe -ErrorAction Stop).Source

    if (-not (Get-Command cmake.exe -ErrorAction SilentlyContinue)) {
        $vsCmake = Join-Path $vsRoot "Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin"
        if (Test-Path (Join-Path $vsCmake "cmake.exe")) {
            $env:PATH = "$vsCmake;$env:PATH"
        }
    }
    if (-not (Get-Command ninja.exe -ErrorAction SilentlyContinue)) {
        $vsNinja = Join-Path $vsRoot "Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja"
        if (Test-Path (Join-Path $vsNinja "ninja.exe")) {
            $env:PATH = "$vsNinja;$env:PATH"
        }
    }

    if (-not (Get-Command cmake.exe -ErrorAction SilentlyContinue)) {
        throw "cmake.exe not found. Install Visual Studio C++ CMake tools or CMake separately."
    }
    if (-not (Get-Command ninja.exe -ErrorAction SilentlyContinue)) {
        throw "ninja.exe not found. Install Visual Studio C++ CMake tools or Ninja separately."
    }

    return [pscustomobject]@{
        Root = $vsRoot
        Cl = $cl
        Toolset = $env:VCToolsVersion
    }
}

function Find-CudaRoot {
    $candidates = @()

    if ($env:NINFER_CUDA_ROOT) {
        $candidates += $env:NINFER_CUDA_ROOT
    }

    $candidates += "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4"

    if ($env:CUDA_PATH) {
        $candidates += $env:CUDA_PATH
    }

    $cudaParent = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"
    if (Test-Path $cudaParent) {
        $candidates += Get-ChildItem $cudaParent -Directory -Filter "v13.*" |
            Sort-Object Name -Descending |
            ForEach-Object FullName
    }

    foreach ($candidate in $candidates | Select-Object -Unique) {
        if (-not $candidate) { continue }
        $nvcc = Join-Path $candidate "bin\nvcc.exe"
        $nvprune = Join-Path $candidate "bin\nvprune.exe"
        if ((Test-Path $nvcc) -and (Test-Path $nvprune)) {
            return $candidate
        }
    }

    throw "CUDA Toolkit 13.x with nvcc and nvprune was not found. Ryan validated CUDA 13.4.2. CUDA 13.0 is allowed by this script; set NINFER_CUDA_ROOT if CUDA is installed in a nonstandard location."
}

function Copy-CudaDll {
    param([string]$CudaRoot,[string]$Name,[string]$Destination)

    foreach ($dir in @((Join-Path $CudaRoot "bin"), (Join-Path $CudaRoot "bin\x64"))) {
        $path = Join-Path $dir $Name
        if (Test-Path $path) {
            Copy-Item $path $Destination -Force
            return
        }
    }
    throw "Required CUDA runtime DLL not found: $Name"
}

if ($Jobs -lt 1) { throw "Jobs must be at least 1." }

& (Join-Path $Root "scripts\bootstrap-v3.ps1")
if (-not (Test-Path (Join-Path $Source "CMakeLists.txt"))) {
    throw "Pinned Ryan-gsq source is missing after bootstrap."
}

$vs = Import-VsEnvironment
$cudaRoot = Find-CudaRoot
$nvcc = Join-Path $cudaRoot "bin\nvcc.exe"
$nvprune = Join-Path $cudaRoot "bin\nvprune.exe"

$cudaVersionText = (& $nvcc --version | Out-String)
if ($cudaVersionText -notmatch 'release\s+([0-9]+)\.([0-9]+)') {
    throw "Could not determine CUDA Toolkit version from nvcc."
}
$cudaMajor = [int]$matches[1]
$cudaMinor = [int]$matches[2]
if ($cudaMajor -ne 13) {
    throw "This RTX 50-series build expects CUDA 13.x; found $cudaMajor.$cudaMinor."
}
if ($cudaMinor -lt 4) {
    Write-Warning "CUDA $cudaMajor.$cudaMinor detected. Ryan validated 13.4.2; attempting the build with your installed toolkit."
    Write-Warning "If compilation fails in CUDA APIs/kernels, install 13.4.2 side-by-side and rerun with NINFER_CUDA_ROOT pointing to v13.4."
}

$env:CUDA_PATH = $cudaRoot
$env:PATH = "$(Join-Path $cudaRoot 'bin');$(Join-Path $cudaRoot 'bin\x64');$env:PATH"

Write-Host ""
Write-Host "Ryan-gsq native engine build"
Write-Host "  source:  $Source"
Write-Host "  build:   $Build"
Write-Host "  runtime: $Runtime"
Write-Host "  MSVC:    $($vs.Toolset)"
Write-Host "  CUDA:    $cudaMajor.$cudaMinor ($cudaRoot)"
Write-Host "  jobs:    $Jobs"
Write-Host ""

if ($Fresh -and (Test-Path $Build)) {
    Write-Host "Removing previous build tree (-Fresh)..."
    Remove-Item $Build -Recurse -Force
}
New-Item -ItemType Directory -Force (Split-Path -Parent $Build) | Out-Null

if (-not (Test-Path (Join-Path $Vcpkg ".git"))) {
    Write-Host "Cloning vcpkg..."
    Invoke-Checked git clone https://github.com/microsoft/vcpkg.git $Vcpkg
}
$vcpkgExe = Join-Path $Vcpkg "vcpkg.exe"
if (-not (Test-Path $vcpkgExe)) {
    Invoke-Checked (Join-Path $Vcpkg "bootstrap-vcpkg.bat") -disableMetrics
}
$toolchain = Join-Path $Vcpkg "scripts\buildsystems\vcpkg.cmake"

$configureArgs = @(
    "-S", $Source,
    "-B", $Build,
    "-G", "Ninja",
    "-DCMAKE_TOOLCHAIN_FILE=$toolchain",
    "-DVCPKG_TARGET_TRIPLET=x64-windows",
    "-DVCPKG_MANIFEST_MODE=ON",
    "-DCMAKE_BUILD_TYPE=Release",
    "-DCMAKE_CUDA_ARCHITECTURES=120a",
    "-DNINFER_SM120_NATIVE=ON",
    "-DNINFER_BUILD_APPS=ON",
    "-DBUILD_TESTING=OFF",
    "-DNINFER_BUILD_BENCHMARKS=OFF",
    "-DNINFER_DIRECTSTORAGE=OFF",
    "-DNINFER_D3D12_RESIDENCY=OFF",
    "-DCUDAToolkit_ROOT=$cudaRoot",
    "-DCMAKE_CUDA_COMPILER=$nvcc",
    "-DCMAKE_CUDA_HOST_COMPILER=$($vs.Cl)",
    "-DCMAKE_C_COMPILER=$($vs.Cl)",
    "-DCMAKE_CXX_COMPILER=$($vs.Cl)"
)

Write-Host "Configuring CMake + vcpkg..."
Invoke-Checked cmake.exe @configureArgs

Write-Host "Building CUDA operators..."
Invoke-Checked cmake.exe --build $Build --target ninfer_ops -j $Jobs

$opsArchive = Join-Path $Build "src\ops\ninfer_ops.lib"
$prunedArchive = Join-Path $Build "src\ops\ninfer_ops.native.lib"
$backupRoot = Join-Path $Build "archive-backup"
if (-not (Test-Path $opsArchive)) {
    throw "ninfer_ops build completed but archive was not found: $opsArchive"
}
New-Item -ItemType Directory -Force $backupRoot | Out-Null
$backupName = "ninfer_ops-" + (Get-Date -Format "yyyyMMdd-HHmmssfff") + ".with-ptx.lib"
Copy-Item $opsArchive (Join-Path $backupRoot $backupName)

if (Test-Path $prunedArchive) { Remove-Item $prunedArchive -Force }
Write-Host "Pruning redundant PTX; retaining sm_120a SASS..."
Invoke-Checked $nvprune -arch sm_120a $opsArchive -o $prunedArchive
Copy-Item $prunedArchive $opsArchive -Force

Write-Host "Linking ninfer-serve.exe..."
Invoke-Checked cmake.exe --build $Build --target ninfer-serve -j $Jobs

$builtExe = Join-Path $Build "apps\ninfer-serve.exe"
if (-not (Test-Path $builtExe)) {
    throw "Build succeeded but ninfer-serve.exe was not found: $builtExe"
}

Write-Host "Assembling standalone runtime..."
if (Test-Path $Runtime) {
    Remove-Item $Runtime -Recurse -Force
}
New-Item -ItemType Directory -Force $Runtime | Out-Null
Copy-Item $builtExe $Runtime -Force

$dependencyBin = Join-Path $Build "vcpkg_installed\x64-windows\bin"
if (-not (Test-Path $dependencyBin)) {
    throw "vcpkg runtime bin directory not found: $dependencyBin"
}
Get-ChildItem $dependencyBin -File -Filter "*.dll" | Copy-Item -Destination $Runtime -Force

foreach ($dll in @("cublas64_13.dll","cublasLt64_13.dll","cudart64_13.dll")) {
    Copy-CudaDll -CudaRoot $cudaRoot -Name $dll -Destination $Runtime
}

$crtRoot = if ($env:VCToolsRedistDir) {
    Join-Path $env:VCToolsRedistDir "x64\Microsoft.VC143.CRT"
} else {
    $null
}
if (-not $crtRoot -or -not (Test-Path $crtRoot)) {
    throw "MSVC x64 redistributable directory was not found through VCToolsRedistDir."
}
Get-ChildItem $crtRoot -File -Filter "*.dll" | Copy-Item -Destination $Runtime -Force

$runtimeExe = Join-Path $Runtime "ninfer-serve.exe"

Write-Host "Checking standalone runtime with a minimal PATH..."
$savedPath = $env:PATH
try {
    $env:PATH = "$env:SystemRoot\System32;$env:SystemRoot"
    & $runtimeExe --help *> $null
    if ($LASTEXITCODE -ne 0) {
        throw "Standalone ninfer-serve.exe --help failed with exit code $LASTEXITCODE. A runtime DLL may still be missing."
    }
} finally {
    $env:PATH = $savedPath
}

$manifest = [ordered]@{
    source_repository = "Ryan-gsq/ninfer-16g-5070ti-5080-5090-qwen3.8-27b-gsq-rco"
    source_commit = "b06908ba3caa4f73269274fc7984b96f16d4295c"
    architecture = "sm_120a"
    build_type = "Release"
    ninfer_sm120_native = $true
    directstorage = $false
    d3d12_residency = $false
    cuda_toolkit = "$cudaMajor.$cudaMinor"
    cuda_root = $cudaRoot
    msvc_toolset = $vs.Toolset
    built_utc = [DateTime]::UtcNow.ToString("o")
    exe_sha256 = (Get-FileHash $runtimeExe -Algorithm SHA256).Hash.ToLowerInvariant()
}
$manifest | ConvertTo-Json | Set-Content (Join-Path $Runtime "build-manifest.json") -Encoding UTF8

Write-Host ""
Write-Host "Native SM120a runtime is ready:"
Write-Host "  $runtimeExe"
Write-Host "  SHA-256: $($manifest.exe_sha256)"
Write-Host ""
Write-Host "No Docker or WSL is required for normal inference after this build."
