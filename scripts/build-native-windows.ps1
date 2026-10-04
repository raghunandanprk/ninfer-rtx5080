$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$Engine = Join-Path $Root "engine"
$Build = Join-Path $Engine "build-windows"
$Deps = Join-Path $Root ".deps"
$Vcpkg = Join-Path $Deps "vcpkg"
$Installed = Join-Path $Deps "vcpkg-installed"

& (Join-Path $Root "scripts\prepare-native-windows.ps1")

# Import a VS 2022 x64 developer environment into this PowerShell process.
$VsWhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
if (-not (Test-Path $VsWhere)) {
    throw "vswhere.exe not found. Install Visual Studio 2022 Build Tools with Desktop development with C++."
}
$VsRoot = (& $VsWhere -latest -products * -version "[17.0,18.0)" -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath).Trim()
if (-not $VsRoot) {
    throw "Visual Studio 2022 C++ toolchain not found. Install VS 2022 Desktop development with C++."
}
$VsDevCmd = Join-Path $VsRoot "Common7\Tools\VsDevCmd.bat"
$EnvDump = & cmd.exe /s /c "`"$VsDevCmd`" -no_logo -arch=x64 -host_arch=x64 && set"
foreach ($Line in $EnvDump) {
    $Pos = $Line.IndexOf("=")
    if ($Pos -gt 0) {
        $Name = $Line.Substring(0, $Pos)
        $Value = $Line.Substring($Pos + 1)
        Set-Item -Path "Env:$Name" -Value $Value
    }
}

if (-not (Get-Command cl.exe -ErrorAction SilentlyContinue)) { throw "cl.exe is not available after VsDevCmd." }
if (-not (Get-Command cmake.exe -ErrorAction SilentlyContinue)) { throw "cmake.exe is not on PATH." }
if (-not (Get-Command nvcc.exe -ErrorAction SilentlyContinue)) { throw "nvcc.exe is not on PATH. Install CUDA Toolkit 12.8+ (13.x recommended)." }

$NvccText = (& nvcc.exe --version | Out-String)
if ($NvccText -notmatch "release\s+([0-9]+)\.([0-9]+)") { throw "Could not parse nvcc version." }
$CudaMajor = [int]$Matches[1]
$CudaMinor = [int]$Matches[2]
if (($CudaMajor -lt 12) -or ($CudaMajor -eq 12 -and $CudaMinor -lt 8)) {
    throw "CUDA Toolkit 12.8 or newer is required; found $CudaMajor.$CudaMinor."
}
Write-Host "Using CUDA Toolkit $CudaMajor.$CudaMinor"
Write-Host "Using VS 2022: $VsRoot"

# Prefer Ninja. VS 2022 ships a copy with its CMake integration on most installs.
if (-not (Get-Command ninja.exe -ErrorAction SilentlyContinue)) {
    $VsNinja = Join-Path $VsRoot "Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe"
    if (Test-Path $VsNinja) {
        $env:PATH = "$(Split-Path -Parent $VsNinja);$env:PATH"
    }
}
if (-not (Get-Command ninja.exe -ErrorAction SilentlyContinue)) {
    throw "ninja.exe not found. Install Ninja or the Visual Studio CMake tools component."
}

# Bootstrap a private vcpkg copy. The engine has a pinned vcpkg.json manifest.
New-Item -ItemType Directory -Force $Deps | Out-Null
if (-not (Test-Path (Join-Path $Vcpkg ".git"))) {
    git clone https://github.com/microsoft/vcpkg.git $Vcpkg
}
$VcpkgExe = Join-Path $Vcpkg "vcpkg.exe"
if (-not (Test-Path $VcpkgExe)) {
    & (Join-Path $Vcpkg "bootstrap-vcpkg.bat") -disableMetrics
    if ($LASTEXITCODE -ne 0) { throw "vcpkg bootstrap failed." }
}

New-Item -ItemType Directory -Force $Installed | Out-Null
New-Item -ItemType Directory -Force $Build | Out-Null

$Toolchain = Join-Path $Vcpkg "scripts\buildsystems\vcpkg.cmake"
$Jobs = if ($env:NINFER_BUILD_JOBS) { [int]$env:NINFER_BUILD_JOBS } else { [Math]::Max(2, [Environment]::ProcessorCount - 2) }

Write-Host "Configuring native Windows NInfer..."
& cmake.exe -S $Engine -B $Build -G Ninja `
    -DCMAKE_BUILD_TYPE=Release `
    -DCMAKE_CUDA_ARCHITECTURES=120a `
    -DCMAKE_TOOLCHAIN_FILE=$Toolchain `
    -DVCPKG_TARGET_TRIPLET=x64-windows `
    -DVCPKG_MANIFEST_MODE=ON `
    -DVCPKG_INSTALLED_DIR=$Installed `
    -DNINFER_BUILD_APPS=ON `
    -DBUILD_TESTING=OFF `
    -DNINFER_BUILD_BENCHMARKS=OFF
if ($LASTEXITCODE -ne 0) { throw "CMake configure failed." }

Write-Host "Building ninfer-serve.exe with $Jobs parallel jobs..."
& cmake.exe --build $Build --target ninfer-serve --parallel $Jobs
if ($LASTEXITCODE -ne 0) { throw "Native Windows build failed." }

$ExeDir = Join-Path $Build "apps"
$Exe = Join-Path $ExeDir "ninfer-serve.exe"
if (-not (Test-Path $Exe)) { throw "Build completed but ninfer-serve.exe was not found at $Exe" }

# Copy vcpkg runtime DLLs beside the executable so Windows can start it directly.
$Bin = Join-Path $Installed "x64-windows\bin"
if (Test-Path $Bin) {
    Get-ChildItem $Bin -Filter *.dll | Copy-Item -Destination $ExeDir -Force
}

Write-Host ""
Write-Host "Native Windows NInfer server built:"
Write-Host "  $Exe"
Write-Host "No Docker or WSL is required to run this executable."
