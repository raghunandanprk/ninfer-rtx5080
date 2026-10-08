param(
    [Parameter(Mandatory=$false)]
    [string]$PackageRoot = "",
    [string]$ReleaseUrl = "https://github.com/raghunandanprk/ninfer-rtx5080/releases/download/v0.3.0/ninfer-sm120a-engine.zip"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$Destination = Join-Path $Root "runtime-v3\engine"

if (-not $PackageRoot) {
    Write-Host "No local package root provided; downloading prebuilt SM120a engine from GitHub Releases..."
    Write-Host "  Source: $ReleaseUrl"
    $TempDir = Join-Path $Root ".deps"
    New-Item -ItemType Directory -Force $TempDir | Out-Null
    $ZipPath = Join-Path $TempDir "ninfer-sm120a-engine.zip"

    if (Get-Command curl.exe -ErrorAction SilentlyContinue) {
        & curl.exe -L --fail --retry 3 --progress-bar -o $ZipPath $ReleaseUrl
        if ($LASTEXITCODE -ne 0) { throw "Failed to download prebuilt engine ZIP with curl." }
    } else {
        Invoke-WebRequest -Uri $ReleaseUrl -OutFile $ZipPath
    }

    New-Item -ItemType Directory -Force $Destination | Out-Null
    Write-Host "Extracting engine into $Destination..."
    if (Get-Command tar.exe -ErrorAction SilentlyContinue) {
        & tar.exe -xf $ZipPath -C $Destination
        if ($LASTEXITCODE -ne 0) { throw "Failed to extract engine ZIP with tar." }
    } else {
        Expand-Archive -LiteralPath $ZipPath -DestinationPath $Destination -Force
    }
} else {
    $PackageRoot = (Resolve-Path $PackageRoot).Path
    $Candidates = @(
        (Join-Path $PackageRoot "engine\ninfer-serve.exe"),
        (Join-Path $PackageRoot "ninfer-serve.exe")
    )
    $Exe = $Candidates | Where-Object { Test-Path $_ -PathType Leaf } | Select-Object -First 1
    if (-not $Exe) {
        $Found = Get-ChildItem -LiteralPath $PackageRoot -Filter ninfer-serve.exe -File -Recurse -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($Found) { $Exe = $Found.FullName }
    }
    if (-not $Exe) {
        throw "Could not find ninfer-serve.exe under $PackageRoot. Point PackageRoot at the extracted Ryan-gsq Windows package."
    }

    $EngineDir = Split-Path -Parent $Exe
    New-Item -ItemType Directory -Force (Split-Path -Parent $Destination) | Out-Null
    if (Test-Path $Destination) { Remove-Item $Destination -Recurse -Force }
    Copy-Item -LiteralPath $EngineDir -Destination $Destination -Recurse
}

$InstalledExe = Join-Path $Destination "ninfer-serve.exe"
if (-not (Test-Path $InstalledExe)) { throw "Runtime installation completed but ninfer-serve.exe is missing." }

Write-Host ""
Write-Host "Standalone SM120a runtime is ready:"
Write-Host "  $InstalledExe"
Write-Host "All required CUDA, cuBLAS, and VC-runtime DLLs are present."

