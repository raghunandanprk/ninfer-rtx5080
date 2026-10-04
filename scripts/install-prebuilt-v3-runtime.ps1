param(
    [Parameter(Mandatory=$true)]
    [string]$PackageRoot
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$Destination = Join-Path $Root "runtime-v3\engine"
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

$InstalledExe = Join-Path $Destination "ninfer-serve.exe"
if (-not (Test-Path $InstalledExe)) { throw "Runtime copy completed but ninfer-serve.exe is missing." }

Write-Host "Precompiled SM120a runtime installed:"
Write-Host "  $InstalledExe"
Write-Host "All DLLs from the source engine directory were copied with it."
