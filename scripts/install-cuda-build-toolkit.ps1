param([string]$Release = "13.4.2")

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$Downloads = Join-Path $Root ".deps\toolchain-downloads"
$Toolkit = Join-Path $Root ".deps\cuda-$Release"
$ProgressPreference = "SilentlyContinue"
New-Item -ItemType Directory -Force $Downloads,$Toolkit | Out-Null
$manifestPath = Join-Path $Downloads "redistrib_$Release.json"
$baseUrl = "https://developer.download.nvidia.com/compute/cuda/redist/"
if (-not (Test-Path -LiteralPath $manifestPath)) {
    Invoke-WebRequest -UseBasicParsing -Uri ($baseUrl + "redistrib_$Release.json") -OutFile $manifestPath
}
$manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
if ($manifest.release_label -ne $Release) { throw "Unexpected CUDA release manifest." }

# Use NVIDIA's official Windows archives without altering any system installation,
# driver, registry entry, or persistent PATH. These are the compiler and cuBLAS
# components needed by NInfer, plus cuobjdump for verifying the final GPU images.
$components = @("cccl","cuda_crt","cuda_cudart","cuda_nvcc","libnvvm",
                "cuda_tileiras","cuda_nvprune","cuda_cuobjdump","cuda_nvtx","libcublas")
foreach ($name in $components) {
    $component = $manifest.$name
    $package = $component.'windows-x86_64'
    $archive = Join-Path $Downloads (Split-Path -Leaf $package.relative_path)
    if (-not (Test-Path -LiteralPath $archive)) {
        Write-Host "Downloading $name $($component.version)..."
        & curl.exe --fail --location --retry 3 --silent --show-error --output $archive ($baseUrl + $package.relative_path)
        if ($LASTEXITCODE -ne 0) { throw "CUDA component download failed: $name" }
    }
    if ((Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash.ToLowerInvariant() -ne $package.sha256) {
        throw "SHA-256 mismatch for $name; download must be replaced."
    }
    $extract = Join-Path $Downloads ([IO.Path]::GetFileNameWithoutExtension($archive))
    if (-not (Test-Path -LiteralPath $extract)) {
        Write-Host "Extracting $name..."
        & tar.exe -xf $archive -C $Downloads
        if ($LASTEXITCODE -ne 0) { throw "CUDA component extraction failed: $name" }
    }
    foreach ($item in Get-ChildItem -LiteralPath $extract) {
        if ($item.Name -match '^(LICENSE|EULA|NOTICE)') {
            $licenseDir = Join-Path $Toolkit "licenses\$name"
            New-Item -ItemType Directory -Force $licenseDir | Out-Null
            Copy-Item -LiteralPath $item.FullName -Destination $licenseDir -Recurse -Force
        } else {
            Copy-Item -LiteralPath $item.FullName -Destination $Toolkit -Recurse -Force
        }
    }
}
Copy-Item -LiteralPath $manifestPath -Destination (Join-Path $Toolkit "redistrib-manifest.json") -Force
@{ cuda = @{ name = "CUDA SDK"; version = $Release } } | ConvertTo-Json -Depth 3 |
    Set-Content -LiteralPath (Join-Path $Toolkit "version.json") -Encoding UTF8
& (Join-Path $Toolkit "bin\nvcc.exe") --version
if ($LASTEXITCODE -ne 0) { throw "Assembled CUDA compiler cannot run." }
Write-Host "Private CUDA $Release toolkit: $Toolkit"
