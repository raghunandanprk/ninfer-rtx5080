# Native Windows runner

The RentedNoodle path can run directly on Windows without Docker Desktop or WSL2.

## Requirements

- Windows 11
- NVIDIA RTX 5080 / 5070 Ti Blackwell GPU
- NVIDIA driver with CUDA support
- CUDA Toolkit 12.8 or newer (CUDA 13.x recommended; CUDA 13.0 is acceptable for this source tree)
- Visual Studio 2022 Build Tools or Community with **Desktop development with C++**
- Git and CMake

The build script bootstraps its own private vcpkg checkout under `.deps/` and uses the
repository's pinned `vcpkg.json` manifest for FFmpeg and curl. Docker and WSL are not used.

## Build

```powershell
git pull
.\scripts\build-native-windows.ps1
```

Output:

```text
engine\build-windows\apps\ninfer-serve.exe
```

The build applies an MSVC-only stub for the Blackwell NVFP4 TMA kernels. This does not
affect the RentedNoodle GSQ/RCO artifact, which uses NInfer's Q3/Q4/Q5 paths.

## Run

```powershell
.\scripts\run-rentednoodle.ps1
```

Native Windows is now the default. The API binds to:

```text
http://127.0.0.1:8080/v1
```

To explicitly use Docker instead:

```powershell
$env:NINFER_RUNTIME="docker"
.\scripts\run-rentednoodle.ps1
```

## RAM-conscious defaults

The native profile deliberately avoids NInfer's large server defaults:

- Host KV: **512 MiB** instead of the engine's 8192 MiB default
- host state slots: 2
- private continuations: 2
- shared prefixes: 2
- media cache: 256 MiB
- media live budget: 768 MiB
- concurrency: 1

Change Host KV if you need more persistent prefix/session capacity:

```powershell
$env:NINFER_HOST_KV_MIB="0"      # minimum RAM, little/no host KV retention
# or
$env:NINFER_HOST_KV_MIB="2048"   # more host prefix/session cache
```

## Context profiles

With Vision enabled (default):

```powershell
$env:NINFER_CONTEXT="65536"
.\scripts\run-rentednoodle.ps1
```

Text-only starts at 100K:

```powershell
$env:NINFER_VISION="0"
$env:NINFER_CONTEXT="100000"
.\scripts\run-rentednoodle.ps1
```

You can increase context after measuring actual free VRAM and artifact size.

Useful overrides:

```powershell
$env:NINFER_PORT="8100"
$env:NINFER_VISION_MAX_TOKENS="1792"
$env:NINFER_PREFILL_CHUNK="2688"
$env:NINFER_CUDA_GRAPH="0"       # troubleshooting only; graphs stay on by default
```

The launcher prints current GPU memory from `nvidia-smi` immediately before loading the model.
